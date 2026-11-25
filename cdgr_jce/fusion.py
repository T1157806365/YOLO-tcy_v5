"""Main-path joint encoder; no geometry guide, attention, or fixed residual gain."""
import torch
from torch import nn


def group_concat(xs, groups):
    """Each group sees corresponding channels of EVERY source (not modality-separated groups)."""
    b, _, h, w = xs[0].shape
    return torch.cat([x.reshape(b, groups, x.shape[1]//groups, h, w) for x in xs], 2).reshape(b,-1,h,w)


class JointEncoder(nn.Module):
    def __init__(self, channels=512, dim=64, groups=8):
        super().__init__()
        if channels % groups or dim % groups or dim<1: raise ValueError('Invalid channels/groups')
        self.channels=channels; self.dim=dim; self.groups=groups
        self.r=nn.Sequential(nn.Conv2d(channels,dim,1,bias=False),nn.GroupNorm(1,dim))
        self.t=nn.Sequential(nn.Conv2d(channels,dim,1,bias=False),nn.GroupNorm(1,dim))
        self.joint=nn.Conv2d(2*channels+4*dim,channels,1,groups=groups,bias=False)
        self.spatial=nn.Conv2d(channels,channels,3,padding=1,groups=channels,bias=False)
        # Initialize the MAIN encoder close to Add, with nonzero interaction gradients.
        # There is no separate Add bypass and no fixed 0.1 gain.
        with torch.no_grad():
            c=channels//groups
            nn.init.normal_(self.joint.weight,std=.01)
            for i in range(channels):
                self.joint.weight[i,:2*c].zero_()
                self.joint.weight[i,i%c,0,0]=1.
                self.joint.weight[i,c+i%c,0,0]=1.
            self.spatial.weight.zero_();self.spatial.weight[:,0,1,1]=1.
        self.collect_stats=False;self.last_stats={};self.intervention='normal'
    def forward(self,rgb,tir):
        if rgb.shape!=tir.shape:raise ValueError('JCE expects matching ROI feature grids')
        if self.intervention=='add':return rgb+tir
        if self.intervention=='rgb_only':return rgb
        if self.intervention=='tir_only':return tir
        if self.intervention=='zero_tir':tir=torch.zeros_like(tir)
        if self.intervention=='zero_rgb':rgb=torch.zeros_like(rgb)
        r=self.r(rgb);t=self.t(tir)
        diff=(r-t).abs();prod=r*t
        if self.intervention=='no_interaction':diff=torch.zeros_like(diff);prod=torch.zeros_like(prod)
        z=self.spatial(self.joint(group_concat((rgb,tir,r,t,diff,prod),self.groups)))
        if self.collect_stats:
            with torch.no_grad():
                c=self.channels//self.groups; w=self.joint.weight.detach().float().abs()
                den=(rgb.detach().float()+tir.detach().float()).abs().mean().clamp_min(1e-6)
                self.last_stats={'delta_over_add':(z.detach().float()-rgb.detach().float()-tir.detach().float()).abs().mean()/den,
                    'rgb_weight':w[:,:c].mean(),'tir_weight':w[:,c:2*c].mean(),
                    'interaction_weight':w[:,2*c:].mean(),'fused_abs':z.detach().float().abs().mean()}
        return z


class JointFeedback(nn.Module):
    """C/D only: fused and donor features form directional messages for the next backbone stage."""
    def __init__(self,channels=512,groups=8):
        super().__init__();self.groups=groups
        self.to_r=nn.Conv2d(2*channels,channels,1,groups=groups,bias=False)
        self.to_t=nn.Conv2d(2*channels,channels,1,groups=groups,bias=False)
        # Nonzero feedback, no fixed attenuation/gate. Main path is JointEncoder.
        for op in (self.to_r,self.to_t):nn.init.normal_(op.weight,std=.01)
        self.collect_stats=False;self.last_stats={}
    def forward(self,r,t,z):
        dr=self.to_r(group_concat((t,z),self.groups));dt=self.to_t(group_concat((r,z),self.groups))
        if self.collect_stats:
            with torch.no_grad():
                self.last_stats={'t_to_r_ratio':dr.detach().float().abs().mean()/r.detach().float().abs().mean().clamp_min(1e-6),
                    'r_to_t_ratio':dt.detach().float().abs().mean()/t.detach().float().abs().mean().clamp_min(1e-6)}
        return r+dr,t+dt
