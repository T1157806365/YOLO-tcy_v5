"""Same-grid progressive complement, preserving both native residual streams."""
import torch
from torch import nn
import torch.nn.functional as F

class ProgressiveComplement(nn.Module):
    def __init__(self, channels=512, dim=64, directional=True, geometry=True, gain=0.1):
        super().__init__()
        self.directional=bool(directional); self.geometry=bool(geometry)
        self.gain=float(gain); self.intervention='normal'; self.collect_stats=False; self.last_stats={}
        if dim<1 or not 0<self.gain<=1: raise ValueError('Invalid GPCF dim/gain')
        self.r=nn.Conv2d(channels,dim,1,bias=False)
        self.t=nn.Conv2d(channels,dim,1,bias=False)
        # Normalize the candidate common space independently from the baseline.
        self.rnorm=nn.GroupNorm(1,dim); self.tnorm=nn.GroupNorm(1,dim)
        self.joint=nn.Sequential(nn.Conv2d(3*dim,dim,1,bias=False),nn.GroupNorm(1,dim),nn.SiLU(),
                                 nn.Conv2d(dim,dim,3,padding=1,groups=dim,bias=False),nn.GroupNorm(1,dim),nn.SiLU())
        if directional:
            self.gate_r=nn.Conv2d(3*dim,dim,1)
            self.gate_t=nn.Conv2d(3*dim,dim,1)
            self.out_r=nn.Conv2d(dim,channels,1,bias=False)
            self.out_t=nn.Conv2d(dim,channels,1,bias=False)
            for g in (self.gate_r,self.gate_t):nn.init.zeros_(g.weight);nn.init.zeros_(g.bias)
            for g in (self.out_r,self.out_t):nn.init.normal_(g.weight,std=1e-3)
        else:
            self.out=nn.Conv2d(dim,channels,1,bias=False);nn.init.normal_(self.out.weight,std=1e-3)

    def forward(self,rgb,tir,guide=None):
        if rgb.shape!=tir.shape:raise ValueError('GPCF needs same-grid RGB/TIR feature shapes')
        if self.intervention=='off':return rgb,tir
        r=self.rnorm(self.r(rgb));t=self.tnorm(self.t(tir));diff=(r-t).abs()
        joint=self.joint(torch.cat((r,t,diff),1))
        if self.directional:
            # Donor features supply actual values; recipient and joint features condition them.
            gr=torch.sigmoid(self.gate_r(torch.cat((r,t,joint),1)))
            gt=torch.sigmoid(self.gate_t(torch.cat((t,r,joint),1)))
            dr=self.out_r(t*gr);dt=self.out_t(r*gt)
        else:
            dr=dt=self.out(joint);gr=gt=None
        if self.geometry and self.intervention!='no_geometry':
            if guide is None or guide.shape!=(rgb.shape[0],4,*rgb.shape[-2:]):raise RuntimeError('Missing GPCF geometry context')
            # [RGB support, TIR support, joint support, source validity].
            # Joint support is a heuristic, not a calibrated alignment confidence.
            guide=guide.detach().to(dtype=dr.dtype)
            valid=guide[:,3:4];joint_support=guide[:,2:3]
            weight=valid*(0.5+0.5*joint_support)
        else: weight=1.
        dr=dr*self.gain*weight;dt=dt*self.gain*weight
        if self.intervention=='no_t_to_r':dr=torch.zeros_like(dr)
        if self.intervention=='no_r_to_t':dt=torch.zeros_like(dt)
        if self.collect_stats:
            with torch.no_grad():
                eps=1e-6
                self.last_stats={'t_to_r_ratio':dr.detach().float().abs().mean()/(rgb.detach().float().abs().mean()+eps),
                                 'r_to_t_ratio':dt.detach().float().abs().mean()/(tir.detach().float().abs().mean()+eps),
                                 'geo_weight':weight.detach().float().mean() if torch.is_tensor(weight) else rgb.new_tensor(1.).float(),
                                 'gate_t_to_r':gr.detach().float().mean() if gr is not None else rgb.new_tensor(1.).float(),
                                 'gate_r_to_t':gt.detach().float().mean() if gt is not None else rgb.new_tensor(1.).float()}
        return rgb+dr,tir+dt
