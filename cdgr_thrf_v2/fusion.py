"""P3-only, phase-preserving thermal-query RGB detail retrieval.
Pixel-unshuffle preserves all four P2 phases before any learned compression.
Local attention mixes neighboring P3 cells separately for each phase.
No independent ADD bypass in the P3 fusion path.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

class PhasePreservingRetrieval(nn.Module):
    def __init__(self,p2_channels=256,p3_channels=512,key_dim=16,value_dim=24,
                 radius=1,temperature=4.,geo_sigma=1.):
        super().__init__()
        if radius not in (0,1,2) or min(key_dim,value_dim)<1 or min(temperature,geo_sigma)<=0:
            raise ValueError('Invalid retrieval dimensions')
        self.p2_channels=p2_channels;self.p3_channels=p3_channels
        self.key_dim=key_dim;self.value_dim=value_dim;self.radius=radius
        self.temperature=temperature;self.geo_sigma=geo_sigma
        self.offsets=[(dy,dx) for dy in range(-radius,radius+1) for dx in range(-radius,radius+1)]
        self.q2=nn.Conv2d(p2_channels,key_dim,1,bias=False)
        self.q3=nn.Conv2d(p3_channels,key_dim,1,bias=False)
        self.k=nn.Conv2d(p2_channels,key_dim,1,bias=False)
        # Full native RGB P2, not only its high-pass component.
        self.v=nn.Conv2d(p2_channels,value_dim,1,bias=False)
        self.heat=nn.Sequential(nn.Conv2d(key_dim,16,1),nn.SiLU(),nn.Conv2d(16,1,1))
        self.geo_bias=nn.Sequential(nn.Linear(7,16),nn.SiLU(),nn.Linear(16,len(self.offsets)))
        nn.init.zeros_(self.geo_bias[-1].weight);nn.init.zeros_(self.geo_bias[-1].bias)
        self.phase_bias=nn.Parameter(torch.zeros(4))
        self.last_stats={}

    @staticmethod
    def phases(x):
        b,c,h,w=x.shape
        if h%2 or w%2:raise ValueError('P2 shape must be even')
        return F.pixel_unshuffle(x,2).reshape(b,c,4,h//2,w//2).permute(0,2,1,3,4)

    def forward(self,r2,t2,t3,geo):
        if r2.shape!=t2.shape or r2.shape[1]!=self.p2_channels:
            raise ValueError('Unexpected RGB/TIR P2 shapes')
        b,_,h,w=t3.shape
        if t3.shape[1]!=self.p3_channels or r2.shape!=(b,self.p2_channels,2*h,2*w) or geo.shape!=(b,7):
            raise ValueError('P2/P3/CDGR geometry shape mismatch')
        q_raw=self.q2(F.avg_pool2d(t2,2,2))+self.q3(t3)
        heat=self.heat(q_raw)
        q=F.normalize(q_raw.float(),dim=1)
        k=F.normalize(self.phases(self.k(r2)).float(),dim=2)
        v=self.phases(self.v(r2))
        radius=self.radius;N=len(self.offsets)
        kp=F.pad(k,(radius,radius,radius,radius))
        vp=F.pad(v,(radius,radius,radius,radius))
        valid=F.pad(torch.ones(1,1,h,w,device=q.device,dtype=torch.bool),(radius,radius,radius,radius),value=False)
        geo_bias=self.geo_bias(geo.float()).float()
        scores=[]
        for j,(dy,dx) in enumerate(self.offsets):
            y=radius+dy;x=radius+dx
            key=kp[...,y:y+h,x:x+w]
            sim=(q[:,None]*key).sum(2)*self.temperature
            bias=-(dy*dy+dx*dx)/(2*self.geo_sigma**2)
            sim=sim+bias+geo_bias[:,j,None,None,None]+self.phase_bias[None,:,None,None]
            scores.append(sim.masked_fill(~valid[...,y:y+h,x:x+w],-1e4))
        logits=torch.stack(scores,dim=2) # B,4,9,H,W
        # One normalized neighborhood distribution PER phase.  No phase collapse.
        a=logits.softmax(2)
        phase=(logits.logsumexp(2)).softmax(1)
        # Keep a positive floor for each phase; do not hard-discard any phase.
        phase_gain=0.5+2.*phase
        retrieved=torch.zeros_like(v)
        for j,(dy,dx) in enumerate(self.offsets):
            y=radius+dy;x=radius+dx
            retrieved=retrieved+a[:,:,j,None].to(v.dtype)*vp[...,y:y+h,x:x+w]
        retrieved=retrieved*phase_gain[:,:,None].to(v.dtype)
        # Preserve the central projected detail alongside retrieved detail.
        # B, 4*(2*value_dim), H,W: ordered TL/TR/BL/BR, each central+retrieved.
        detail=torch.cat((v,retrieved),dim=2).reshape(b,8*self.value_dim,h,w)
        with torch.no_grad():
            ent=-(a*a.clamp_min(1e-8).log()).sum(2)
            self.last_stats={
                'neighbor_entropy':ent.mean()/max(math.log(N),1e-8),
                'neighbor_peak':a.amax(2).mean(),
                'center_attention':a[:,:,N//2].mean(),
                'phase_peak':phase.amax(1).mean(),
                'detail_abs':detail.abs().mean(),
                'tir_heat_mean':heat.float().sigmoid().mean(),
            }
        return detail,heat

class P3JointFusion(nn.Module):
    """Separate R,T,D streams; single grouped learned projection is the main path.
    Group-wise input order [R_g,T_g,D_g], near-ADD but nonzero D init.
    The grouped 1x1 avoids a large dense 1216->512 projection at 80x80.
    """
    def __init__(self,channels=512,detail_channels=192,groups=8,detail_init=0.05):
        super().__init__()
        if channels%groups or detail_channels%groups or detail_init<=0:
            raise ValueError('Invalid fusion groups/init')
        self.channels=channels;self.detail_channels=detail_channels;self.groups=groups
        self.mix=nn.Conv2d(2*channels+detail_channels,channels,1,groups=groups,bias=False)
        cg=channels//groups;dg=detail_channels//groups
        with torch.no_grad():
            self.mix.weight.zero_()
            for g in range(groups):
                for i in range(cg):
                    self.mix.weight[g*cg+i,i,0,0]=1.
                    self.mix.weight[g*cg+i,cg+i,0,0]=1.
            nn.init.normal_(self.mix.weight[:,2*cg:,:,:],std=detail_init/math.sqrt(dg))
        self.last_stats={}
    def forward(self,r,t,d):
        if r.shape!=t.shape or r.shape[1]!=self.channels or d.shape!=(r.shape[0],self.detail_channels,*r.shape[-2:]):
            raise ValueError('Invalid P3 triplet')
        b,c,h,w=r.shape;g=self.groups
        # Every output group receives all three streams, not just one modality.
        # Interleave phase channels across groups so each group sees each phase.
        d=d.reshape(b,4,self.detail_channels//4,h,w).transpose(1,2).reshape(b,self.detail_channels,h,w)
        x=torch.cat((r.reshape(b,g,c//g,h,w),t.reshape(b,g,c//g,h,w),
                     d.reshape(b,g,self.detail_channels//g,h,w)),2).reshape(b,-1,h,w)
        out=self.mix(x)
        with torch.no_grad():
            self.last_stats={'delta_over_add':(out-r-t).abs().mean()/(r.abs().mean()+t.abs().mean()).clamp_min(1e-6),
                             'rgb_abs':r.abs().mean(),'tir_abs':t.abs().mean()}
        return out

class P2Context:
    def __init__(self):self.clear()
    def clear(self):self.rgb=None;self.tir=None;self.geo=None;self.heat=None

class THRFV2P3(nn.Module):
    def __init__(self,context,p2_channels=256,p3_channels=512,key_dim=16,value_dim=24,
                 radius=1,temperature=4.,geo_sigma=1.,groups=8,detail_init=.05):
        super().__init__();self.context=context
        self.retrieval=PhasePreservingRetrieval(p2_channels,p3_channels,key_dim,value_dim,radius,temperature,geo_sigma)
        self.joint=P3JointFusion(p3_channels,8*value_dim,groups,detail_init)
        self.last_stats={}
    def forward(self,r,t):
        ctx=self.context
        if any(x is None for x in (ctx.rgb,ctx.tir,ctx.geo)):
            raise RuntimeError('Missing local P2/CDGR context')
        d,ctx.heat=self.retrieval(ctx.rgb,ctx.tir,t,ctx.geo)
        out=self.joint(r,t,d)
        self.last_stats={**self.retrieval.last_stats,**self.joint.last_stats}
        return out
