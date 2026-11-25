import copy,io,unittest
import torch
from torch import nn
from cdgr_sw.sharing import share_pair,assert_pair,count,compact_cpu_state_dict

torch.set_num_threads(1)
class Block(nn.Module):
    def __init__(self):
        super().__init__();self.conv=nn.Conv2d(4,8,3,padding=1,bias=False);self.bn=nn.BatchNorm2d(8);self.act=nn.SiLU()
    def forward(self,x):return self.act(self.bn(self.conv(x)))
class Pair(nn.Module):
    def __init__(self):
        super().__init__();self.g=nn.ModuleList([Block()]);self.l=copy.deepcopy(self.g)
    def forward(self,g,l):return self.g[0](g),self.l[0](l)
class TestSharing(unittest.TestCase):
    def test_identity_and_count(self):
        n=Pair();before=count(n);saved=share_pair(n.g,n.l)
        self.assertIs(n.g[0].conv,n.l[0].conv);self.assertIsNot(n.g[0].bn,n.l[0].bn)
        self.assertEqual(before-count(n),saved);self.assertEqual(saved,4*8*9)
        self.assertIsNot(n.g[0].bn.running_mean,n.l[0].bn.running_mean)
    def test_bn_independent(self):
        n=Pair();share_pair(n.g,n.l);n.train()
        with torch.no_grad():n.g[0].conv.weight.fill_(1);n(torch.ones(4,4,8,8)*10,torch.zeros(4,4,8,8))
        self.assertGreater(n.g[0].bn.running_mean.abs().sum().item(),0);self.assertEqual(n.l[0].bn.running_mean.abs().sum().item(),0)
    def test_grad_sum_and_optimizer(self):
        n=Pair().eval();share_pair(n.g,n.l);p=n.l[0].conv.weight
        x=torch.randn(2,4,8,8);y=torch.randn_like(x);a,b=n(x,y)
        ga=torch.autograd.grad(a.square().mean(),p,retain_graph=True)[0];gb=torch.autograd.grad(b.square().mean(),p,retain_graph=True)[0]
        (a.square().mean()+b.square().mean()).backward();torch.testing.assert_close(p.grad,ga+gb)
        opt=torch.optim.SGD(n.parameters(),lr=.01);ids=[id(p) for g in opt.param_groups for p in g['params']];self.assertEqual(len(ids),len(set(ids)));self.assertEqual(ids.count(id(p)),1)
        before=p.detach().clone();opt.step();self.assertFalse(torch.equal(before,p))
    def test_dtype_and_deepcopy(self):
        n=Pair();share_pair(n.g,n.l);n.double();assert_pair(n.g,n.l,True)
        cloned=copy.deepcopy(n);assert_pair(cloned.g,cloned.l,True);self.assertIsNot(cloned.g[0].conv,n.g[0].conv)
    def test_compact_strict_roundtrip(self):
        n=Pair();share_pair(n.g,n.l);n.g[0].bn.running_mean.fill_(3)
        s=compact_cpu_state_dict(n);self.assertIs(s['g.0.conv.weight'],s['l.0.conv.weight']);self.assertIsNot(s['g.0.bn.running_mean'],s['l.0.bn.running_mean'])
        f=io.BytesIO();torch.save(s,f);f.seek(0);state=torch.load(f,weights_only=True)
        self.assertEqual(state['g.0.conv.weight'].data_ptr(),state['l.0.conv.weight'].data_ptr())
        m=Pair();share_pair(m.g,m.l);m.load_state_dict(state,strict=True);assert_pair(m.g,m.l,True)
        n.eval();m.eval();x=torch.randn(2,4,8,8)
        for a,b in zip(n(x,x),m(x,x)):torch.testing.assert_close(a,b)
    def test_incompatible_fail_before_write(self):
        n=Pair();n.l[0].conv=nn.Conv2d(4,8,1,bias=False);old=n.g[0].conv
        with self.assertRaises(RuntimeError):share_pair(n.g,n.l)
        self.assertIs(n.g[0].conv,old)
    def test_disabled(self):
        n=Pair();self.assertEqual(share_pair(n.g,n.l,False),0);self.assertIsNot(n.g[0].conv,n.l[0].conv)
if __name__=='__main__':unittest.main()
