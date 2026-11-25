"""Dependency-light CPU tests for CDGR-THRF; no Ultralytics needed."""
import unittest
import tempfile
from pathlib import Path
from copy import deepcopy
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cdgr_thrf.fusion import LocalP2Context,ThermalQueryRGBDetail,TriModalResidualFusion
from cdgr_thrf.geometry import roi_cdgr_descriptor
from cdgr_thrf.loss import make_center_heatmap,thermal_targetness_loss
from check_cdgr_thrf import check

class THRFUnitTests(unittest.TestCase):
    def _modules(self,channels=32,p2_channels=16):
        ctx=LocalP2Context()
        retrieval=ThermalQueryRGBDetail(p2_channels,8,12,3.0)
        mods={s:TriModalResidualFusion(channels,12,16,7,retrieval if s=='p3' else None,
                    ctx,scale=s) for s in ('p3','p4','p5')}
        return ctx,mods

    def test_initial_equivalence_to_original_add(self):
        torch.manual_seed(10)
        ctx,m=self._modules()
        ctx.rgb_p2=torch.randn(2,16,16,16,requires_grad=True)
        ctx.tir_p2=torch.randn(2,16,16,16,requires_grad=True)
        for s,shape in (('p3',8),('p4',4),('p5',2)):
            a=torch.randn(2,32,shape,shape,requires_grad=True)
            b=torch.randn_like(a,requires_grad=True)
            m[s].set_geometry(torch.randn(2,7))
            actual=m[s](a,b)
            self.assertTrue(torch.equal(actual,a+b),f'{s} initial ADD mismatch')
            self.assertTrue(torch.isfinite(actual).all().item())
        self.assertEqual(ctx.detail_p3.shape,(2,12,8,8))
        self.assertEqual(ctx.tir_heat_logits.shape,(2,1,8,8))

    def test_three_stream_gradient_and_nonzero_change(self):
        torch.manual_seed(20)
        ctx,m=self._modules()
        with torch.no_grad():
            for mod in m.values():
                torch.nn.init.normal_(mod.interact[-1].weight,std=0.01)
        ctx.rgb_p2=torch.randn(2,16,16,16,requires_grad=True)
        ctx.tir_p2=torch.randn(2,16,16,16,requires_grad=True)
        outputs=[]
        for s,hw in (('p3',8),('p4',4),('p5',2)):
            mod=m[s];mod.set_geometry(torch.rand(2,7))
            a=torch.randn(2,32,hw,hw,requires_grad=True)
            b=torch.randn(2,32,hw,hw,requires_grad=True)
            y=mod(a,b)
            self.assertGreater((y-a-b).abs().mean().item(),0.)
            outputs.append(y.square().mean())
        loss=sum(outputs)
        loss.backward()
        self.assertIsNotNone(ctx.rgb_p2.grad)
        self.assertIsNotNone(ctx.tir_p2.grad)
        self.assertGreater(ctx.rgb_p2.grad.abs().sum().item(),0)
        self.assertGreater(ctx.tir_p2.grad.abs().sum().item(),0)
        self.assertGreater(m['p3'].retriever.tir_query.weight.grad.abs().sum().item(),0)
        self.assertGreater(m['p3'].retriever.rgb_value.weight.grad.abs().sum().item(),0)

    def test_p2_is_real_four_token_rgb_detail_query(self):
        mod=ThermalQueryRGBDetail(p2_channels=8,key_dim=8,detail_dim=8)
        torch.manual_seed(1)
        rgb=torch.randn(2,8,16,16)
        tir1=torch.randn(2,8,16,16)
        tir2=torch.randn_like(tir1)*5
        a,_=mod(rgb,tir1)
        b,_=mod(rgb,tir2)
        self.assertGreater((a-b).abs().mean().item(),1e-5)
        self.assertTrue(0.<=mod.last_stats['attention_entropy'].item()<=1.0001)

    def test_thermal_heat_empty_and_paired_mask(self):
        logits=torch.zeros(4,1,8,8,requires_grad=True)
        labels={'rgb_bboxes':torch.tensor([[.5,.5,.10,.12]]),
                'rgb_batch_idx':torch.tensor([0])}
        loss,stats=thermal_targetness_loss(logits,labels,
            torch.tensor([True,False,False,False]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(stats['tir_heat_supervised_roi_frac'].item(),.25)
        zero_labels={'rgb_bboxes':torch.empty(0,4),'rgb_batch_idx':torch.empty(0,dtype=torch.long)}
        no_labels_loss,_=thermal_targetness_loss(logits.detach().requires_grad_(),zero_labels,
                            torch.zeros(4,dtype=torch.bool))
        self.assertEqual(no_labels_loss.item(),0.)

    def test_cdgr_descriptor_repeats_by_two_and_grad_detached(self):
        s=torch.ones(2,2,requires_grad=True)
        s0=torch.ones(2,2)
        batch={'rgb_img':torch.zeros(2,3,192,192),
               'rgb_semantic_img':torch.zeros(2,3,64,64),
               'tir_img':torch.zeros(2,3,64,64)}
        info={'sampling_scale':s,'sampling_translation':torch.zeros(2,2),
              'sampling_scale_raw_gtga':s0,'sampling_translation_raw_gtga':torch.zeros(2,2),
              'coarse_anchor':torch.full((2,2),.5)}
        class Geo:pass
        geo=Geo();geo.tir_valid=torch.ones(4)
        ex={'alignment_info':info,'rois_per_image':2,'roi':geo}
        value=roi_cdgr_descriptor(batch,ex)
        self.assertEqual(tuple(value.shape),(4,7))
        self.assertFalse(value.requires_grad)
        self.assertTrue(torch.equal(value[0],value[1]))

    def test_config_integrity_uavcb_antiuav(self):
        from install_cdgr_thrf import DATASETS,config_from_baseline,OUT_CFG
        for kind,(orig,name,_,_) in DATASETS.items():
            src=ROOT/'original_reference_configs'/orig
            produced=config_from_baseline(src,kind)
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/name;p.write_bytes(produced)
                check(p,src)

if __name__=='__main__':unittest.main()
