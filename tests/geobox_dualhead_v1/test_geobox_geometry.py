import unittest
import torch
from geobox_dualhead_v1.geometry import final_box_geometry_loss, _iou_giou_xywh


class GeometryTests(unittest.TestCase):
    def batch(self, r, t, ri=None, ti=None):
        ri = [0] if ri is None else ri
        ti = [0] if ti is None else ti
        return dict(rgb_semantic_bboxes=torch.tensor(r, dtype=torch.float32).reshape(-1,4),
                    tir_bboxes=torch.tensor(t, dtype=torch.float32).reshape(-1,4),
                    rgb_batch_idx=torch.tensor(ri, dtype=torch.int64),
                    tir_batch_idx=torch.tensor(ti, dtype=torch.int64),
                    rgb_semantic_img=torch.zeros(1,3,640,640))

    def test_identity_zero_loss_and_grad(self):
        b = self.batch([[.45,.53,.08,.05]],[[.45,.53,.08,.05]])
        s = torch.ones(1,2,requires_grad=True)
        t = torch.zeros(1,2,requires_grad=True)
        size,giou,n,iou,err = final_box_geometry_loss(b,{'sampling_scale':s,'sampling_translation':t})
        self.assertEqual(n,1)
        self.assertAlmostEqual(iou.item(),1.,places=5)
        self.assertAlmostEqual(size.item(),0.,places=5)
        self.assertAlmostEqual(giou.item(),0.,places=5)
        (size+giou).backward()
        self.assertTrue(torch.isfinite(s.grad).all())
        self.assertTrue(torch.isfinite(t.grad).all())

    def test_scale_and_translation_have_gradients(self):
        b = self.batch([[.45,.53,.08,.05]],[[.52,.56,.1,.08]])
        s = torch.tensor([[1.1,1.05]],requires_grad=True)
        t = torch.tensor([[.01,-.01]],requires_grad=True)
        size,giou,n,iou,err = final_box_geometry_loss(b,{'sampling_scale':s,'sampling_translation':t})
        (size+.5*giou).backward()
        self.assertEqual(n,1)
        self.assertTrue(torch.isfinite(s.grad).all())
        self.assertTrue(torch.isfinite(t.grad).all())
        self.assertGreater(s.grad.abs().sum().item(),0)
        self.assertGreater(t.grad.abs().sum().item(),0)

    def test_empty_batch_graph_safe(self):
        b=self.batch([],[],[],[])
        s=torch.ones(1,2,requires_grad=True)
        t=torch.zeros(1,2,requires_grad=True)
        size,giou,n,iou,err=final_box_geometry_loss(b,{'sampling_scale':s,'sampling_translation':t})
        self.assertEqual(n,0)
        (size+giou).backward()
        self.assertTrue(torch.isfinite(s.grad).all())
        self.assertTrue(torch.isfinite(t.grad).all())

    def test_matching_transform(self):
        rgb = [.3,.6,.04,.06]
        scale=[1.2,.8]
        shift=[.1,.05]
        tir=[rgb[0]*scale[0]+shift[0],rgb[1]*scale[1]+shift[1],rgb[2]*scale[0],rgb[3]*scale[1]]
        b=self.batch([rgb],[tir])
        size,giou,n,iou,err=final_box_geometry_loss(b,{'sampling_scale':torch.tensor([scale]), 'sampling_translation':torch.tensor([shift])})
        self.assertAlmostEqual(float(iou),1.,places=5)
        self.assertAlmostEqual(float(giou),0.,places=5)

    def test_nonoverlap(self):
        i,g=_iou_giou_xywh(torch.tensor([[.05,.05,.02,.03]]),torch.tensor([[.9,.9,.1,.1]]))
        self.assertAlmostEqual(i.item(),0.,places=6)
        self.assertTrue(-1.<=g.item()<=0.)


if __name__ == '__main__':unittest.main()
