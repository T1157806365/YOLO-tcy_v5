"""HVAD-v1: high-resolution evidence verified geometric hypothesis routing.

Works with the existing centered GTGA/GMLS-v2 alignment and P3 Tri-Fusion.

Algorithm:
  (1) Reuse GMLS four candidate translations [B,K,2] and priors.
  (2) Use the real BHLR detail_stem output as a high-resolution evidence bank.
  (3) Compare RGB and candidate-warped TIR patches around the GTGA anchor.
  (4) Predict candidate scores; choose exactly one hypothesis via STE argmax.
  (5) Resample one full TIR P3 map with the chosen translation, retaining
      GTGA's already-estimated scale and Fine Translation residual.
  (6) Add a zero-initialized, bounded correction to the existing P3 fusion.

The classifier sees neither ground truth nor labels at inference. A rank loss
can supervise hypotheses using paired RGB/TIR boxes during training only.
"""
from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from models.modules.alignment.rlsfa_gtga_centered import warp_with_geometry


class HVADGeoHR(nn.Module):
    def __init__(self, channels=512, detail_channels=48, hidden=24,
                 roi_size=11, roi_radius=6, max_gain=0.25,
                 prior_strength=0.5, max_gt_offset=20.0,
                 lambda_rank=0.03, use_hr_evidence=True):
        super().__init__()
        self.channels = int(channels)
        self.detail_channels = int(detail_channels)
        self.hidden = int(hidden)
        self.roi_size = int(roi_size)
        self.roi_radius = float(roi_radius)
        self.max_gain = float(max_gain)
        self.prior_strength = float(prior_strength)
        self.max_gt_offset = float(max_gt_offset)
        self.lambda_rank = float(lambda_rank)
        self.use_hr_evidence = bool(use_hr_evidence)
        # V5-GEOHR-SPATIAL-V1: plain Python attributes, NOT state_dict parameters.
        # Defaults reproduce the original HVAD forward exactly.
        self.spatial_hr_roi = False
        self.hr_roi_mix = 1.0
        self.confidence_gate_enabled = False
        self.confidence_gate_threshold = 0.50
        if self.roi_size < 3 or self.roi_size % 2 != 1:
            raise ValueError('hvad.roi_size must be an odd integer >= 3')
        if self.channels <= 0 or self.hidden <= 0:
            raise ValueError('HVAD channels/hidden must be positive')

        c = self.hidden
        self.rgb_proj = nn.Conv2d(self.channels, c, 1, bias=False)
        self.tir_proj = nn.Conv2d(self.channels, c, 1, bias=False)
        self.hr_proj = nn.Sequential(
            nn.Conv2d(self.detail_channels, c, 1, bias=False),
            nn.SiLU(inplace=False),
        )
        self.score_net = nn.Sequential(
            nn.Linear(c * 5 + 2, 64), nn.SiLU(inplace=False),
            nn.Linear(64, 1),
        )
        # At init retain original GMLS candidate prior, not random scores.
        nn.init.zeros_(self.score_net[-1].weight)
        nn.init.zeros_(self.score_net[-1].bias)

        self.gate_net = nn.Sequential(
            nn.Conv2d(c * 4 + 1, c, 3, padding=1),
            nn.SiLU(inplace=False), nn.Conv2d(c, 1, 1),
        )
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.zeros_(self.gate_net[-1].bias)
        self.delta_tir = nn.Conv2d(self.channels, self.channels, 1, bias=False)
        self.delta_hr = nn.Conv2d(c, self.channels, 1, bias=False)
        self.gamma_raw = nn.Parameter(torch.zeros(()))

        # Runtime-only state (not checkpoint parameters/buffers).
        self._detail_bank = None
        self.last_aux = None
        self.last_stats = {}
        self.enabled_runtime = True
        self.rank_enabled = True

    def capture_detail(self, _module, _inputs, output):
        if not torch.is_tensor(output):
            raise TypeError('BHLR detail_stem must return a tensor')
        self._detail_bank = output

    def _patch_grid(self, anchor, h, w, dtype, device):
        """RGB-coordinate ROI in normalized [0,1], B×P×P×2."""
        b = int(anchor.shape[0])
        r = torch.linspace(-self.roi_radius, self.roi_radius,
                           self.roi_size, dtype=dtype, device=device)
        yy, xx = torch.meshgrid(r, r, indexing='ij')
        offsets = torch.stack((xx / max(w - 1, 1),
                               yy / max(h - 1, 1)), dim=-1)
        return anchor.to(dtype=dtype).reshape(b, 1, 1, 2) + offsets[None]

    @staticmethod
    def _sample(x, coord01):
        return F.grid_sample(x, coord01 * 2 - 1, mode='bilinear',
                             padding_mode='border', align_corners=True)

    def forward(self, features: dict, detail_bank: torch.Tensor, scale_index: int):
        if not self.enabled_runtime:
            self.last_aux = None
            self.last_stats = {'active': 0.0}
            return features

        info = features['alignment_info']
        offsets = info.get('coarse_gmls_candidate_offsets', None)
        priors = info.get('coarse_gmls_candidate_weights', None)
        if offsets is None or priors is None:
            raise RuntimeError('HVAD requires GMLS candidates/weights in GTGA alignment_info. '
                               'Check coarse_gmls_enabled: true.')
        if offsets.ndim != 3 or offsets.shape[-1] != 2:
            raise ValueError(f'GMLS candidates must be [B,K,2], got {offsets.shape}')
        b, k, _ = offsets.shape
        if k < 2:
            raise ValueError('HVAD requires at least two geometric hypotheses')
        idx = int(scale_index)
        fused_old = features['fused_backbone'][idx]
        r = features['rgb_backbone_raw'][idx]
        t = features['tir_backbone'][idx]
        aligned_old = features['aligned_tir_backbone'][idx]
        if r.shape != t.shape or fused_old.shape != aligned_old.shape:
            raise ValueError('HVAD requires aligned, equal-channel RGB/TIR P3 features')
        if r.shape[1] != self.channels:
            raise ValueError(f'P3 expected {self.channels} channels but got {r.shape[1]}')
        if detail_bank is None:
            raise RuntimeError('HVAD BHLR detail hook did not run: ensure BHLR enabled and RGB high != semantic')
        if detail_bank.shape[1] != self.detail_channels:
            raise ValueError(f'BHLR detail channels {detail_bank.shape[1]} != {self.detail_channels}')

        h, w = r.shape[-2:]
        rp = self.rgb_proj(r)
        tp = self.tir_proj(t)
        hr = self.hr_proj(F.adaptive_avg_pool2d(detail_bank, (h, w)))
        if not self.use_hr_evidence:
            hr_for_score = hr.detach() * 0.0
        else:
            hr_for_score = hr

        anchor = info['coarse_anchor'].reshape(b, 2).clamp(0.0, 1.0)
        scale = info['sampling_scale'].reshape(b, 2)
        translation = info['sampling_translation'].reshape(b, 2)
        # GMLS offsets in P3 pixels, final selected coarse offset (pre fine)
        # in same pixel units. Keep original GTGA geometry fixed during
        # candidate ranking to avoid selector disrupting alignment itself.
        candidate = offsets.detach().float()
        coarse_base = info['coarse_offset_initial'].reshape(b, 2).detach().float()
        ref_size = candidate.new_tensor([max(w-1, 1), max(h-1, 1)])
        delta = candidate - coarse_base[:, None, :]
        base01 = self._patch_grid(anchor.detach(), h, w, rp.dtype, rp.device)
        rr = self._sample(rp, base01)
        hh_pooled = self._sample(hr_for_score, base01)
        if self.spatial_hr_roi and self.use_hr_evidence:
            # Sample BHLR's native 640x640 detail bank directly at P3 ROI
            # coordinates. Its channels contain 1920-RGB pixel-unshuffle detail.
            # Avoid projecting the full 640x640 bank (unnecessary VRAM).
            native_roi = self._sample(
                detail_bank, base01.to(dtype=detail_bank.dtype)
            )
            hh_native = self.hr_proj(native_roi)
            hh = hh_pooled + self.hr_roi_mix * (hh_native - hh_pooled)
            spatial_gap = (hh_native.float() - hh_pooled.float()).detach().abs().mean()
        else:
            hh = hh_pooled
            spatial_gap = hh_pooled.float().new_zeros(())
        roi_scores = []
        for ci in range(k):
            source01 = (
                base01.float() * scale.detach().float()[:, None, None, :]
                + translation.detach().float()[:, None, None, :]
                + delta[:, ci, :][:, None, None, :] / ref_size
            )
            tt = self._sample(tp, source01.to(dtype=tp.dtype))
            mismatch = (rr - tt).abs()
            evidence = torch.cat((rr, tt, mismatch, hh,
                                  mismatch * torch.sigmoid(hh)), dim=1)
            pooled = evidence.mean(dim=(2, 3))
            dnorm = (delta[:, ci, :] / float(max(h, w))).to(dtype=pooled.dtype)
            roi_scores.append(self.score_net(torch.cat((pooled, dnorm), dim=1)))
        residual_logits = torch.cat(roi_scores, dim=1)
        prior = priors.detach().float().clamp_min(1e-6)
        logits = residual_logits.float() + self.prior_strength * prior.log()
        probs = F.softmax(logits, dim=1)
        winner = logits.argmax(dim=1)
        hot = F.one_hot(winner, num_classes=k).to(dtype=probs.dtype)
        ste = hot - probs.detach() + probs  # hard forward, soft backward
        selected = (ste[:, :, None] * candidate).sum(dim=1)
        chosen_delta = (selected - coarse_base) / ref_size
        trans_selected = translation + chosen_delta.to(dtype=translation.dtype)

        tir_selected = warp_with_geometry(
            t, scale, trans_selected, output_hw=(h, w),
            padding_mode='border', align_corners=True,
        )
        selected_tir_proj = self.tir_proj(tir_selected)
        support = info.get('rgb_target_support')
        if support is None:
            support = r.new_zeros(b, 1, h, w)
        if support.shape[-2:] != (h, w):
            support = F.interpolate(support, size=(h, w), mode='bilinear', align_corners=False)
        support = support.to(dtype=rp.dtype).clamp(0, 1)
        gate_input = torch.cat((rp, selected_tir_proj,
                                (rp - selected_tir_proj).abs(), hr_for_score,
                                support), dim=1)
        gate = torch.sigmoid(self.gate_net(gate_input))
        # A bounded, residual-only correction. At initialization gamma=0, so
        # full forward == original GTGA+BHLR+TriFusion+P2 (bitwise P3).
        correction = (self.delta_tir(tir_selected - aligned_old)
                      + 0.10 * self.delta_hr(hr_for_score))
        gain = self.max_gain * torch.tanh(self.gamma_raw)
        if self.confidence_gate_enabled:
            # Uncertain hypothesis selection cannot dominate the baseline.
            maxprob = probs.amax(dim=1)
            threshold = float(self.confidence_gate_threshold)
            guard = ((maxprob - threshold) / (1.0 - threshold)).clamp(0.0, 1.0)
            guard = guard.to(dtype=gate.dtype).reshape(b, 1, 1, 1)
        else:
            guard = torch.ones((b, 1, 1, 1), dtype=gate.dtype, device=gate.device)
        fused_new = fused_old + gain * gate * guard * correction
        fused = list(features['fused_backbone'])
        fused[idx] = fused_new
        features['fused_backbone'] = fused
        fusion_features = dict(features.get('fusion_features', {}))
        fusion_features[idx] = fused_new
        features['fusion_features'] = fusion_features

        self.last_aux = {
            'logits': logits,
            'offsets': candidate,
            'h': int(h), 'w': int(w),
            'winner': winner,
        }
        with torch.no_grad():
            self.last_stats = {
                'active': 1.0,
                'hvad_gain': float(gain.float().abs().item()),
                'hvad_gate_mean': float(gate.detach().float().mean().item()),
                'hvad_fusion_delta': float((fused_new-fused_old).detach().float().abs().mean().item()),
                'hvad_choice_global': float((winner == 0).float().mean().item()),
                'hvad_choice_local': float((winner != 0).float().mean().item()),
                'hvad_entropy': float((-(probs * probs.clamp_min(1e-8).log()).sum(1)).mean().item()),
                'hvad_hr_spatial_gap': float(spatial_gap.item()),
                'hvad_conf_maxprob': float(probs.amax(dim=1).detach().float().mean().item()),
                'hvad_conf_guard_mean': float(guard.detach().float().mean().item()),
            }
        return features

    def ranking_loss(self, batch: Mapping[str, torch.Tensor]):
        aux = self.last_aux
        if aux is None:
            raise RuntimeError('HVAD ranking loss requested before HVAD forward')
        logits = aux['logits']
        candidates = aux['offsets']
        b, k = logits.shape
        h, w = aux['h'], aux['w']
        rgb_bboxes = batch['rgb_semantic_bboxes']
        tir_bboxes = batch['tir_bboxes']
        rgb_indices = batch['rgb_batch_idx'].reshape(-1)
        tir_indices = batch['tir_batch_idx'].reshape(-1)
        losses = []
        correct = []
        oracle_px = []
        for bi in range(b):
            ri = rgb_indices == bi
            ti = tir_indices == bi
            if int(ri.sum().item()) != 1 or int(ti.sum().item()) != 1:
                continue  # Avoid ambiguous GT pair assignments.
            rb = rgb_bboxes[ri][0].float()
            tb = tir_bboxes[ti][0].float()
            gt = torch.stack(((tb[0] - rb[0]) * max(w - 1, 1),
                              (tb[1] - rb[1]) * max(h - 1, 1)))
            gt = gt.clamp(-self.max_gt_offset, self.max_gt_offset)
            distances = (candidates[bi].float() - gt[None]).square().sum(-1)
            label = distances.argmin().view(1)
            losses.append(F.cross_entropy(logits[bi:bi+1].float(), label))
            correct.append((aux['winner'][bi] == label[0]).detach().float())
            oracle_px.append(distances.min().detach().clamp_min(0).sqrt())
        rank = torch.stack(losses).mean() if losses else logits.sum() * 0.0
        if correct:
            accuracy = torch.stack(correct).mean()
            oracle = torch.stack(oracle_px).mean()
        else:
            accuracy = logits.new_zeros(())
            oracle = logits.new_zeros(())
        return rank, {
            'hvad_rank_loss': rank.detach(),
            'hvad_rank_accuracy': accuracy.detach(),
            'hvad_oracle_err_px': oracle.detach(),
            'hvad_valid_pairs': float(len(losses)),
            **{k: float(v) for k, v in self.last_stats.items()},
        }
