"""Identity-forward, scaled-backward bridge; no project source modifications."""
from __future__ import annotations

import inspect
import torch


class _ScaleGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, eta):
        ctx.eta = float(eta)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.eta, None


def scale_gradient(x, eta):
    if not 0.0 <= float(eta) <= 1.0:
        raise ValueError('bridge eta must be in [0, 1]')
    return _ScaleGradient.apply(x, float(eta)) if x.requires_grad else x


class GuidanceBridge:
    """Scale only BHLR -> alignment gradients, preserving all forward values.

    1. Raw RGB/TIR/joint support enters the existing 3-channel guidance fusion.
       Scaling is BEFORE that trainable fusion, so its own gradients are full.
    2. The aligned TIR feature entering a BHLR adapter retains its original gradient.
       The SAME feature's direct Add-fusion route is not modified.
    3. Raw support/geometry aliases in external_guidance are scaled if consumed.
       Learned support and guidance residual logits are NOT scaled twice.
    """
    def __init__(self, core, eta=0.0):
        if hasattr(core, '_module_only_bridge'):
            raise RuntimeError('Bridge already installed')
        required = ('alignment', 'bhlr', 'alignment_guidance_fusion',
                    'bhlr_detach_alignment_guidance')
        if any(not hasattr(core, n) for n in required):
            raise RuntimeError('Expected original three-channel guidance + BHLR core')
        if not len(core.alignment_guidance_fusion):
            raise RuntimeError('No original alignment_guidance_fusion modules')
        self.core = core
        self.handles = []
        self.flags = []
        self.support_calls = 0
        self.adapter_calls = 0
        self.set_eta(eta)
        self._set_flag(core, 'bhlr_detach_alignment_guidance', False)
        for mod in core.bhlr.modules():
            # These are Python flags, never checkpoint tensors.
            if hasattr(mod, 'detach_external_guidance'):
                self._set_flag(mod, 'detach_external_guidance', False)
        for mod in core.alignment_guidance_fusion.values():
            self.handles.append(mod.register_forward_pre_hook(self._support_hook))
        for adapter in core.bhlr.adapters.values():
            sig = inspect.signature(adapter.forward)
            if 'tir_feat' not in sig.parameters or 'external_guidance' not in sig.parameters:
                self.close()
                raise RuntimeError('Unrecognized BHLR adapter.forward signature')
            self.handles.append(adapter.register_forward_pre_hook(
                self._make_adapter_hook(sig), with_kwargs=True))
        object.__setattr__(core, '_module_only_bridge', self)

    def _set_flag(self, obj, name, value):
        self.flags.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def set_eta(self, eta):
        eta = float(eta)
        if not 0 <= eta <= 1:
            raise ValueError('bridge eta must be in [0, 1]')
        self.eta = eta

    def reset_counts(self):
        self.support_calls = self.adapter_calls = 0

    def _support_hook(self, module, args):
        if len(args) != 1 or not torch.is_tensor(args[0]) or args[0].shape[1] != 3:
            raise RuntimeError('Expected existing RGB/TIR/joint three-channel support')
        self.support_calls += 1
        return (scale_gradient(args[0], self.eta),)

    def _make_adapter_hook(self, sig):
        def hook(module, args, kwargs):
            bound = sig.bind(*args, **kwargs)
            tir = bound.arguments['tir_feat']
            bound.arguments['tir_feat'] = tir  # Preserve the ORIGINAL feature feedback path
            guide = bound.arguments.get('external_guidance')
            if not isinstance(guide, dict) or 'joint_target_support' not in guide:
                raise RuntimeError('Original alignment-guided BHLR path is not active')
            guide = dict(guide)
            for key, value in guide.items():
                # These two are outputs of the learnable fusion. Its input
                # already has the bridge; preserve its parameter gradients.
                if key in ('joint_target_support', 'guidance_residual_logit'):
                    continue
                if torch.is_tensor(value):
                    guide[key] = scale_gradient(value, self.eta)
            bound.arguments['external_guidance'] = guide
            self.adapter_calls += 1
            return bound.args, bound.kwargs
        return hook

    def assert_used(self):
        if self.support_calls < 1 or self.adapter_calls < 1:
            raise RuntimeError('Original BHLR/guidance branch was bypassed')

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for obj, name, value in reversed(self.flags):
            setattr(obj, name, value)
        self.flags.clear()
        if getattr(self.core, '_module_only_bridge', None) is self:
            delattr(self.core, '_module_only_bridge')
