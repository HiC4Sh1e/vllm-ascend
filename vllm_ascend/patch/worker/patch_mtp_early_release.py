# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Plan-D: early release of async-lagged peak tensors in the DeepSeek-MTP draft path.

Why:
    GLM-5.2 MTP (resolved architecture ``DeepseekV32MTPModel`` -> upstream
    ``DeepSeekMTP``) hits a 57978 MB activation peak. At the peak instant 5
    x 93.75 MB full-token tensors co-reside with the 1168 MB MegaMoE workspace
    purely because of the NPU async-free lag (~1 ms): Python drops the ref when
    the tensor leaves logical scope, but the device-side free is enqueued on the
    same stream and fires ~1 ms later — landing after the MoE workspace alloc.
    See D:\\FFNChunk\\profiler\\mtp\\aten_empty_timeline.html §3/§5.

    In the *upstream* ``DeepSeekMultiTokenPredictorLayer.forward`` the tensors
    ``inputs_embeds`` (enorm output) and ``previous_hidden_states`` (hnorm
    output) are bound to local names and stay alive until the function returns
    — i.e. AFTER ``self.mtp_block(...)`` runs (which is where the 1168 MB MoE
    workspace is allocated). Explicitly ``del``-ing them right after
    ``eh_proj`` consumes their concatenation enqueues the device frees on the
    stream BEFORE the MoE workspace alloc, giving the ~1 ms lag time to land.

    The ``torch.cat`` result is anonymous and already auto-freed after
    ``eh_proj``; no ``del`` needed for it. The decoder-layer-internal tensors
    (input_layernorm output, attn output) are auto-freed by rebinding in
    upstream ``DeepseekV2DecoderLayer.forward`` — no patch needed there.

    DO NOT release the true cross-step read-dep 3a: that tensor is held by the
    Proposer (``self.hidden_states`` buffer, rejection sampling reads it) and
    only a *local* ref is dropped here, so it stays alive — exactly what we
    want.

How:
    Monkey-patch ``DeepSeekMultiTokenPredictorLayer.forward`` with a copy of the
    upstream body plus two explicit ``del`` + log points after ``eh_proj``.
    Gated by the ``mtp_memory_release`` additional-config switch (default ON,
    see AscendConfig). The patched forwards read the switch at runtime via
    ``get_ascend_config()``, so the monkey-patch is always installed and the
    switch gates the actual ``del`` + log behavior per call. Logged as
    ``[Plan-D] ... is released`` so release ordering can be verified in stdout.

Future Plan:
    Remove once the async-free lag is handled in CANN or a reorder/sync strategy
    is upstreamed. See [[glm52-mtp-memory-peak-analysis]].

Plan-A (mtp_moe_sync, default OFF):
    Plan-D only recovers the *Type-A* async-lagged tensors (2 enorm-out, 1a
    hnorm-out) whose refs survive past MoE. The *Type-B* tensors (3b MTP-
    internal all_gather, 1b o_proj output, 4c input_layernorm output) are
    auto-rebound at a read op *before* MoE (self_attn / post_attn_norm) — their
    frees are already enqueued at the earliest legal point, so no del can move
    them earlier; only a device sync can force the enqueued frees to execute
    before the MoE workspace alloc. ``mtp_moe_sync`` inserts that sync right
    before ``mtp_block``, recovering up to 3 x 93.75 = 281 MB more peak, at the
    cost of one sync per draft step. It is a separate switch so the zero-latency
    Plan-D can ship ON by default and Plan-A be opt-in when peak matters.
"""

import logging

import torch
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.model_executor.models import deepseek_mtp as _upstream_mtp

from vllm_ascend.ascend_config import get_ascend_config

_logger = logging.getLogger(__name__)

_OriginalPredictorLayerForward = _upstream_mtp.DeepSeekMultiTokenPredictorLayer.forward
_OriginalPredictorForward = _upstream_mtp.DeepSeekMultiTokenPredictor.forward
_OriginalPredictorLayerInit = _upstream_mtp.DeepSeekMultiTokenPredictorLayer.__init__

# Marker: a DeepseekV2DecoderLayer that is the ``mtp_block`` of an MTP draft
# layer. Only these layers should pay the Plan-A pre-MoE sync; the 77 regular
# decoder layers must not.
_MTP_BLOCK_MARKER = "_vllm_ascend_mtp_block_for_sync"


def _mtp_memory_release_enabled() -> bool:
    """Read the ``mtp_memory_release`` additional-config switch at runtime.

    AscendConfig may not be initialized yet at patch-import time, so the switch
    is read on each forward call (config is guaranteed ready by then). Falls
    back to True (the default field value) if config is unavailable, matching
    the AscendConfig field default.
    """
    try:
        return getattr(get_ascend_config(), "mtp_memory_release", True)
    except RuntimeError:
        return True


def _mtp_moe_sync_enabled() -> bool:
    """Read the ``mtp_moe_sync`` additional-config switch at runtime.

    Plan-A: a per-draft-step ``torch.npu.synchronize()`` before the MTP block's
    MoE workspace alloc, to force the Type-B async-lagged frees (3b/1b/4c —
    which Plan-D cannot move, since they are auto-rebound at a read op before
    MoE) to actually execute before the 1168 MB workspace alloc. Default OFF
    because a device sync costs latency on every draft step, unlike Plan-D's
    zero-cost reorder. Falls back to False (the default field value).
    """
    try:
        return getattr(get_ascend_config(), "mtp_moe_sync", False)
    except RuntimeError:
        return False


def _plan_d_log(name: str, tensor: torch.Tensor | None) -> None:
    """Log that a tensor is about to be released (Plan-D early release).

    Only logs; the caller is responsible for the actual ``del`` — a ``del``
    inside this helper would only drop the *local* parameter binding and leave
    the caller's reference (and the underlying storage) alive.
    """
    if tensor is None:
        return
    try:
        nbytes = tensor.storage().nbytes()
    except Exception:
        nbytes = -1
    _logger.info("[Plan-D] %s is released (shape=%s, bytes=%s)",
                 name, tuple(tensor.shape), nbytes)


def _plan_a_sync_and_mlp(self_mlp, *args, **kwargs):
    """Plan-A: synchronize before the MoE workspace alloc, then call mlp.

    Bound per-instance on the MTP draft ``mtp_block.mlp`` only. The
    ``post_attention_layernorm`` upstream call has already returned by the time
    ``mlp`` is entered, so the auto-rebind frees of the Type-B async-lagged
    tensors (3b / 1b / 4c) are enqueued on the stream. A synchronize here
    drains them to the device *before* the 1168 MB MoE workspace is allocated
    inside the original ``mlp.forward``. Gated by ``mtp_moe_sync`` at runtime.
    """
    if _mtp_moe_sync_enabled():
        torch.npu.synchronize()
        _logger.debug("[Plan-A] torch.npu.synchronize() before mtp_block MoE "
                      "(recovers Type-B async-lagged 3b/1b/4c)")
    orig = getattr(self_mlp, "_vllm_ascend_plan_a_orig_forward", None)
    return orig(self_mlp, *args, **kwargs)


def _plan_a_install_on_mtp_block(mtp_block):
    """Tag an MTP draft ``mtp_block`` (a DeepseekV2DecoderLayer) and wrap its
    ``mlp.forward`` with the Plan-A sync-first path. Idempotent."""
    if mtp_block is None or getattr(mtp_block, _MTP_BLOCK_MARKER, False):
        return
    setattr(mtp_block, _MTP_BLOCK_MARKER, True)
    mlp = getattr(mtp_block, "mlp", None)
    if mlp is None:
        return
    # Cache the original forward on the instance (one per mlp object) and bind
    # the sync-first wrapper. The wrapper reads the live switch every call.
    if not hasattr(mlp, "_vllm_ascend_plan_a_orig_forward"):
        mlp._vllm_ascend_plan_a_orig_forward = mlp.forward
    import types as _types  # local import keeps the module import list clean
    mlp.forward = _types.MethodType(_plan_a_sync_and_mlp, mlp)


def _plan_a_predictor_layer_init(self, *args, **kwargs):
    """Wrap DeepSeekMultiTokenPredictorLayer.__init__ to install Plan-A on its
    ``mtp_block`` once it exists. The sync-first mlp wrapper reads the
    ``mtp_moe_sync`` switch at every call, so installing it is harmless when
    the switch is off — it just calls through to the original mlp.forward.
    """
    _OriginalPredictorLayerInit(self, *args, **kwargs)
    mtp_block = getattr(self, "mtp_block", None)
    _plan_a_install_on_mtp_block(mtp_block)


def _plan_d_predictor_layer_forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    previous_hidden_states: torch.Tensor,
    inputs_embeds: torch.Tensor | None = None,
    spec_step_index: int = 0,
) -> torch.Tensor:
    """Plan-D patched forward for DeepSeekMultiTokenPredictorLayer.

    Identical to upstream except for the explicit ``del`` + log of
    ``inputs_embeds`` and ``previous_hidden_states`` right after ``eh_proj``
    consumes their concatenation, so the NPU device-side frees enqueue on the
    stream before the MoE workspace allocation inside ``mtp_block``.
    """
    assert inputs_embeds is not None
    # masking inputs at position 0, as not needed by MTP
    inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
    inputs_embeds = self.enorm(inputs_embeds)
    previous_hidden_states = self.hnorm(previous_hidden_states)

    hidden_states = self.eh_proj(
        torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
    )

    if _mtp_memory_release_enabled():
        # --- Plan-D (mtp_memory_release): eh_proj has consumed the cat of
        # (inputs_embeds, prev). Both are now dead for the rest of this forward,
        # but upstream keeps them bound until function return (after mtp_block's
        # MoE workspace alloc). Drop them now so their device-side frees enqueue
        # before the 1168 MB workspace alloc.
        # NOTE: previous_hidden_states here is the *hnorm output* (a fresh
        # tensor), not the cross-step 3a held by the Proposer — dropping the
        # local ref is safe; 3a stays alive in the caller.
        _plan_d_log("inputs_embeds(enorm-out)", inputs_embeds)
        del inputs_embeds
        _plan_d_log("previous_hidden_states(hnorm-out)", previous_hidden_states)
        del previous_hidden_states

    hidden_states, residual = self.mtp_block(
        positions=positions,
        hidden_states=hidden_states,
        residual=None,
    )
    hidden_states = residual + hidden_states  # pre-final-norm (logits hidden)
    if self.mtp_block.use_sequence_parallel_moe:
        hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
        hidden_states = hidden_states[: positions.shape[0]]
    # Recycle the post-final-norm hidden into the next draft step.
    # compute_logits applies shared_head (== final norm) to the pre-norm
    # element, so logits and the recycle each get exactly one final-norm.
    # Matches SGLang's deepseek_nextn.
    return hidden_states, self.shared_head(hidden_states)


def _plan_d_predictor_forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    previous_hidden_states: torch.Tensor,
    inputs_embeds: torch.Tensor | None = None,
    spec_step_idx: int = 0,
) -> torch.Tensor:
    """Plan-D patched forward for DeepSeekMultiTokenPredictor.

    Identical to upstream except it drops the Predictor-frame ``inputs_embeds``
    reference after handing it to the layer, so the device-side free of the
    embedding tensor (tensor #2, 93.75 MB) enqueues before the next draft
    step's MoE workspace instead of lingering until this frame returns.
    ``previous_hidden_states`` (3a) is intentionally NOT released: it is a
    true cross-step read-dep held by the Proposer for rejection sampling.
    """
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    current_step_idx = spec_step_idx % self.num_mtp_layers
    out = self.layers[str(self.mtp_start_layer_idx + current_step_idx)](
        input_ids,
        positions,
        previous_hidden_states,
        inputs_embeds,
        current_step_idx,
    )
    if _mtp_memory_release_enabled():
        # Plan-D (mtp_memory_release): inputs_embeds (tensor #2, 93.75 MB) is
        # fully consumed by the layer's enorm and not used afterwards in this
        # frame. Drop it so the free enqueues before the next draft step's MoE
        # workspace instead of lingering until this frame returns.
        _plan_d_log("inputs_embeds(embed)", inputs_embeds)
        del inputs_embeds
    return out


def apply_patch() -> None:
    """Install the Plan-D early-release forwards + Plan-A sync on the MTP
    predictor + layer.

    The patched forwards read the ``mtp_memory_release`` AscendConfig switch at
    runtime, so the monkey-patch is always installed unconditionally and the
    switch gates the actual ``del`` + log behavior per call. Plan-A wraps
    ``DeepSeekMultiTokenPredictorLayer.__init__`` so each draft ``mtp_block``'s
    ``mlp.forward`` gets a sync-first shim (gated at runtime by
    ``mtp_moe_sync``); installing the shim when the switch is off is a no-op
    pass-through to the original mlp.forward.
    """
    _upstream_mtp.DeepSeekMultiTokenPredictorLayer.forward = (
        _plan_d_predictor_layer_forward
    )
    _upstream_mtp.DeepSeekMultiTokenPredictor.forward = (
        _plan_d_predictor_forward
    )
    _upstream_mtp.DeepSeekMultiTokenPredictorLayer.__init__ = (
        _plan_a_predictor_layer_init
    )
    _logger.info(
        "[Plan-D] patched DeepSeekMultiTokenPredictor(Layer).forward "
        "(mtp_memory_release switch gates early release of "
        "enorm/hnorm/embed outputs before MoE workspace alloc; "
        "mtp_moe_sync switch gates a pre-MoE synchronize for Type-B "
        "async-lagged tensors)"
    )


apply_patch()
