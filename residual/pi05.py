"""Frozen lerobot π0.5 as the residual's base flow: loading, prefix encoding, velocity + action-expert hidden states.

`encode` copies the prefix block of PI05Pytorch.sample_actions and `velocity_and_hidden` copies
PI05Pytorch.denoise_step (third_party/lerobot/src/lerobot/policies/pi05/modeling_pi05.py), returning the
pre-projection hidden states as well. tests/test_residual_pi05.py checks the composed sampler against the
stock sample_actions, so drift in either copy shows up there.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, make_att_2d_masks
from lerobot.processor.normalize_processor import UnnormalizerProcessorStep
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)
from torch import Tensor
from transformers import DynamicCache

from residual.qam_e import Context, Dims

IMAGE_KEYS = (f"{OBS_IMAGES}.image", f"{OBS_IMAGES}.image2")  # VLS LiberoEnv camera mapping (agentview, wrist)


@dataclass
class Pi05Base:
    kv: Any            # prefix KV cache, fixed for every flow step of a chunk
    pad_masks: Tensor  # [B, L] prefix padding mask

    def repeat(self, n: int) -> "Pi05Base":
        """The same prefix n times along the batch (a copy; the cache is not modified)."""
        kv = DynamicCache.from_legacy_cache(tuple((k.repeat(n, 1, 1, 1), v.repeat(n, 1, 1, 1))
                                                  for k, v in self.kv.to_legacy_cache()))
        return Pi05Base(kv=kv, pad_masks=self.pad_masks.repeat(n, 1))


class Pi05:
    def __init__(self, pretrained_path: str, device: str):
        config = PreTrainedConfig.from_pretrained(pretrained_path)
        config.compile_model = False  # compile only wraps sample_actions, which the residual sampler replaces
        config.device = device
        self.policy = PI05Policy.from_pretrained(pretrained_path, config=config)
        self.policy.to(device).eval().requires_grad_(False)
        self.model = self.policy.model
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config, pretrained_path=pretrained_path,
            preprocessor_overrides={"device_processor": {"device": device}})
        unnormalizer = next(s for s in self.postprocessor.steps if isinstance(s, UnnormalizerProcessorStep))
        stats = unnormalizer._tensor_stats[ACTION]
        self.action_mean, self.action_std = stats["mean"].float().cpu(), stats["std"].float().cpu()
        # The env clips raw actions to [-1, 1]; these are those bounds in the policy's normalized space.
        self.action_low = (-1 - self.action_mean) / self.action_std
        self.action_high = (1 - self.action_mean) / self.action_std
        self.device = device
        self.u8_scale = torch.tensor(255.0, device=device)

    def dims(self, exec_horizon: int, flow_steps: int) -> Dims:
        cfg = self.policy.config
        if flow_steps != cfg.num_inference_steps:
            raise ValueError(f"policy.num_inference_steps={flow_steps} but the checkpoint uses {cfg.num_inference_steps}")
        return Dims(horizon=cfg.chunk_size, pad_dim=cfg.max_action_dim, action_dim=cfg.output_features[ACTION].shape[0],
                    exec_horizon=exec_horizon, ae_dim=self.model.action_out_proj.in_features,
                    prefix_dim=self.model.paligemma_with_expert.paligemma.config.text_config.hidden_size,
                    state_dim=cfg.input_features[OBS_STATE].shape[0], flow_steps=flow_steps)

    # ---------------------------------------------------------------- observations
    def preprocess(self, image: np.ndarray, image2: np.ndarray, state: np.ndarray, tasks: list[str]) -> dict:
        """uint8 [B, 3, H, W] env-preprocessed cameras + raw state [B, S] -> policy batch (normalized, tokenized)."""
        def to_float(u8):  # copy uint8 to the GPU (4x fewer bytes than float32), convert there. A CUDA tensor
            # divisor keeps true division bit-identical to the CPU /255 (a Python scalar multiplies by 1/255).
            return torch.from_numpy(np.ascontiguousarray(u8)).to(self.device).float() / self.u8_scale
        obs = {IMAGE_KEYS[0]: to_float(image), IMAGE_KEYS[1]: to_float(image2),
               OBS_STATE: torch.from_numpy(np.ascontiguousarray(state)).float(), "task": list(tasks)}
        return self.preprocessor(obs)

    @torch.no_grad()
    def context(self, batch: dict, progress: Tensor) -> Context:
        base, h_prefix = self.encode(batch)
        return Context(h_prefix=h_prefix, state=batch[OBS_STATE].float(), progress=progress.to(self.device), base=base)

    def prefix_inputs(self, batch: dict) -> tuple[list[Tensor], list[Tensor], Tensor, Tensor]:
        """embed_prefix inputs without tokens that are masked out of attention anyway: the checkpoint's empty camera
        (mask 0) and the right-padded prompt tail. ~970 -> ~570 prefix tokens; the output only changes by bf16
        rounding (same size as changing the batch size, measured 2026-11-09)."""
        images, img_masks = self.policy._preprocess_images(batch)
        keep = [i for i, m in enumerate(img_masks) if m.any()]
        lang_len = int(batch[OBS_LANGUAGE_ATTENTION_MASK].sum(1).max())
        return ([images[i] for i in keep], [img_masks[i] for i in keep],
                batch[OBS_LANGUAGE_TOKENS][:, :lang_len], batch[OBS_LANGUAGE_ATTENTION_MASK][:, :lang_len])

    @torch.no_grad()
    def encode(self, batch: dict) -> tuple[Pi05Base, Tensor]:
        """Prefix forward (images + tokenized prompt with state) -> KV cache and masked-mean prefix features."""
        model = self.model
        prefix_embs, pad_masks, att_masks = model.embed_prefix(*self.prefix_inputs(batch))
        att_2d = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        (prefix_out, _), kv = model.paligemma_with_expert.forward(
            attention_mask=model._prepare_attention_masks_4d(att_2d), position_ids=position_ids,
            past_key_values=None, inputs_embeds=[prefix_embs, None], use_cache=True)
        weights = pad_masks[..., None].float()
        h_prefix = (prefix_out.float() * weights).sum(1) / weights.sum(1)
        return Pi05Base(kv=kv, pad_masks=pad_masks), h_prefix

    # ---------------------------------------------------------------- flow
    def velocity_and_hidden(self, base: Pi05Base, x_t: Tensor, timestep: Tensor) -> tuple[Tensor, Tensor]:
        """denoise_step returning (velocity [B, H, D], action-expert hidden [B, H, D_ae]); differentiable in x_t."""
        model = self.model
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = model.embed_suffix(x_t, timestep)
        batch_size, prefix_len = base.pad_masks.shape
        suffix_len = suffix_pad_masks.shape[1]
        prefix_pad_2d = base.pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        full_att_2d = torch.cat([prefix_pad_2d, make_att_2d_masks(suffix_pad_masks, suffix_att_masks)], dim=2)
        position_ids = torch.sum(base.pad_masks, dim=-1)[:, None] + torch.cumsum(suffix_pad_masks, dim=1) - 1
        model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
        (_, suffix_out), _ = model.paligemma_with_expert.forward(
            attention_mask=model._prepare_attention_masks_4d(full_att_2d), position_ids=position_ids,
            past_key_values=base.kv, inputs_embeds=[None, suffix_embs], use_cache=False,
            adarms_cond=[None, adarms_cond])
        hidden = suffix_out[:, -model.config.chunk_size:].to(dtype=torch.float32)
        return model.action_out_proj(hidden), hidden

    def unnormalize(self, actions: Tensor) -> Tensor:
        """Normalized [B, T, A] -> raw env actions on CPU (the policy's own postprocessor, as VLS uses it)."""
        return self.postprocessor(actions)
