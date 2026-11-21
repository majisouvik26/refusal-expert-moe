"""Sarvam-30B on vLLM 0.15: Bailing MoE subclass with optional FP32 router logits."""

import os

import torch
from vllm.model_executor.models.bailing_moe import BailingMoE, BailingMoeForCausalLM


def normalized_weights(weights):
    for name, value in weights:
        if name.endswith(".mlp.gate.e_score_correction_bias") or name.endswith(".gate.e_score_correction_bias"):
            if value.numel():
                value = value - value.mean()
        yield name, value


ROUTER_FP32 = os.environ.get("SARVAM_ROUTER_LOGITS_FP32", "0") == "1"


def fp32_router_forward(self, hidden_states):
    """BailingMoE.forward (vLLM 0.15.0) keeping router logits in FP32, as the HF reference does."""
    num_tokens, hidden_size = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_size)
    router_logits = self.gate(hidden_states.to(self.router_dtype))
    final_hidden_states = self.experts(hidden_states=hidden_states, router_logits=router_logits)
    if self.shared_experts is not None:
        shared_output, final_hidden_states = final_hidden_states
    else:
        shared_output = None
    final_hidden_states *= self.routed_scaling_factor
    if shared_output is not None:
        final_hidden_states = final_hidden_states + shared_output
    if self.tp_size > 1:
        final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(final_hidden_states)
    return final_hidden_states.view(num_tokens, hidden_size)


class SarvamMoEForCausalLM(BailingMoeForCausalLM):
    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._sarvam_router_logits_fp32 = ROUTER_FP32
        if ROUTER_FP32:
            import types

            for module in self.modules():
                if isinstance(module, BailingMoE):
                    if module.router_dtype != torch.float32:
                        raise RuntimeError("FP32 router logits need router_dtype fp32 in config.json")
                    module.forward = types.MethodType(fp32_router_forward, module)

    def load_weights(self, weights):
        return super().load_weights(normalized_weights(weights))
