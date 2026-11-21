"""vLLM general plugin: registers the Sarvam MoE model class in every worker."""


def register():
    from vllm import ModelRegistry

    ModelRegistry.register_model("SarvamMoEForCausalLM", "refusal_moe.sarvam.vllm_plugin.model:SarvamMoEForCausalLM")
