def top_k_renorm_probs(probs, top_k):
    from sglang.kernels.ops.sampling.renorm_triton import top_k_renorm_probs_triton

    return top_k_renorm_probs_triton(probs, top_k)


def top_p_renorm_probs(probs, top_p):
    from sglang.kernels.ops.sampling.renorm_triton import top_p_renorm_probs_triton

    return top_p_renorm_probs_triton(probs, top_p)


def install():
    import importlib
    import sys

    import flashinfer.sampling as sampling

    from hcu_adapter import registry

    for name, implementation in (
        ("top_k_renorm_probs", top_k_renorm_probs),
        ("top_p_renorm_probs", top_p_renorm_probs),
    ):
        setattr(sampling, name, implementation)
        registry.register(
            "sampling", implementation, f"flashinfer.sampling.{name}",
            registry.MECH_REBIND,
            detail="SGLang native HIP Triton renormalization through CUDA import face",
        )
    from hcu_adapter.kernels.sglang_tree_sampling_triton import (
        tree_speculative_sampling_target_only,
    )

    module = importlib.import_module("sglang.kernels.ops.speculative.sampling")
    original = module.tree_speculative_sampling_target_only
    for name, loaded in list(sys.modules.items()):
        if name.startswith("sglang.") and loaded is not None:
            if vars(loaded).get("tree_speculative_sampling_target_only") is original:
                loaded.tree_speculative_sampling_target_only = tree_speculative_sampling_target_only
    registry.register(
        "sglang_tree_sampling", tree_speculative_sampling_target_only,
        "sglang.kernels.ops.speculative.sampling.tree_speculative_sampling_target_only",
        registry.MECH_REBIND,
        detail="Target-only tree traversal and residual sampling without FlashInfer headers",
    )
