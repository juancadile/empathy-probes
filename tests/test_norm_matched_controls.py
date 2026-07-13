"""apply_norm_matched_random on a toy bf16 model: (1) a shared rand_vec yields
ONE underlying residual direction across every component of a set (the same
contract as the targeted edit, which removes one shared direction everywhere);
(2) the realized post-bf16 delta norm is measured and within 3% of requested.

CPU-only, no model download.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from norm_matched_controls import (  # noqa: E402
    apply_norm_matched_random, orthogonalize_component_measured,
    targeted_delta_norm,
)
from weight_orthogonalization import effective_direction, parse_component  # noqa: E402

D_MODEL, D_FF, N_HEADS, HEAD_DIM = 64, 128, 4, 16


def toy_model(seed=0):
    g = torch.Generator().manual_seed(seed)
    layers = []
    for _ in range(2):
        layers.append(SimpleNamespace(
            mlp=SimpleNamespace(down_proj=SimpleNamespace(
                weight=torch.randn(D_MODEL, D_FF, generator=g).to(torch.bfloat16))),
            self_attn=SimpleNamespace(o_proj=SimpleNamespace(
                weight=torch.randn(D_MODEL, N_HEADS * HEAD_DIM,
                                   generator=g).to(torch.bfloat16))),
            # Gemma-style sandwich norms with distinct per-component gains
            post_feedforward_layernorm=SimpleNamespace(
                weight=torch.rand(D_MODEL, generator=g) * 0.2),
            post_attention_layernorm=SimpleNamespace(
                weight=torch.rand(D_MODEL, generator=g) * 0.2),
        ))
    config = SimpleNamespace(hidden_size=D_MODEL, num_attention_heads=N_HEADS,
                             head_dim=HEAD_DIM)
    return SimpleNamespace(model=SimpleNamespace(layers=layers), config=config)


COMPONENTS = ["L0MLP", "L1MLP", "L0H1", "L1H3"]


def apply_set(model, rand_vec, generator):
    g = torch.Generator().manual_seed(7)
    direction = torch.randn(D_MODEL, generator=g)
    direction /= torch.linalg.vector_norm(direction)
    deltas, edits = {}, {}
    for name in COMPONENTS:
        comp = parse_component(name)
        target = targeted_delta_norm(model, comp, direction)
        weight = (model.model.layers[comp["layer"]].mlp.down_proj.weight
                  if comp["head"] is None
                  else model.model.layers[comp["layer"]].self_attn.o_proj.weight)
        before = weight.detach().clone()
        edits[name] = apply_norm_matched_random(
            model, comp, direction, target, generator, rand_vec=rand_vec)
        deltas[name] = (before - weight).float()
        weight.copy_(before)  # restore for the next component
    return deltas, edits


def left_residual_direction(model, name, delta):
    """Recover the residual-stream direction of a rank-1 delta by undoing the
    per-component sandwich-norm gain from its dominant left singular vector."""
    comp = parse_component(name)
    layer = model.model.layers[comp["layer"]]
    if comp["head"] is not None:
        delta = delta[:, comp["head"] * HEAD_DIM:(comp["head"] + 1) * HEAD_DIM]
    gain = 1.0 + (layer.post_feedforward_layernorm.weight
                  if comp["head"] is None
                  else layer.post_attention_layernorm.weight).float()
    u = torch.linalg.svd(delta, full_matrices=False)[0][:, 0]
    v = u / gain
    return v / torch.linalg.vector_norm(v)


def test_shared_rand_vec_is_one_direction_across_components():
    model = toy_model()
    g = torch.Generator().manual_seed(123)
    rand_vec = torch.randn(D_MODEL, generator=g)
    deltas, _ = apply_set(model, rand_vec, g)
    ref = rand_vec / torch.linalg.vector_norm(rand_vec)
    for name, delta in deltas.items():
        cos = abs(float(left_residual_direction(model, name, delta) @ ref))
        assert cos > 0.99, f"{name}: edit direction cos {cos:.4f} vs shared rand_vec"


def test_no_rand_vec_draws_fresh_directions():
    # regression guard for the R1 flaw: omitting rand_vec gives each component
    # an INDEPENDENT direction, which must not be procedure-matched behavior
    model = toy_model()
    g = torch.Generator().manual_seed(123)
    deltas, _ = apply_set(model, None, g)
    dirs = [left_residual_direction(model, n, d) for n, d in deltas.items()]
    cos = abs(float(dirs[0] @ dirs[1]))
    assert cos < 0.5, f"independent draws unexpectedly aligned (cos {cos:.4f})"


def test_targeted_measured_edit_returns_realized_and_theoretical_norms():
    model = toy_model()
    g = torch.Generator().manual_seed(7)
    direction = torch.randn(D_MODEL, generator=g)
    direction /= torch.linalg.vector_norm(direction)
    for name in COMPONENTS:
        comp = parse_component(name)
        weight = (model.model.layers[comp["layer"]].mlp.down_proj.weight
                  if comp["head"] is None
                  else model.model.layers[comp["layer"]].self_attn.o_proj.weight)
        before = weight.detach().clone()
        theoretical = targeted_delta_norm(model, comp, direction)
        info = orthogonalize_component_measured(model, comp, direction)
        actual = float(torch.linalg.matrix_norm((before - weight).float()))
        weight.copy_(before)
        # existing orthogonalize_component metadata must be preserved
        assert set(info) >= {"component", "relative_component_weight_change",
                             "removed_alignment_norm", "theoretical_delta_norm",
                             "realized_delta_norm",
                             "realized_vs_theoretical_rel_error"}
        assert info["component"] == name
        assert info["theoretical_delta_norm"] == theoretical
        # realized norm must equal the actual post-bf16 change on the weights
        assert abs(actual - info["realized_delta_norm"]) < 1e-4 * max(actual, 1.0)
        assert abs(info["realized_delta_norm"] / theoretical - 1.0
                   - info["realized_vs_theoretical_rel_error"]) < 1e-9
        # bf16 cast noise is small: realized within 3% of theoretical here
        assert abs(info["realized_vs_theoretical_rel_error"]) <= 0.03


def test_null_gate_is_against_targeted_realized_norm():
    # the exact-matching contract: requesting the targeted edit's REALIZED
    # norm makes the helper's 3% assert a realized-vs-realized gate
    model = toy_model()
    g = torch.Generator().manual_seed(123)
    rand_vec = torch.randn(D_MODEL, generator=g)
    direction = torch.randn(D_MODEL, generator=g)
    direction /= torch.linalg.vector_norm(direction)
    for name in COMPONENTS:
        comp = parse_component(name)
        weight = (model.model.layers[comp["layer"]].mlp.down_proj.weight
                  if comp["head"] is None
                  else model.model.layers[comp["layer"]].self_attn.o_proj.weight)
        before = weight.detach().clone()
        tgt = orthogonalize_component_measured(model, comp, direction)
        weight.copy_(before)
        edit = apply_norm_matched_random(model, comp, direction,
                                         tgt["realized_delta_norm"], g,
                                         rand_vec=rand_vec)
        weight.copy_(before)
        assert edit["requested_delta_norm"] == tgt["realized_delta_norm"]
        assert (abs(edit["realized_delta_norm"] / tgt["realized_delta_norm"] - 1.0)
                <= 0.03)


def test_realized_norm_within_3pct_and_measured_from_weights():
    model = toy_model()
    g = torch.Generator().manual_seed(123)
    rand_vec = torch.randn(D_MODEL, generator=g)
    deltas, edits = apply_set(model, rand_vec, g)
    for name, e in edits.items():
        assert set(e) >= {"requested_delta_norm", "realized_delta_norm",
                          "relative_norm_error"}
        assert abs(e["relative_norm_error"]) <= 0.03
        # realized norm must equal the actual post-bf16 change on the weights
        actual = float(torch.linalg.matrix_norm(deltas[name]))
        assert abs(actual - e["realized_delta_norm"]) < 1e-4 * max(actual, 1.0)
        assert abs(e["realized_delta_norm"] / e["requested_delta_norm"] - 1.0
                   - e["relative_norm_error"]) < 1e-9
