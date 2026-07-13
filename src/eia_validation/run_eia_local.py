"""EIA-harness behavioral validation of weight edits (gate 3, log entry E16).

Drives the ORIGINAL Empathy-in-Action games (vendored at third_party/eia)
with a local gemma-2-9b-it player under weight-edit conditions, by
monkeypatching the harness's single LLM entrypoint.

Routing inside the patched call:
  - player decisions + initial self-assessment  -> local HF model (edited or not)
  - rubric judge calls (hard-coded provider="openai" in main.py) -> DEFERRED:
    recorded to deferred_judgements.jsonl for offline scoring (Batches API),
    returning {"score": -1}. Rule-based scores can be computed from
    experiment.json histories afterwards.

Integrity Repair A QA (Q1/Q10, 2026-07-13):
  * component sets resolve EXPLICITLY through the versioned registry
    (``--component-set``) or complete literal specs, always bound to the
    supplied direction (path + SHA-256); omission cannot silently choose a
    set, and the complete resolution is persisted in the run manifest. The
    removed generic POSITIVE_WRITERS/SUPPRESSORS/TARGETED constants are NOT
    reintroduced.
  * the shared accepted/exploratory run contract applies: accepted mode
    requires immutable verified model/tokenizer revisions, a fresh output
    directory, clean source binding, and the registry entry named even for
    the baseline arm (a paired design's baseline must bind to the same
    entry/direction as the edited arm).
  * every run directory gets a run-level manifest: requested + resolved
    revisions, direction path/hash, component resolution, edit diagnostics,
    seed/run identity, generation parameters, scenario list and message-pool
    hashes, vendored EIA code hash + repo commit, and environment provenance.
  * each (scenario, seed) game gets a FRESH player whose sampling-seed
    derivation (run seed x call counter) is persisted, so paired seeds never
    depend on preceding trajectories.
  * failure states are preserved explicitly (per-run ok/error rows plus a
    nonzero exit) so partial results cannot be mistaken for complete grids.

Usage (Spark, `empathy` env):
  python -u src/eia_validation/run_eia_local.py \
    --direction results/controlled_directions_gemma2_9b_it/direction_M_resid_block20.npy \
    --component-set gemma2_9b_it_resid_2026-07-12 \
    --condition baseline            # or positive_writers | suppressors | targeted
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "third_party" / "eia"))
sys.path.insert(0, str(REPO / "src"))

from weight_orthogonalization import (  # noqa: E402
    orthogonalize_component, parse_component, snapshot_weights,
)
from component_sets import ComponentSetError, resolve_component_sets  # noqa: E402
from utils.evidence_run import (  # noqa: E402
    EvidenceRunError, atomic_write_json, evaluate_run_contract,
    require_keys_validator, revalidate_run_contract,
)
from utils.run_provenance import (  # noqa: E402
    collect_run_provenance, resolve_hf_commit, resolve_model_and_tokenizer,
    sha256_file,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eia-local")

SCENARIOS = ["food_delivery", "the_maze", "the_listener", "the_protector", "the_duel"]

#: Condition -> registry role. Baseline applies no edit but must still bind
#: to the registry entry/direction in a paired accepted design (QA Q10).
CONDITION_ROLES = {
    "baseline": None,
    "positive_writers": "positive_writers",
    "suppressors": "suppressors",
    "targeted": "targeted",
}

RUN_MANIFEST_SCHEMA = "empathy-action-probes/eia-run-manifest/1"

#: Every run-level manifest must carry these keys with non-None values
#: (``edits`` may be an empty list for the baseline arm).
REQUIRED_MANIFEST_KEYS = (
    "schema", "driver", "run_mode", "condition", "model", "revisions",
    "hf_resolved", "direction", "component_resolution", "edits", "seeds",
    "scenarios", "max_steps", "generation", "player_seed_derivation",
    "message_pools", "eia_code", "run_contract", "provenance",
)

PLAYER_SEED_DERIVATION = {
    "formula": "torch.manual_seed(run_seed * 100003 + call_index)",
    "call_index": "1-based, incremented before sampling; RESET per game run "
                  "(fresh player), so paired seeds do not depend on "
                  "preceding trajectories",
}


def derive_player_call_seed(run_seed, call_index):
    """Persisted sampling-seed derivation (QA Q10)."""
    return (run_seed or 0) * 100003 + call_index


def content_tree_hash(root, suffixes=(".py", ".png")):
    """Deterministic content hash of a code tree (sorted relpath + bytes)."""
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*")
                       if p.is_file() and p.suffix in suffixes):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def validate_unique_run_dimensions(scenarios, seeds):
    """Reject duplicate/ill-typed dimensions before output names are built."""
    if not scenarios or not all(isinstance(v, str) and v for v in scenarios):
        raise EvidenceRunError("scenarios must be non-empty strings")
    if len(scenarios) != len(set(scenarios)):
        raise EvidenceRunError("scenarios contain duplicates")
    if not seeds or not all(isinstance(v, int) and not isinstance(v, bool)
                            for v in seeds):
        raise EvidenceRunError("seeds must be integers")
    if len(seeds) != len(set(seeds)):
        raise EvidenceRunError("seeds contain duplicates")
    return {"scenarios": list(scenarios), "seeds": list(seeds)}


def validate_run_manifest(manifest):
    """The manifest must be complete before any game runs (QA Q10)."""
    missing = [key for key in REQUIRED_MANIFEST_KEYS
               if key not in manifest or manifest[key] is None]
    if missing:
        raise EvidenceRunError(
            f"run manifest incomplete; missing/None keys: {missing}")
    return manifest


class LocalPlayer:
    """One player per game run: the call counter feeds the sampling seed, so
    reuse across runs would break cross-condition seed pairing (QA Q10)."""

    def __init__(self, model, tokenizer, device, max_new_tokens=220,
                 temperature=0.7, top_p=0.95):
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.calls = 0
        self.deferred = []

    @torch.no_grad()
    def __call__(self, query, system=None, provider=None, model=None, seed=None):
        from empathy.core.prompt import _extract_json_from_text

        if provider == "openai":  # the harness's hard-coded judge call
            self.deferred.append({"query": query, "system": system, "seed": seed})
            return {"score": -1, "reason": "deferred to offline judging"}

        self.calls += 1
        text = (system + "\n\n" if system else "") + query
        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True,
        ).to(self.device)
        torch.manual_seed(derive_player_call_seed(seed, self.calls))
        out = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=True,
            temperature=self.temperature, top_p=self.top_p,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        n_prompt = inputs["input_ids"].shape[1]
        completion = self.tokenizer.decode(out[0, n_prompt:], skip_special_tokens=True)
        parsed = _extract_json_from_text(completion)
        return parsed if parsed is not None else {"raw": completion}


def resolve_condition_components(condition, *, set_key, explicit, model,
                                 direction, run_mode,
                                 allow_direction_mismatch=False):
    """Resolve the condition's components with no silent defaults (QA Q1).

    Edit conditions require the registry key or a complete literal spec —
    omission raises. The baseline arm resolves no components but still binds
    the registry entry + direction; in accepted mode the registry key is
    REQUIRED for every arm so the paired baseline carries the same binding
    as the edited arm.
    """
    role = CONDITION_ROLES[condition]
    if run_mode == "accepted" and set_key is None:
        raise ComponentSetError(
            "accepted paired game designs must name --component-set so the "
            f"{condition!r} arm (baseline included) binds to the same "
            "registry entry and direction as every other arm")
    roles = (role,) if role else ()
    return resolve_component_sets(
        roles=roles,
        explicit={role: explicit.get(role)} if role else {},
        set_key=set_key,
        model=model,
        direction_path=direction,
        run_mode=run_mode,
        allow_direction_mismatch=allow_direction_mismatch,
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-2-9b-it")
    parser.add_argument("--revision", default=None,
                        help="explicit HF model revision (accepted runs must pin this)")
    parser.add_argument("--tokenizer-revision", default=None,
                        help="explicit tokenizer revision (defaults to --revision)")
    parser.add_argument("--direction", required=True)
    parser.add_argument("--condition", choices=list(CONDITION_ROLES),
                        default="baseline")
    parser.add_argument("--component-set", default=None,
                        help="versioned set-of-record key from src/component_sets.py")
    parser.add_argument("--positive-writers", default=None,
                        help="explicit literal spec for the positive_writers role")
    parser.add_argument("--suppressors", default=None,
                        help="explicit literal spec for the suppressors role")
    parser.add_argument("--targeted", default=None,
                        help="explicit literal spec for the targeted role")
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1234, 999])
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--run-mode", choices=("accepted", "exploratory"),
                        default="exploratory",
                        help="accepted = evidence-eligible (pinned immutable "
                             "revisions, verified direction binding, FRESH "
                             "output directory, clean source, registry-bound "
                             "baseline); exploratory artifacts are persisted "
                             "as ineligible for confirmatory evidence")
    parser.add_argument("--allow-direction-mismatch", action="store_true",
                        help="EXPLORATORY-ONLY override; persisted as "
                             "non-confirmatory")
    parser.add_argument("--allowed-dirty", action="append", default=[],
                        help="explicit source-binding exclusion rule (glob) "
                             "for accepted mode; persisted in the manifest")
    parser.add_argument("--out", default="results/eia_local")
    args = parser.parse_args(argv)

    try:
        validate_unique_run_dimensions(args.scenarios, args.seeds)
    except EvidenceRunError as exc:
        parser.error(str(exc))

    try:
        resolution = resolve_condition_components(
            args.condition,
            set_key=args.component_set,
            explicit={"positive_writers": args.positive_writers,
                      "suppressors": args.suppressors,
                      "targeted": args.targeted},
            model=args.model,
            direction=args.direction,
            run_mode=args.run_mode,
            allow_direction_mismatch=args.allow_direction_mismatch,
        )
    except ComponentSetError as exc:
        parser.error(str(exc))
    log.info("component resolution: %s", resolution)

    outdir = Path(args.out) / args.condition
    try:
        if args.run_mode == "accepted" and outdir.exists():
            raise EvidenceRunError(
                f"accepted mode refuses the existing output directory "
                f"{outdir}; prior accepted game runs are never overwritten "
                "and partial trees must not be extended in place")
        contract = evaluate_run_contract(
            args.run_mode,
            revisions={
                "model": {
                    "requested": args.revision,
                    "resolution": resolve_hf_commit(
                        args.model, revision=args.revision or "main"),
                },
                "tokenizer": {
                    "requested": args.tokenizer_revision or args.revision,
                    "resolution": resolve_hf_commit(
                        args.model,
                        revision=(args.tokenizer_revision or args.revision
                                  or "main")),
                },
            },
            output_paths=(outdir / "run_manifest.json",
                          outdir / "run_summary.json"),
            source_rules=args.allowed_dirty,
            input_paths=(args.direction,),
        )
    except EvidenceRunError as exc:
        parser.error(str(exc))
    if contract.get("warning"):
        log.warning("%s", contract["warning"])

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import empathy.main as eia_main
    import empathy.games.the_listener as listener_mod

    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir.mkdir(parents=True, exist_ok=True)

    role = CONDITION_ROLES[args.condition]
    condition_spec = resolution["sets"].get(role, "") if role else ""

    manifest = {
        "schema": RUN_MANIFEST_SCHEMA,
        "driver": "run_eia_local",
        "run_mode": args.run_mode,
        "condition": args.condition,
        "model": args.model,
        "revisions": {"model_requested": args.revision,
                      "tokenizer_requested": (args.tokenizer_revision
                                              or args.revision)},
        "hf_resolved": resolve_model_and_tokenizer(
            args.model, revision=args.revision or "main",
            tokenizer_revision=args.tokenizer_revision or args.revision or "main",
        ),
        "direction": {"path": args.direction,
                      "sha256": sha256_file(args.direction)},
        "component_resolution": resolution,
        "edits": [],
        "seeds": list(args.seeds),
        "scenarios": list(args.scenarios),
        "max_steps": args.max_steps,
        "generation": {"max_new_tokens": args.max_new_tokens,
                       "temperature": args.temperature,
                       "top_p": args.top_p, "do_sample": True},
        "player_seed_derivation": PLAYER_SEED_DERIVATION,
        "message_pools": {
            "the_listener_distress_master": hashlib.sha256(json.dumps(
                list(listener_mod.DEPRESSIVE_MESSAGES_MASTER)
            ).encode("utf-8")).hexdigest(),
        },
        "eia_code": {
            "vendored_tree": str(REPO / "third_party" / "eia"),
            "tree_sha256_behavioral": content_tree_hash(
                REPO / "third_party" / "eia"),
            "included_suffixes": [".py", ".png"],
        },
        "run_contract": contract,
        "provenance": collect_run_provenance(
            files={"direction": args.direction}),
    }

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.tokenizer_revision or args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager",
        revision=args.revision,
    ).to(device)
    model.eval()

    if condition_spec:
        direction = torch.tensor(np.load(args.direction), dtype=torch.float32,
                                 device=device)
        components = [parse_component(v) for v in condition_spec.split(",")]
        snapshot_weights(model, components)  # edits persist for the whole run
        for component in components:
            edit = orthogonalize_component(model, component, direction)
            manifest["edits"].append(edit)
            log.info("edited %s (rel change %.4f)", component["name"],
                     edit["relative_component_weight_change"])

    # complete manifest is validated and written BEFORE any game runs, so a
    # crashed/partial tree is still fully attributable
    validate_run_manifest(manifest)
    atomic_write_json(outdir / "run_manifest.json", manifest,
                      require_fresh=True,
                      validate_fn=require_keys_validator(*REQUIRED_MANIFEST_KEYS))

    eia_main._ensure_api_keys = lambda *_a, **_k: None

    summary, deferred, failures = [], [], []
    n_player_calls = 0
    for scenario in args.scenarios:
        for seed in args.seeds:
            log.info("=== %s | %s | seed %d ===", args.condition, scenario, seed)
            # fresh player per run (QA Q10): call counter and deferred log
            # reset so sampling seeds are independent of prior trajectories
            player = LocalPlayer(model, tokenizer, device,
                                 max_new_tokens=args.max_new_tokens,
                                 temperature=args.temperature,
                                 top_p=args.top_p)
            eia_main.call_llm_with_prompt = player
            try:
                result = eia_main.run_single_game(
                    game=scenario, max_steps=args.max_steps, use_llm=True,
                    outdir_root=str(outdir), run_name=f"{scenario}_seed{seed}",
                    print_prompt=False, save_html=False, save_gif=False,
                    save_video=False, provider="local", model=args.condition,
                    seed=seed,
                )
                summary.append({"scenario": scenario, "seed": seed, "ok": True,
                                "n_player_calls": player.calls,
                                "result": str(result)[:500]})
            except Exception as exc:
                log.exception("run failed: %s %s", scenario, seed)
                summary.append({"scenario": scenario, "seed": seed, "ok": False,
                                "n_player_calls": player.calls,
                                "error": str(exc)[:500]})
                failures.append(f"{scenario}/seed{seed}")
            n_player_calls += player.calls
            deferred.extend(player.deferred)

    with open(outdir / "deferred_judgements.jsonl", "w") as f:
        for row in deferred:
            f.write(json.dumps(row) + "\n")
    revalidate_run_contract(
        contract, source_rules=args.allowed_dirty,
        input_paths=(args.direction,))
    atomic_write_json(
        outdir / "run_summary.json",
        {"condition": args.condition, "run_mode": args.run_mode,
         "complete": not failures, "failed_runs": failures, "runs": summary,
         "n_player_calls": n_player_calls},
        require_fresh=True,
        validate_fn=require_keys_validator(
            "condition", "run_mode", "complete", "failed_runs", "runs"),
    )
    log.info("done: %d runs (%d failed), %d player calls, %d deferred judgements",
             len(summary), len(failures), n_player_calls, len(deferred))
    if failures:
        log.error("EIA_RUNS_FAILED: %s — failure states preserved in "
                  "run_summary.json; this tree is NOT a complete grid",
                  failures)
        sys.exit(1)


if __name__ == "__main__":
    main()
