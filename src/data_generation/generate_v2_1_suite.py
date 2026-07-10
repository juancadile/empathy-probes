"""Generate the V2.1 stimulus suite (cells A-H, S, R) with SMALL API models.

Design: notes/v2_1-stimulus-suite-design.md
Cells:  data/eia_scenarios/v2_1_cells.json

Cheapest current models per provider (~$5-10 total for the full suite):
  Anthropic: claude-haiku-4-5-20251001
  OpenAI:    gpt-4o-mini
  Google:    gemini-2.5-flash

Usage:
  python src/data_generation/generate_v2_1_suite.py --dry-run
  python src/data_generation/generate_v2_1_suite.py                 # cells A B D E F G H S
  python src/data_generation/generate_v2_1_suite.py --cells A B
  python src/data_generation/generate_v2_1_suite.py --cells R       # after A exists

Output: data/contrastive_pairs/v2_1/{cell}_{model_key}.jsonl (resumable).
"""

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CELLS_PATH = PROJECT_ROOT / "data" / "eia_scenarios" / "v2_1_cells.json"
EIA_SCENARIOS_PATH = PROJECT_ROOT / "data" / "eia_scenarios" / "scenarios.json"
OUT_DIR = PROJECT_ROOT / "data" / "contrastive_pairs" / "v2_1"

MODELS = {
    "claude-haiku": "claude-haiku-4-5-20251001",
    "gpt-4o-mini": "gpt-4o-mini",
    "gemini-flash": "gemini-2.5-flash",
}
TEMPERATURE = 0.8
MAX_TOKENS = 1024

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("v2_1")


def load_env():
    env = PROJECT_ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# ---------------------------------------------------------------- providers

def gen_claude(prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=MODELS["claude-haiku"],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
    )
    return next(b.text for b in resp.content if b.type == "text")


def gen_openai(prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model=MODELS["gpt-4o-mini"],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def gen_gemini(prompt: str) -> str:
    import google.generativeai as genai

    genai.configure(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    model = genai.GenerativeModel(MODELS["gemini-flash"])
    resp = model.generate_content(
        prompt,
        generation_config={"temperature": TEMPERATURE, "max_output_tokens": MAX_TOKENS},
    )
    return resp.text


GENERATORS = {"claude-haiku": gen_claude, "gpt-4o-mini": gen_openai, "gemini-flash": gen_gemini}


def generate(model_key: str, prompt: str, retries: int = 3) -> str | None:
    for attempt in range(retries):
        try:
            text = GENERATORS[model_key](prompt)
            if text and len(text.strip()) > 50:
                return text.strip()
            raise ValueError("empty or too-short completion")
        except Exception as e:
            wait = 2 ** (attempt + 1)
            log.warning("%s attempt %d failed (%s); retrying in %ds", model_key, attempt + 1, e, wait)
            time.sleep(wait)
    log.error("%s: giving up on prompt after %d attempts", model_key, retries)
    return None


# ---------------------------------------------------------------- cell plumbing

def load_cells():
    spec = json.loads(CELLS_PATH.read_text())
    cells = spec["cells"]
    fmt = spec["shared_format_instruction"]
    eia = json.loads(EIA_SCENARIOS_PATH.read_text())
    for cid, cell in cells.items():
        if cell.get("inherit_templates_from"):
            src = cells[cell["inherit_templates_from"]]
            cell.setdefault("pos_templates", src["pos_templates"])
            cell.setdefault("neg_templates", src["neg_templates"])
        if cell.get("scenarios") == "EIA_V1":
            cell["scenarios"] = eia
    return cells, fmt


def render(template: str, scenario: dict, fmt: str) -> str:
    class SafeDict(dict):
        def __missing__(self, key):
            return "{" + key + "}"

    return template.format_map(SafeDict(scenario)) + "\n\n" + fmt


def existing_keys(path: Path) -> set:
    keys = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                d = json.loads(line)
                keys.add((d["scenario_id"], d.get("pair_index", d.get("triple_index", 0)), d.get("level", "")))
            except (json.JSONDecodeError, KeyError):
                continue
    return keys


def run_pair_cell(cell_id: str, cell: dict, fmt: str, model_key: str):
    out = OUT_DIR / f"{cell_id}_{model_key}.jsonl"
    done = existing_keys(out)
    n = cell["n_pairs_per_model_per_scenario"]
    pos_t, neg_t = cell["pos_templates"], cell["neg_templates"]
    with open(out, "a") as f:
        for scen in cell["scenarios"]:
            for i in range(n):
                if (scen["id"], i, "") in done:
                    continue
                variant = i % min(len(pos_t), len(neg_t))
                pos = generate(model_key, render(pos_t[variant], scen, fmt))
                neg = generate(model_key, render(neg_t[variant], scen, fmt))
                if not (pos and neg):
                    continue
                f.write(json.dumps({
                    "cell": cell_id, "cell_name": cell["name"],
                    "scenario_id": scen["id"], "pair_index": i,
                    "template_variant": variant, "source_model": MODELS[model_key],
                    "pos_text": pos, "neg_text": neg,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }) + "\n")
                f.flush()
            log.info("[%s/%s] %s done", cell_id, model_key, scen["id"])


def run_severity_cell(cell: dict, fmt: str, model_key: str):
    out = OUT_DIR / f"S_{model_key}.jsonl"
    done = existing_keys(out)
    n = cell["n_triples_per_model_per_scenario"]
    with open(out, "a") as f:
        for scen in cell["scenarios"]:
            for i in range(n):
                texts = {}
                for level, pressure in scen["levels"].items():
                    if (scen["id"], i, level) in done:
                        texts = None
                        break
                    prompt = render(cell["template"], {**scen, "pressure": pressure}, fmt)
                    texts[level] = generate(model_key, prompt)
                if texts is None or not all(texts.values()):
                    continue
                for level, text in texts.items():
                    f.write(json.dumps({
                        "cell": "S", "cell_name": cell["name"],
                        "scenario_id": scen["id"], "triple_index": i, "level": level,
                        "source_model": MODELS[model_key], "text": text,
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                    }) + "\n")
                f.flush()
            log.info("[S/%s] %s done", model_key, scen["id"])


def run_robustness_cell(cell: dict):
    """Perturb cell-A pairs (paraphrase / name swap / register shift) with claude-haiku."""
    sources = []
    for mk in MODELS:
        p = OUT_DIR / f"A_{mk}.jsonl"
        if p.exists():
            sources += [json.loads(l) for l in p.read_text().splitlines()]
    if not sources:
        log.error("Cell R needs cell A output first — run cells A before R")
        return
    sources = sources[: cell["n_source_pairs"]]
    out = OUT_DIR / "R_claude-haiku.jsonl"
    done = existing_keys(out)
    with open(out, "a") as f:
        for i, src in enumerate(sources):
            for pert in cell["perturbations"]:
                pert_key = pert.split(" ")[0]
                if (f"{src['scenario_id']}~{pert_key}", i, "") in done:
                    continue
                texts = {}
                for side in ("pos_text", "neg_text"):
                    prompt = (
                        f"Rewrite the following text applying this transformation: {pert}. "
                        "Preserve every decision, action, and factual claim exactly. "
                        "Return ONLY the rewritten text, plain text, no preamble.\n\n" + src[side]
                    )
                    texts[side] = generate("claude-haiku", prompt)
                if not all(texts.values()):
                    continue
                f.write(json.dumps({
                    "cell": "R", "cell_name": cell["name"],
                    "scenario_id": f"{src['scenario_id']}~{pert_key}", "pair_index": i,
                    "perturbation": pert_key, "source_cell": "A",
                    "source_model": src["source_model"], "perturbed_by": MODELS["claude-haiku"],
                    "pos_text": texts["pos_text"], "neg_text": texts["neg_text"],
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }) + "\n")
                f.flush()
            if i % 5 == 0:
                log.info("[R] %d/%d source pairs perturbed", i + 1, len(sources))


def count_calls(cells, selected):
    total = 0
    for cid in selected:
        cell = cells[cid]
        if cid == "R":
            total += cell["n_source_pairs"] * len(cell["perturbations"]) * 2
        elif cid == "S":
            total += len(cell["scenarios"]) * cell["n_triples_per_model_per_scenario"] * 3 * len(MODELS)
        else:
            total += len(cell["scenarios"]) * cell["n_pairs_per_model_per_scenario"] * 2 * len(MODELS)
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cells", nargs="+", default=["A", "B", "D", "E", "F", "G", "H", "S"])
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS),
                   help="subset of providers (claude cells are cheaper via generate_v2_1_claude_batch.py)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    load_env()
    cells, fmt = load_cells()
    selected = [c for c in args.cells if c in cells]

    calls = count_calls(cells, selected)
    log.info("cells %s -> ~%d API calls (~$%.2f at small-model rates)", selected, calls, calls * 0.0015)
    if args.dry_run:
        for cid in selected:
            cell = cells[cid]
            n_scen = len(cell["scenarios"]) if isinstance(cell["scenarios"], list) else 0
            log.info("  %s (%s): %d scenarios", cid, cell["name"], n_scen)
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def run_model(model_key):
        for cid in selected:
            if cid == "R":
                continue
            if cid == "S":
                run_severity_cell(cells["S"], fmt, model_key)
            else:
                run_pair_cell(cid, cells[cid], fmt, model_key)
        log.info("=== %s complete ===", model_key)

    with ThreadPoolExecutor(max_workers=len(args.models)) as ex:
        list(ex.map(run_model, args.models))

    if "R" in selected:
        run_robustness_cell(cells["R"])
    log.info("all done -> %s", OUT_DIR)


if __name__ == "__main__":
    main()
