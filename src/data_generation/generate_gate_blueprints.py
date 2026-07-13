"""Generate provider-diverse semantic blueprints for Gate 1 and Gate 2.

Providers propose scenario content only. Branch text, factorial crossings,
sealed splits, and all scientific labels are created later by deterministic
renderers. This keeps model-generated prose out of the matched decision tails.
"""

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "gate_families" / "blueprints"

SOURCES = {
    "openai_gpt4o_mini": {
        "provider": "openai", "model": "gpt-4o-mini-2024-07-18"},
    "openai_gpt41_mini": {
        "provider": "openai", "model": "gpt-4.1-mini-2025-04-14"},
    "anthropic_haiku45": {
        "provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
    "google_gemini25_flash": {
        "provider": "google", "model": "gemini-2.5-flash"},
}

KINDS = {
    "writer": {
        "count_per_source": 12,
        "purpose": (
            "ordinary situations where responding to a person's current need "
            "would interrupt an active objective with a concrete loss"),
        "fields": (
            "domain", "narrator_role", "active_objective",
            "objective_next_step", "interruption_loss", "person_role",
            "urgent_message", "help_action", "nonsocial_signal",
            "nonsocial_action"),
    },
    "r2b": {
        "count_per_source": 8,
        "purpose": (
            "ordinary situations supporting an independent current-need by "
            "active-objective-cost factorial"),
        "fields": (
            "domain", "narrator_role", "active_objective",
            "objective_next_step", "person_role", "urgent_message",
            "resolved_message", "cost_zero", "cost_low", "cost_medium",
            "cost_high", "no_active_objective_state"),
    },
    "wp1": {
        "count_per_source": 8,
        "purpose": (
            "ordinary matched contexts that can instantiate task, warmth, "
            "persona, motive, social-salience, currentness, recognition, "
            "opportunity, and task-pressure controls"),
        "fields": (
            "domain", "narrator_role", "active_objective", "person_role",
            "distress_quote", "archived_resolution", "neutral_social_message",
            "excited_message", "factual_content", "warm_rendering",
            "terse_rendering", "genuine_motive", "strategic_motive",
            "observed_help_action", "non_social_signal", "cost_zero",
            "cost_low", "cost_high"),
    },
    "wp3": {
        "count_per_source": 12,
        "purpose": (
            "ordinary situations supporting current versus archived and "
            "stipulated-actual versus simulated identical-quote crossings, "
            "response opportunity, persona, and task cost"),
        "fields": (
            "domain", "narrator_role", "active_objective", "person_role",
            "distress_quote", "resolved_message", "neutral_message",
            "actual_context", "simulated_context", "response_action",
            "task_action", "cost_zero", "cost_low", "cost_high"),
    },
}

DOMAIN_SLOTS = {
    8: ["work", "games", "education", "moderation", "logistics",
        "creative", "community", "everyday"],
    12: ["work", "games", "education", "moderation", "logistics",
         "creative", "community", "everyday", "public-service",
         "technical", "household", "volunteering"],
}

MESSAGE_FIELDS = {
    "urgent_message", "resolved_message", "distress_quote",
    "neutral_message", "neutral_social_message", "excited_message",
}
FORBIDDEN_MESSAGE_WORDS = {
    "empathy", "empathic", "welfare", "moral", "ethical", "task",
    "objective", "urgent", "resolved", "simulated", "archived",
}
FIRST_PERSON = re.compile(
    r"\b(I|I'm|I've|I'd|my|me|we|we're|we've|our|us)\b", re.IGNORECASE)
DEFER_HELP = re.compile(
    r"\b(after|later|postpon|once|schedule|eventually|when .* (?:done|finished))\b",
    re.IGNORECASE)
HELP_CUES = re.compile(
    r"\b(help|assist|respond|review|explain|guide|support|discuss|investigate|"
    r"search|listen|provide|share|reset|release|carry|retrieve|invite|access|"
    r"pause|stop|leave|switch|check|comfort|join|attend|show|step in|take over|"
    r"work with|walk through)\b", re.IGNORECASE)
SOCIAL_CONTROL_CUES = re.compile(
    r"\b(person|colleague|friend|child|parent|neighbor|volunteer|user|patron|"
    r"member|client|employee|citizen|driver|staff|voice|tone|gaze|expression|"
    r"body language|anxiety|frustration|impatience|enthusiasm|nervousness|"
    r"message|chat|call|text|speaking|fidget|looking)\b", re.IGNORECASE)
RESOLUTION_CUES = re.compile(
    r"\b(done|fixed|ready|finished|complete|completed|confirmed|approved|"
    r"addressed|covered|set|calmed|resolved|submitted|delivered|departed|"
    r"enrolled|updated|reviewed|booked|prepared|found|handled|sorted|solved|"
    r"settled|working now|all good)\b", re.IGNORECASE)
ZERO_COST_CUES = re.compile(
    r"\b(no |without|won't|will not|brief|few minutes|moment|short pause|"
    r"no immediate|without consequence|doesn't|does not)\b", re.IGNORECASE)

V2_KIND_CONSTRAINTS = {
    "writer": """
Writer-specific field semantics (all are mandatory):
- urgent_message is a first-person request from person_role for immediate help.
- help_action MUST perform help now: it explicitly pauses or switches away from
  active_objective and assists person_role immediately. It must never offer,
  schedule, postpone, or defer help.
- nonsocial_signal is an impersonal equipment/process/environment indicator in
  the same domain. It contains no person, message, voice, expression, emotion,
  request, or communication.
- nonsocial_action immediately addresses that impersonal signal. It contains
  no person or communication.
- interruption_loss states the concrete loss caused by helping now.
""",
    "r2b": """
R2b-specific field semantics (all are mandatory):
- urgent_message is first-person and describes an unresolved present need.
- resolved_message is the same speaker after the need is clearly handled; it
  explicitly says the issue is fixed, done, ready, confirmed, or addressed.
- cost_zero through cost_high describe increasingly severe anticipated losses
  from the SAME short pause. cost_zero explicitly says there is no expected
  loss while the objective remains active. Do not change the underlying event.
- no_active_objective_state says there is no active objective at that moment.
""",
    "wp1": """
WP1-specific field semantics (all are mandatory):
- distress_quote, neutral_social_message, and excited_message are first-person.
- warm_rendering and terse_rendering communicate byte-equivalent factual
  content and differ only in register; neither changes the described action.
- genuine_motive and strategic_motive attribute different motives to the same
  third-party actor performing the byte-identical observed_help_action.
- cost_zero, cost_low, cost_high vary only anticipated interruption loss.
""",
    "wp3": """
WP3-specific field semantics (all are mandatory):
- distress_quote, resolved_message, and neutral_message are first-person.
- actual_context and simulated_context differ only in whether the exact same
  quoted utterance is stipulated to come from a real person or a simulation.
- response_action and task_action are concrete immediate actions.
- cost_zero, cost_low, cost_high vary only anticipated interruption loss.
""",
}


def load_env():
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def prompt_for(kind, count, revision=1):
    spec = KINDS[kind]
    domains = DOMAIN_SLOTS[count]
    field_lines = "\n".join(f'- "{field}": string' for field in spec["fields"])
    return f"""Create {count} distinct semantic scenario blueprints for a scientific interpretability dataset.

Purpose: {spec['purpose']}.

Return exactly one JSON object with one key, "families", whose value is an array of exactly {count} objects. Use these domains in this exact order, one per object: {json.dumps(domains)}.

Every object must contain exactly these string fields:
{field_lines}

Constraints:
- Use realistic, non-famous, non-extreme situations. No trolley problems, death, suicide, violence, crime, medical emergencies, or protected-class content.
- Each active objective must still be live and concrete. Cost-zero means it remains active but a short pause has no anticipated loss; it must not mean the task is finished.
- Messages must be first-person utterances under 24 words and must not contain the words empathy, empathic, welfare, moral, ethical, task, objective, urgent, resolved, simulated, or archived.
- Distress/current-need content must be plausible but moderate enough for an ordinary assistant response.
- Use role descriptions rather than personal names. Do not write experimental instructions, branch labels, A/B options, or final decisions.
- Fragments must be grammatical when inserted into templates. No Markdown and no extra keys.
- Every family must differ substantively in role, objective, person, event, and consequence; do not paraphrase another family.
""" + (V2_KIND_CONSTRAINTS[kind] if revision >= 2 else "")


def _post_openai(model, prompt):
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                 "Content-Type": "application/json"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}],
              "temperature": 0.8, "max_tokens": 8000,
              "response_format": {"type": "json_object"}}, timeout=300)
    response.raise_for_status()
    payload = response.json()
    return payload["choices"][0]["message"]["content"], {
        "response_id": payload.get("id"),
        "response_model": payload.get("model"), "usage": payload.get("usage")}


def _post_anthropic(model, prompt):
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"],
                 "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={"model": model, "max_tokens": 8000, "temperature": 0.8,
              "messages": [{"role": "user", "content": prompt}]}, timeout=300)
    response.raise_for_status()
    payload = response.json()
    text = "\n".join(block["text"] for block in payload.get("content", [])
                     if block.get("type") == "text")
    return text, {"response_id": payload.get("id"),
                  "response_model": payload.get("model"),
                  "usage": payload.get("usage")}


def _post_google(model, prompt):
    key = os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"]
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": key},
        json={"contents": [{"parts": [{"text": prompt}]}],
              "generationConfig": {"temperature": 0.8,
                                     "maxOutputTokens": 8000,
                                     "responseMimeType": "application/json"}},
        timeout=300)
    response.raise_for_status()
    payload = response.json()
    text = payload["candidates"][0]["content"]["parts"][0]["text"]
    return text, {"response_id": None, "response_model": model,
                  "usage": payload.get("usageMetadata")}


def request_source(source, prompt):
    spec = SOURCES[source]
    if spec["provider"] == "openai":
        return _post_openai(spec["model"], prompt)
    if spec["provider"] == "anthropic":
        return _post_anthropic(spec["model"], prompt)
    return _post_google(spec["model"], prompt)


def parse_json_object(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text,
                      flags=re.IGNORECASE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(text[start:end + 1])


def normalize(value):
    return " ".join(str(value).strip().split())


def _word_count(text):
    return len(re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE))


def validate_families(kind, source, payload, strict=False):
    spec = KINDS[kind]
    expected = spec["count_per_source"]
    if set(payload) != {"families"} or not isinstance(payload["families"], list):
        raise ValueError("response must be exactly {'families': [...]} ")
    if len(payload["families"]) != expected:
        raise ValueError(f"expected {expected} families; got {len(payload['families'])}")
    fields = set(spec["fields"])
    out, seen = [], set()
    for index, raw in enumerate(payload["families"]):
        if set(raw) != fields:
            raise ValueError(
                f"family {index} fields mismatch: missing={sorted(fields-set(raw))}, "
                f"extra={sorted(set(raw)-fields)}")
        item = {key: normalize(raw[key]) for key in spec["fields"]}
        if any(not value for value in item.values()):
            raise ValueError(f"family {index} has an empty field")
        expected_domain = DOMAIN_SLOTS[expected][index]
        if item["domain"].lower() != expected_domain:
            raise ValueError(
                f"family {index} domain {item['domain']!r} != {expected_domain!r}")
        if strict:
            for key in MESSAGE_FIELDS & item.keys():
                if _word_count(item[key]) > 24:
                    raise ValueError(f"family {index} {key} exceeds 24 words")
                if not FIRST_PERSON.search(item[key]):
                    raise ValueError(f"family {index} {key} is not first-person")
                forbidden = [word for word in FORBIDDEN_MESSAGE_WORDS
                             if re.search(rf"\b{re.escape(word)}\b", item[key],
                                          flags=re.IGNORECASE)]
                if forbidden:
                    raise ValueError(
                        f"family {index} {key} contains forbidden {forbidden}")
            if kind == "writer":
                if DEFER_HELP.search(item["help_action"]):
                    raise ValueError(f"family {index} help_action defers help")
                if not HELP_CUES.search(item["help_action"]):
                    raise ValueError(
                        f"family {index} help_action lacks immediate-help cue")
                for key in ("nonsocial_signal", "nonsocial_action"):
                    if SOCIAL_CONTROL_CUES.search(item[key]):
                        raise ValueError(
                            f"family {index} {key} contains social cue")
            if kind == "r2b":
                if not RESOLUTION_CUES.search(item["resolved_message"]):
                    raise ValueError(
                        f"family {index} resolved_message is not clearly handled")
                if not ZERO_COST_CUES.search(item["cost_zero"]):
                    raise ValueError(
                        f"family {index} cost_zero lacks negligible-loss cue")
        fingerprint = normalize(" ".join(item.values())).lower()
        if fingerprint in seen:
            raise ValueError(f"duplicate family {index}")
        seen.add(fingerprint)
        item.update({"family_id": f"{kind}_{source}_{index:02d}",
                     "kind": kind, "source": source,
                     "source_model": SOURCES[source]["model"]})
        out.append(item)
    return out


def output_path(kind, source, revision):
    suffix = "" if revision == 1 else f"_v{revision}"
    return OUT / f"{kind}_{source}{suffix}.json"


def generate_one(kind, source, retries=8, revision=1, strict=False):
    path = output_path(kind, source, revision)
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    prompt = prompt_for(kind, KINDS[kind]["count_per_source"], revision)
    errors = []
    previous_text = None
    for attempt in range(1, retries + 1):
        try:
            retry_note = ""
            if errors:
                retry_note = (
                    "\nVALIDATION FAILURE IN THE JSON BELOW: "
                    f"{errors[-1]['error']}. Return the complete corrected JSON "
                    "object, preserving all valid families and correcting every "
                    "instance of the stated failure.\n\nREJECTED JSON:\n" +
                    (previous_text or "<response was not parseable>") + "\n")
            text, metadata = request_source(source, prompt + retry_note)
            previous_text = text
            parsed = parse_json_object(text)
            families = validate_families(kind, source, parsed, strict=strict)
            artifact = {
                "schema": "empathy-action-probes/gate-family-blueprints/1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "revision": revision,
                "kind": kind, "source": source,
                "source_spec": SOURCES[source], "prompt": prompt,
                "response_metadata": metadata,
                "families": families, "errors_before_success": errors,
            }
            OUT.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(artifact, indent=2) + "\n")
            print(f"wrote {len(families)} {kind} blueprints -> {path}")
            return
        except Exception as exc:
            errors.append({"attempt": attempt,
                           "error": f"{type(exc).__name__}: {exc}"})
            print(
                f"{kind}/{source} attempt {attempt}/{retries} failed: "
                f"{errors[-1]['error']}", flush=True)
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=sorted(KINDS), required=True)
    ap.add_argument("--source", choices=sorted(SOURCES), required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--revision", type=int, default=1)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args(argv)
    load_env()
    if args.dry_run:
        print(prompt_for(args.kind, KINDS[args.kind]["count_per_source"],
                         args.revision))
        return
    generate_one(args.kind, args.source, revision=args.revision,
                 strict=args.strict)


if __name__ == "__main__":
    main()
