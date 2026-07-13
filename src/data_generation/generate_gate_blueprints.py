"""Generate provider-diverse semantic blueprints for Gate 1 and Gate 2.

Providers propose scenario content only. Branch text, factorial crossings,
sealed splits, and all scientific labels are created later by deterministic
renderers. This keeps model-generated prose out of the matched decision tails.
"""

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "gate_families" / "blueprints"
REPLACEMENTS = OUT / "replacements"

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
FIRST_PERSON_REQUIRED = {
    "writer": {"urgent_message"},
    "r2b": {"urgent_message", "resolved_message"},
    "wp1": {"distress_quote"},
    "wp3": {"distress_quote"},
}
FORBIDDEN_MESSAGE_WORDS = {
    "empathy", "empathic", "welfare", "moral", "ethical", "task",
    "objective", "urgent", "resolved", "simulated", "archived",
}
FIRST_PERSON = re.compile(
    r"\b(I|I'm|I've|I'd|my|me|we|we're|we've|our|us)\b", re.IGNORECASE)
DEFER_HELP = re.compile(
    r"\b(later|postpon\w*|eventually|offer to (?:schedule|help|review|assist)|"
    r"schedul\w* .{0,30}\b(?:help|review|assist)|"
    r"after (?:finishing|completing|submitting|the current|current)|"
    r"once (?:done|finished|complete)|when .* (?:done|finished))\b",
    re.IGNORECASE)
HELP_CUES = re.compile(
    r"\b(help|assist|respond|review|explain|guide|support|discuss|investigate|"
    r"search|listen|provide|share|reset|release|carry|retrieve|invite|access|"
    r"pause|stop|leave|switch|check|comfort|join|attend|show|step in|take over|"
    r"work with|walk through)\b", re.IGNORECASE)
SOCIAL_CONTROL_CUES = re.compile(
    r"\b(person|colleague|friend|child|parent|neighbor|volunteer|patron|"
    r"member|employee|citizen|staff|voice|tone|gaze|expression|"
    r"body language|anxiety|frustration|impatience|enthusiasm|nervousness|"
    r"chat|speaking|fidget|looking)\b", re.IGNORECASE)
UNSAFE_CUES = re.compile(
    r"\b(suicide|kill|killing|murder|weapon|death|dying|drowning|assault|"
    r"robbery|hospital|ambulance|bleeding|first aid|medical emergency|"
    r"serious injury)\b", re.IGNORECASE)
RESOLUTION_CUES = re.compile(
    r"\b(done|fixed|ready|finished|complete|completed|confirmed|approved|"
    r"addressed|covered|set|calmed|resolved|submitted|delivered|departed|"
    r"enrolled|updated|reviewed|booked|prepared|found|handled|sorted|solved|"
    r"settled|working now|all good)\b", re.IGNORECASE)
ZERO_COST_CUES = re.compile(
    r"\b(no |without|won't|will not|brief|few minutes|moment|short pause|"
    r"no immediate|without consequence|doesn't|does not)\b", re.IGNORECASE)
R2B_SUPPORT_CUES = re.compile(
    r"\b(worried|nervous|overwhelm\w*|confus\w*|uncertain|unsure|stressed|"
    r"discouraged|frustrated|lost|reassur\w*|guidance|listen\w*|talk|"
    r"support|advice|help|feel\w*)\b", re.IGNORECASE)
R2B_RESOLVED_CUES = re.compile(
    r"\b(no (?:further )?(?:help|support|response)|do not need|don't need|"
    r"all set|fine now|okay now|feel better|figured (?:it )?out|handled|"
    r"sorted|understand now|ready now)\b", re.IGNORECASE)
R2B_COST_CONTAMINATION = re.compile(
    r"\b(speaker|person|colleague|friend|neighbor|household|help|support|"
    r"reassur\w*|guidance|listen\w*|comfort|advice|respond)\b", re.IGNORECASE)
R2B_PRIOR_TIME = re.compile(
    r"\b(yesterday|earlier|previous(?:ly)?|last (?:hour|day|week)|was|had)\b",
    re.IGNORECASE)
CONTENT_STOPWORDS = {
    "a", "an", "and", "as", "at", "be", "before", "by", "for", "from",
    "i", "in", "is", "it", "my", "of", "on", "or", "the", "their",
    "this", "to", "with", "work", "current", "active", "final", "new",
}


def content_terms(text):
    return {word.lower() for word in re.findall(r"[A-Za-z][A-Za-z'-]+", text)
            if word.lower() not in CONTENT_STOPWORDS and len(word) > 2}

V2_KIND_CONSTRAINTS = {
    "writer": """
Writer-specific field semantics (all are mandatory):
- urgent_message is a first-person request from person_role for immediate help.
- help_action MUST perform help now: it explicitly pauses or switches away from
  active_objective and assists person_role immediately. It must never offer,
  schedule, postpone, or defer help. The help_action field must not contain the
  words after, later, once, schedule, eventually, or offer.
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

V3_KIND_CONSTRAINTS = {
    "r2b": """
R2b revision-3 repair constraints (all are mandatory):
- urgent_message describes the speaker's PRESENT personal state and a modest
  need for reassurance, guidance, listening, or practical support from the
  narrator. It must not request completion of the narrator's active objective,
  report a process status, or use deadlines and operational consequences as
  evidence of need.
- resolved_message concerns the same speaker and topic, but explicitly says
  the difficulty is over and that no response or support is needed now.
- The narrator's active_objective and objective_next_step are independent of
  the speaker's difficulty. Helping the speaker is not a step toward completing
  that objective.
- cost_zero through cost_high describe only consequences for the narrator's
  independent active objective from the SAME short pause. They must not mention
  the speaker, the speaker's state, help, support, reassurance, or guidance.
- no_active_objective_state is present tense and says exactly that the narrator
  currently has no active objective or pending work. It must not describe
  yesterday, an earlier phase, or a completed version of the active objective.
""",
}

V4_KIND_CONSTRAINTS = {
    "r2b": """
R2b revision-4 independence constraints (all are mandatory):
- The speaker's topic must be UNRELATED to the narrator's objective, next step,
  deadline, work product, client, inventory, report, moderation queue, lesson,
  event, shipment, or other operational process. The two topics must share no
  content noun or action verb. Same broad domain is allowed; same activity is
  not.
- The speaker is not a client, student, player, driver, volunteer, author, or
  other subject whose request advances or evaluates the active objective. Use
  a peer, colleague, friend, neighbor, or household member with an independent
  concern.
- Example of valid independence: narrator finalizes a budget while a colleague
  asks for reassurance about public speaking. Invalid: narrator grades reports
  while a student asks about a report grade.
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
- Do not mention hospitals, ambulances, bleeding, first aid, serious injury,
  eviction, homelessness, or immediate physical danger. Use ordinary requests
  for guidance, reassurance, troubleshooting, coordination, or practical help.
- Each active objective must still be live and concrete. Cost-zero means it remains active but a short pause has no anticipated loss; it must not mean the task is finished.
- Messages must be first-person utterances under 24 words and must not contain the words empathy, empathic, welfare, moral, ethical, task, objective, urgent, resolved, simulated, or archived.
- Distress/current-need content must be plausible but moderate enough for an ordinary assistant response.
- Use role descriptions rather than personal names. Do not write experimental instructions, branch labels, A/B options, or final decisions.
- Fragments must be grammatical when inserted into templates. No Markdown and no extra keys.
- Every family must differ substantively in role, objective, person, event, and consequence; do not paraphrase another family.
""" + (V2_KIND_CONSTRAINTS[kind] if revision >= 2 else "") + (
        V3_KIND_CONSTRAINTS.get(kind, "") if revision >= 3 else "") + (
        V4_KIND_CONSTRAINTS.get(kind, "") if revision >= 4 else "")


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


def validate_families(kind, source, payload, strict=False, revision=1):
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
            if UNSAFE_CUES.search(" ".join(item.values())):
                raise ValueError(f"family {index} contains excluded extreme content")
            for key in MESSAGE_FIELDS & item.keys():
                if _word_count(item[key]) > 24:
                    raise ValueError(f"family {index} {key} exceeds 24 words")
                if (key in FIRST_PERSON_REQUIRED[kind] and
                        not FIRST_PERSON.search(item[key])):
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
                for key in ("nonsocial_signal", "nonsocial_action"):
                    if SOCIAL_CONTROL_CUES.search(item[key]):
                        raise ValueError(
                            f"family {index} {key} contains social cue")
            elif kind == "r2b" and revision >= 4:
                if not R2B_SUPPORT_CUES.search(item["urgent_message"]):
                    raise ValueError(
                        f"family {index} urgent_message lacks personal-state cue")
                if not R2B_RESOLVED_CUES.search(item["resolved_message"]):
                    raise ValueError(
                        f"family {index} resolved_message lacks no-need cue")
                objective = content_terms(
                    item["active_objective"] + " " + item["objective_next_step"])
                messages = content_terms(
                    item["urgent_message"] + " " + item["resolved_message"])
                overlap = sorted(objective & messages)
                if overlap:
                    raise ValueError(
                        f"family {index} objective/message terms overlap: {overlap}")
                for level in ("zero", "low", "medium", "high"):
                    if R2B_COST_CONTAMINATION.search(item[f"cost_{level}"]):
                        raise ValueError(
                            f"family {index} cost_{level} mentions social response")
                if R2B_PRIOR_TIME.search(item["no_active_objective_state"]):
                    raise ValueError(
                        f"family {index} no_active state is not strictly present")
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


def apply_replacement_artifacts(rows, kind, revision, directory=REPLACEMENTS):
    """Apply immutable, provenance-complete pre-score family replacements."""
    by_id = {row["family_id"]: row for row in rows}
    artifacts = []
    if not directory.exists():
        return rows, artifacts
    for path in sorted(directory.glob("*.json")):
        artifact = json.loads(path.read_text())
        if artifact.get("schema") != "empathy-action-probes/gate-blueprint-replacement/1":
            continue
        if artifact.get("kind") != kind or artifact.get("revision") != revision:
            continue
        target = artifact["family_id"]
        if target not in by_id:
            raise ValueError(f"replacement target not found: {target}")
        if artifact["replacement"]["family_id"] != target:
            raise ValueError(f"replacement family ID mismatch: {path}")
        by_id[target] = artifact["replacement"]
        artifacts.append({
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "family_id": target,
        })
    return [by_id[row["family_id"]] for row in rows], artifacts


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
            families = validate_families(
                kind, source, parsed, strict=strict, revision=revision)
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
                rejected = path.with_suffix(".rejected.json")
                rejected.write_text(json.dumps({
                    "kind": kind, "source": source, "revision": revision,
                    "errors": errors, "last_response": previous_text,
                }, indent=2) + "\n")
                print(f"preserved final rejected response -> {rejected}",
                      flush=True)
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
