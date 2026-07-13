"""Frozen inventory of direct weight-edit CLIs (Integrity Repair A QA Q4).

Every script that calls the low-level weight-edit functions
(``orthogonalize_component`` / ``orthogonalize_component_measured``) must be
classified here. Two classifications exist:

  * ``shared_contract`` — migrated to the shared accepted/exploratory
    evidence-run contract (``--run-mode``, direction-bound component
    resolution, immutable revisions, fail-closed outputs).
  * ``historical_or_exploratory_only`` — NOT migrated. These scripts carry a
    module-level ``EVIDENCE_ELIGIBILITY = "historical_or_exploratory_only"``
    constant, persist it in every artifact they emit, and expose NO accepted
    mode. Their outputs are permanently ineligible for confirmatory claims.

The static acceptance test (``tests/test_weight_edit_inventory.py``) AST-scans
``src`` for direct-edit call sites and fails when a caller is missing from
this frozen inventory, so a newly added direct-edit CLI cannot ship
unclassified.

Independently refreshed at architect HEAD ``e256dcf`` (and re-refreshed with
``e14d_random_component_sets.py``): ``analysis/behavioral_dla.py`` imports
only the read-only pair loader and is not a weight-edit caller;
``action_path_mediation.py`` uses activation hooks, not parameter edits.
"""

import ast
from pathlib import Path

SHARED_CONTRACT = "shared_contract"
HISTORICAL_ONLY = "historical_or_exploratory_only"

WEIGHT_EDIT_FUNCTIONS = (
    "orthogonalize_component",
    "orthogonalize_component_measured",
)

#: Frozen classification, keyed by repo-relative POSIX path.
WEIGHT_EDIT_CALLER_CLASSIFICATION = {
    # migrated to the shared accepted/exploratory contract
    "src/weight_orthogonalization.py": SHARED_CONTRACT,
    "src/capability_eval.py": SHARED_CONTRACT,
    "src/norm_matched_controls.py": SHARED_CONTRACT,
    "src/e17b_null_audit.py": SHARED_CONTRACT,
    "src/evaluation/gate0b_task_control.py": SHARED_CONTRACT,
    "src/eia_validation/run_eia_local.py": SHARED_CONTRACT,
    "src/eia_validation/e27_game_variants.py": SHARED_CONTRACT,
    # explicitly non-evidential until individually migrated
    "src/e17_stage3.py": HISTORICAL_ONLY,
    "src/e18_interaction.py": HISTORICAL_ONLY,
    "src/e14d_random_component_sets.py": HISTORICAL_ONLY,
    "src/e26_format_stress.py": HISTORICAL_ONLY,
    "src/e26_matched_nulls.py": HISTORICAL_ONLY,
    "src/e28b_slope_nulls.py": HISTORICAL_ONLY,
    "src/analysis/logit_lens_trajectory.py": HISTORICAL_ONLY,
}


def _calls_weight_edit(tree):
    """True when the AST contains a call to a weight-edit function (by bare
    name or attribute, covering ``module.orthogonalize_component(...)``)."""
    bare_names = set(WEIGHT_EDIT_FUNCTIONS)
    module_aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in WEIGHT_EDIT_FUNCTIONS:
                    bare_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.endswith(("weight_orthogonalization",
                                        "norm_matched_controls")):
                    module_aliases.add(alias.asname or alias.name.split(".")[0])
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                if func.id in bare_names:
                    return True
            elif isinstance(func, ast.Attribute):
                # Attribute calls are recognized by function name; the module
                # alias set is retained to document and test import-alias
                # handling without missing nested module attributes.
                if func.attr in WEIGHT_EDIT_FUNCTIONS:
                    return True
    return False


def scan_direct_edit_callers(repo_root=None):
    """AST-scan ``src`` for files that CALL a weight-edit function.

    Returns a sorted list of repo-relative POSIX paths. Files that fail to
    parse are reported as callers (fail closed: an unparseable file cannot
    prove it does not edit weights).
    """
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[1]
    callers = []
    for path in sorted((root / "src").rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError:
            callers.append(rel)
            continue
        if _calls_weight_edit(tree):
            callers.append(rel)
    return callers
