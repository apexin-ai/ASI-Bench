"""Public Task files must follow the explicit Example exception policy."""

import json
import re
import subprocess
from pathlib import Path, PurePosixPath

import yaml

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "tasks"
POLICY_PATH = ROOT / "config" / "public_examples.json"
PROTECTED_FILENAMES = frozenset({
    "task.yaml",
    "task_eval.yaml",
    "generate_gt.py",
    "precompute_gt.py",
    "custom_scorer.py",
    "reference_specs.md",
    ".env",
    ".env.local",
    "credentials.json",
    "secrets.json",
})
PROTECTED_DIRECTORY_NAMES = frozenset({
    "reference",
    "references",
    "ground_truth",
    "ground-truth",
    "answers",
    "answer_key",
    "answer-key",
    "private",
    "internal",
})
PROTECTED_SUFFIXES = frozenset({".pem", ".key", ".p12", ".pfx"})
PROTECTED_PREFIXES = (
    "ground_truth.",
    "ground-truth.",
    "reference_answer.",
    "reference-answer.",
    "answer_key.",
    "answer-key.",
)
EXPECTED_EXAMPLES = {
    "_template",
    "chemistry.bsse_counterpoise_cbs_extrapolation",
    "materials.phonon_dispersion",
    "medicine.ethics_disclosure_diagnosis_1670",
    "medicine.jama_id0014_malignant_an",
    "robotics.minimum_snap_trajectory_conditioning",
}

EXPECTED_FULL_EXAMPLE_FILES = {
    "chemistry.bsse_counterpoise_cbs_extrapolation": {
        "task.yaml", "prompt_b1.md", "prompt_b2.md", "prompt_b3.md", "prompt_b4.md",
        "generate_gt.py", "custom_scorer.py", "reference_specs.md",
    },
    "materials.phonon_dispersion": {
        "task.yaml", "prompt_b1.md", "prompt_b2.md", "prompt_b3.md", "prompt_b4.md",
        "generate_gt.py", "reference_specs.md",
    },
    "medicine.ethics_disclosure_diagnosis_1670": {
        "task.yaml", "prompt_b1.md", "prompt_b2.md", "prompt_b3.md", "prompt_b4.md",
        "generate_gt.py", "custom_scorer.py",
    },
    "medicine.jama_id0014_malignant_an": {
        "task.yaml", "prompt_b1.md", "prompt_b2.md", "prompt_b3.md", "prompt_b4.md",
        "generate_gt.py", "custom_scorer.py",
    },
    "robotics.minimum_snap_trajectory_conditioning": {
        "task.yaml", "prompt_b1.md", "prompt_b2.md", "prompt_b3.md", "prompt_b4.md",
        "generate_gt.py", "custom_scorer.py", "reference_specs.md",
    },
}

SEED42_RESTORED_TASKS = {
    "astronomy.planet_activity_sep",
    "biostat.perturbation_convergence_testing",
    "computer_science.affinity_subset_bound_audit",
    "computer_science.euclidean_tsp_tour_optimization",
    "computer_science.max3sat_assignment_optimization",
    "earth_science.2_e_soil_carbon_nitrogen_cycling",
    "earth_science.richards_topmodel_hydrology",
    "math.mpsc_safety_filter",
    "math.sparse_hyperbolic_recovery_2d",
    "physics.ag_strip_dielectric_inference",
    "physics.dp_contact_process",
    "physics.lbm_square_wake",
    "physics.paired_nanorod_negative_index_retrieval",
    "robotics.multiregime_icp_degeneracy",
    "robotics.rnea_inverse_dynamics_dh",
}

EXPECTED_SAMPLE_TASKS = {
    "chemistry.bsse_counterpoise_cbs_extrapolation",
    "materials.phonon_dispersion",
    "medicine.ethics_disclosure_diagnosis_1670",
    "medicine.jama_id0014_malignant_an",
    "robotics.minimum_snap_trajectory_conditioning",
}

SENSITIVE_CONTENT_PATTERNS = {
    "private key": re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
    ),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "Slack token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    "long API key": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b"),
    "Portal PAT": re.compile(r"\basi_pat_[A-Za-z0-9_-]{20,}\b"),
}
_BENCHMARK_REPOSITORY_STEM = r"(?:a" + r"si|ai4sci)[-_]?bench"
_NONPUBLIC_MARKER = r"(?:" + "inter" + r"nal|pri" + r"vate)"
BLOCKED_INTERNAL_PATTERNS = {
    "internal/private repository identifier": re.compile(
        r"(?:"
        + _BENCHMARK_REPOSITORY_STEM
        + r"[A-Za-z0-9_.-]*[-_]"
        + _NONPUBLIC_MARKER
        + r"|"
        + _NONPUBLIC_MARKER
        + r"[-_][A-Za-z0-9_.-]*"
        + _BENCHMARK_REPOSITORY_STEM
        + r")",
        re.IGNORECASE,
    ),
}
BLOCKED_TRACKED_FILENAMES = frozenset({
    ".env",
    ".env.local",
    ".env.production",
    "credentials.json",
    "secrets.json",
})
BLOCKED_TRACKED_SUFFIXES = frozenset({
    ".7z", ".gz", ".jks", ".key", ".kdbx", ".p12", ".pem", ".pfx",
    ".rar", ".sqlite", ".tar", ".tgz", ".zip",
})


def _load_policy() -> dict:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def _task_dir(task_id: str) -> Path:
    if task_id == "_template":
        return TASKS / task_id
    domain, name = task_id.split(".", maxsplit=1)
    return TASKS / domain / name


def _is_private_like(relative_path: PurePosixPath) -> bool:
    parts_lower = tuple(part.lower() for part in relative_path.parts)
    name = parts_lower[-1]
    directory_parts = parts_lower[:-1]
    return (
        name in PROTECTED_FILENAMES
        or any(part in PROTECTED_DIRECTORY_NAMES for part in directory_parts)
        or any(name.startswith(prefix) for prefix in PROTECTED_PREFIXES)
        or PurePosixPath(name).suffix.lower() in PROTECTED_SUFFIXES
    )


def _allowlisted_paths(policy: dict) -> set[str]:
    allowed_paths: set[str] = set()
    for entry in policy["examples"]:
        task_root = _task_dir(entry["task_id"]).relative_to(TASKS)
        for raw_relative in entry["allowed_private_like_files"]:
            relative = PurePosixPath(raw_relative)
            assert raw_relative == relative.as_posix(), raw_relative
            assert not relative.is_absolute(), raw_relative
            assert ".." not in relative.parts and "\\" not in raw_relative, raw_relative
            assert _is_private_like(relative), raw_relative
            full_relative = (PurePosixPath(task_root.as_posix()) / relative).as_posix()
            assert full_relative not in allowed_paths, full_relative
            allowed_paths.add(full_relative)
    return allowed_paths


def _tracked_paths() -> list[Path]:
    raw = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return [ROOT / value.decode() for value in raw.split(b"\0") if value]


def _iter_public_text_files():
    for path in _tracked_paths():
        if not path.is_file() or path.is_symlink():
            continue
        data = path.read_bytes()
        if b"\0" not in data[:8192]:
            yield path


def _task_directories() -> list[Path]:
    task_dirs = [TASKS / "_template"]
    for domain_dir in TASKS.iterdir():
        if not domain_dir.is_dir() or domain_dir.name.startswith("_"):
            continue
        task_dirs.extend(path for path in domain_dir.iterdir() if path.is_dir())
    return sorted(task_dirs)


def _task_id_from_directory(task_dir: Path) -> str:
    if task_dir.name == "_template":
        return "_template"
    return f"{task_dir.parent.name}.{task_dir.name}"


def _declared_public_files(task_dir: Path, entry: dict | None) -> set[str]:
    meta_path = task_dir / "task_meta.yaml"
    legacy_path = task_dir / "task.yaml"
    if meta_path.is_file():
        descriptor = meta_path
        allowed = {"task_meta.yaml"}
    else:
        assert entry is not None and "task.yaml" in entry["allowed_private_like_files"], (
            f"{_task_id_from_directory(task_dir)} has no public task_meta.yaml"
        )
        descriptor = legacy_path
        allowed = set()

    document = yaml.safe_load(descriptor.read_text(encoding="utf-8"))
    prompts = document.get("prompts") if isinstance(document, dict) else None
    assert isinstance(prompts, dict), descriptor
    assert set(prompts) == {"b1", "b2", "b3", "b4"}, descriptor
    for band, raw_path in prompts.items():
        assert raw_path == f"prompt_{band}.md", (descriptor, band, raw_path)
        relative = PurePosixPath(raw_path)
        assert relative.parent == PurePosixPath("."), (descriptor, raw_path)

    if entry is not None and entry["kind"] == "full":
        allowed.update(prompts.values())

    if entry is not None:
        allowed.update(entry["allowed_private_like_files"])
    return allowed


def test_example_allowlist_is_exact_and_references_existing_files():
    policy = _load_policy()
    assert policy["schema_version"] == 1
    assert set(policy["private_like_filenames"]) == {
        "task.yaml",
        "task_eval.yaml",
        "generate_gt.py",
        "precompute_gt.py",
        "custom_scorer.py",
        "reference_specs.md",
    }
    assert set(policy["private_like_directories"]) == {"reference"}
    entries = {entry["task_id"]: entry for entry in policy["examples"]}
    assert len(entries) == len(policy["examples"])
    assert set(entries) == EXPECTED_EXAMPLES
    assert sum(entry["kind"] == "reference_only" for entry in entries.values()) == 0
    assert sum(entry["kind"] == "full" for entry in entries.values()) == 5
    assert policy["source_revision"] == "2ba9258442bf53ad6c4911957234e03e767476ad"

    for task_id, entry in entries.items():
        assert entry["kind"] in {"reference_only", "full", "template"}
        task_dir = _task_dir(task_id)
        assert task_dir.is_dir(), task_id
        allowed = entry["allowed_private_like_files"]
        assert len(allowed) == len(set(allowed)), task_id
        for relative_path in allowed:
            target = task_dir / relative_path
            assert target.is_file() and not target.is_symlink(), (task_id, relative_path)

    for task_id, expected_files in EXPECTED_FULL_EXAMPLE_FILES.items():
        assert {
            path.name for path in _task_dir(task_id).iterdir() if path.is_file()
        } == expected_files


def test_seed42_restored_tasks_have_public_metadata_and_prompt_mapping():
    """Every formerly unmatched seed42 instance has a local public contract."""
    for task_id in SEED42_RESTORED_TASKS:
        task_dir = _task_dir(task_id)
        metadata = yaml.safe_load((task_dir / "task_meta.yaml").read_text(encoding="utf-8"))
        assert metadata["id"] == task_id
        assert metadata["status"] == "final"
        assert set(metadata["prompts"]) == {"b1", "b2", "b3", "b4"}
        assert metadata["output"]["files"], task_id


def test_nbody_metadata_matches_the_official_particle_forecast_contract():
    """The public catalog must not regress to the retired scattering contract."""
    metadata = yaml.safe_load(
        (_task_dir("astronomy.nbody_close_encounters") / "task_meta.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert metadata["version"] == "4.0"
    assert [item["name"] for item in metadata["input"]["files"]] == [
        "input_0.json",
        "input_1.npy",
        "input_2.npy",
        "input_3.npy",
        "input_4.npy",
        "input_5.npy",
    ]
    assert [item["name"] for item in metadata["output"]["files"]] == [
        "prediction_quantiles.npy",
        "risk_summary.csv",
    ]


def test_final_task_metadata_contains_only_the_public_contract_surface():
    for descriptor in sorted(TASKS.glob("*/*/task_meta.yaml")):
        metadata = yaml.safe_load(descriptor.read_text(encoding="utf-8"))
        if metadata["status"] != "final":
            continue
        assert {"evaluation", "generation", "execution", "timeout"}.isdisjoint(
            metadata
        ), descriptor
        for output in metadata["output"]["files"]:
            assert {"shape", "dtype"}.isdisjoint(output), (descriptor, output["name"])


def test_public_catalog_has_exact_final_and_sample_statuses():
    statuses: dict[str, str] = {}
    stale_abandoned_reasons: list[str] = []
    for task_dir in sorted(TASKS.glob("*/*")):
        if not task_dir.is_dir() or task_dir.parent.name.startswith("_"):
            continue
        descriptor = task_dir / "task_meta.yaml"
        if not descriptor.is_file():
            descriptor = task_dir / "task.yaml"
        metadata = yaml.safe_load(descriptor.read_text(encoding="utf-8"))
        statuses[metadata["id"]] = metadata["status"]
        if "abandoned_reason" in metadata:
            stale_abandoned_reasons.append(metadata["id"])

    sample_ids = {task_id for task_id, status in statuses.items() if status == "sample"}
    final_ids = {task_id for task_id, status in statuses.items() if status == "final"}

    assert sample_ids == EXPECTED_SAMPLE_TASKS
    assert len(final_ids) == 60
    assert set(statuses.values()) == {"final", "sample"}
    assert len(statuses) == 65
    assert stale_abandoned_reasons == []


def test_non_example_benchmark_tasks_track_only_metadata():
    example_dirs = {_task_dir(task_id) for task_id in EXPECTED_EXAMPLES}
    violations = []
    for path in _tracked_paths():
        if TASKS not in path.parents or path.parent in example_dirs:
            continue
        if path.name != "task_meta.yaml":
            violations.append(path.relative_to(ROOT).as_posix())
    assert violations == []


def test_private_like_files_only_exist_at_allowlisted_example_paths():
    policy = _load_policy()
    allowed_paths = _allowlisted_paths(policy)

    violations = []
    for path in TASKS.rglob("*"):
        relative = PurePosixPath(path.relative_to(TASKS).as_posix())
        if path.is_symlink():
            violations.append(f"symlink is not allowed: {relative}")
        elif path.is_file() and _is_private_like(relative):
            if relative.as_posix() not in allowed_paths:
                violations.append(relative.as_posix())

    assert not violations, (
        "Private-like Task files require an exact config/public_examples.json exception:\n"
        + "\n".join(violations)
    )


def test_task_files_are_fail_closed_to_declared_public_files_and_examples():
    policy = _load_policy()
    entries = {entry["task_id"]: entry for entry in policy["examples"]}
    allowed_paths: set[str] = set()
    known_task_ids: set[str] = set()
    for task_dir in _task_directories():
        task_id = _task_id_from_directory(task_dir)
        known_task_ids.add(task_id)
        task_root = task_dir.relative_to(TASKS)
        for relative in _declared_public_files(task_dir, entries.get(task_id)):
            allowed_paths.add((task_root / relative).as_posix())

    assert set(entries) <= known_task_ids
    violations = []
    for path in _tracked_paths():
        try:
            relative = path.relative_to(TASKS)
        except ValueError:
            continue
        if path.is_symlink():
            violations.append(f"symlink is not allowed: {relative.as_posix()}")
        elif path.is_file() and relative.as_posix() not in allowed_paths:
            violations.append(relative.as_posix())

    assert not violations, (
        "Tracked Task files must be public metadata or an "
        "exact Example exception:\n" + "\n".join(violations)
    )


def test_fail_closed_policy_rejects_unknown_sensitive_looking_files():
    example_ids = {entry["task_id"] for entry in _load_policy()["examples"]}
    ordinary_task = next(
        task_dir for task_dir in _task_directories()
        if _task_id_from_directory(task_dir) not in example_ids
    )
    allowed = _declared_public_files(ordinary_task, None)
    examples = [
        "data/" + "scorer.py",
        "data/" + "expected_output.npy",
        "thresholds.json",
        "data/" + "internal_comments.md",
        ".env.production",
    ]
    assert allowed == {"task_meta.yaml"}
    assert set(examples).isdisjoint(allowed)


def test_internal_repository_pattern_is_generic_and_does_not_block_package_name():
    pattern = BLOCKED_INTERNAL_PATTERNS["internal/private repository identifier"]
    nonpublic_example = (
        "research-org/" + "asi_bench-" + "inter" + "nal"
    )
    assert pattern.search(nonpublic_example)
    assert not pattern.search("agent-ai4sci-bench")


def test_private_path_detection_covers_nested_case_and_abnormal_names():
    protected = (
        "domain/task/nested/task_eval.yaml",
        "domain/task/assets/REFERENCE/answer.npy",
        "domain/task/data/ground_truth.npy",
        "domain/task/data/reference-answer.csv",
        "domain/task/data/signing.key",
        "_unexpected/private/payload.bin",
    )
    for value in protected:
        assert _is_private_like(PurePosixPath(value)), value
    assert not _is_private_like(PurePosixPath("domain/task/data/public_input.csv"))


def test_public_text_has_no_private_repo_identifiers_or_obvious_secrets():
    violations = []
    for path in _iter_public_text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        relative = path.relative_to(ROOT).as_posix()
        for label, pattern in BLOCKED_INTERNAL_PATTERNS.items():
            if pattern.search(text):
                violations.append(f"{relative}: {label}")
        for label, pattern in SENSITIVE_CONTENT_PATTERNS.items():
            if pattern.search(text):
                violations.append(f"{relative}: {label}")
    assert not violations, "Potential public-repo disclosure:\n" + "\n".join(violations)


def test_tracked_tree_has_no_secret_containers_or_symlinks():
    violations = []
    for path in _tracked_paths():
        relative = path.relative_to(ROOT).as_posix()
        name = path.name.lower()
        if path.is_symlink():
            violations.append(f"{relative}: symlink")
        elif name in BLOCKED_TRACKED_FILENAMES:
            violations.append(f"{relative}: blocked credential filename")
        elif path.suffix.lower() in BLOCKED_TRACKED_SUFFIXES:
            violations.append(f"{relative}: blocked secret/archive suffix")
    assert not violations, "Potential secret container:\n" + "\n".join(violations)
