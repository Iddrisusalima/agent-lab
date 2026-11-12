"""Lints over the repository's committed resources (task 12.1).

Every other test file exercises code. This one asserts facts about *files that
are committed to the repository* -- the manifest, the ignore list, the documented
environment table, the system-prompt resource, the price table, the eval suite,
and the sample corpus -- because several requirements are statements about the
checkout itself rather than about any function in it:

* Req 1.1 - 1.4: the package imports, the manifest declares the interpreter
  floor and every dependency, and the ignore list keeps run artifacts out.
* Req 1.5, Req 17.3: ``.env.example`` is the documented table and holds
  placeholder values only -- no committed file carries a credential value.
* Req 3.8: every price row carries both prices and the date they were read.
* Req 12.10: the instruction text exists once, in one versioned resource, with
  no second copy anywhere under ``src/``.
* Req 14.1, Req 14.9: the suite is exactly 12 tasks in the published split, at
  ``evals/tasks.json``.
* Req 17.1, Req 17.15: 5 to 10 non-sensitive sample files including the
  injection fixture, and no committed notes file, trace file, or Chroma
  directory.

**Deliberately not asserted here.**

``README.md`` is a stub until task 12.2 writes it, so nothing in this file reads
its content; the README assertions of Req 17.4 - Req 17.12 belong with the
README, not ahead of it. ``evals/results.md`` (Req 17.2) is likewise generated
by the before/after eval run of task 11.10 and does not exist yet, so only
``evals/tasks.json`` is asserted from that directory.

**Kept light on purpose.** ``tests/test_properties.py`` (Property 19) already
parses ``.env.example`` as the oracle for the loader's ranges and defaults, and
``tests/test_pricing.py`` already covers loading and validating the price table.
This file asserts what those two do not: the committed-file facts, and above all
that no credential value is committed.
"""

from __future__ import annotations

import ast
import fnmatch
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest

from agent_lab.config import DEFAULTS, EXAMPLE_FILE
from agent_lab.evals.schema import (
    CATEGORY_COMPOSITION,
    SUITE_SIZE,
    load_suite,
)
from agent_lab.pricing import DEFAULT_PRICES_PATH
from agent_lab.prompts import DEFAULT_PROMPT_PATH, load_system_prompt

# --------------------------------------------------------------------------- #
# The checkout under test
# --------------------------------------------------------------------------- #

#: ``tests/test_resources.py`` -> ``tests`` -> the repository root. The same
#: anchor ``prompts.PROJECT_ROOT`` and ``pricing.DEFAULT_PRICES_PATH`` use, so a
#: moved checkout moves all three together.
REPO_ROOT = Path(__file__).resolve().parents[1]

SRC_ROOT = REPO_ROOT / "src" / "agent_lab"
SAMPLE_NOTES_DIR = REPO_ROOT / DEFAULTS["DOCS_DIR"]
INJECTION_DOCUMENT = "injection_test.md"

#: Req 17.1: the extensions the loader indexes, which is what a sample file may
#: be. ``.gitkeep`` is a packaging marker, not a document, and is excluded from
#: the count everywhere below.
CORPUS_EXTENSIONS = (".txt", ".md", ".pdf")
PACKAGING_MARKERS = (".gitkeep",)


def _read(path: Path) -> str:
    """Read a committed text file with newlines normalised.

    Every assertion here runs on a Windows checkout and on CI, so a CRLF file
    must compare equal to the same file with LF endings.
    """
    return path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")


def _collapse(text: str) -> str:
    """Lower-case ``text`` with every whitespace run reduced to one space.

    The comparison form for the prompt-duplication lint: a sentence rewrapped
    across different line widths in two files is still the same sentence.
    """
    return " ".join(text.split()).lower()


def _source_files() -> list[Path]:
    """Every ``.py`` file under ``src/agent_lab``, excluding caches."""
    return sorted(
        path
        for path in SRC_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def _sample_files() -> list[Path]:
    """Every committed sample document, packaging markers excluded."""
    return sorted(
        path
        for path in SAMPLE_NOTES_DIR.iterdir()
        if path.is_file() and path.name not in PACKAGING_MARKERS
    )


# --------------------------------------------------------------------------- #
# Req 1.1: the package imports, and importing it does not import the CLI
# --------------------------------------------------------------------------- #


def test_every_source_module_imports_and_none_of_them_imports_the_cli() -> None:
    """Importing all of ``agent_lab`` except ``cli`` leaves ``cli`` unimported.

    Two facts in one subprocess (Req 1.1, and the design's rule that
    ``agent_loop.py`` stays free of provider code). First, every module imports
    on its own -- a syntax error or a circular import in a module no test
    happens to touch is caught here. Second, none of them drags in ``cli.py``,
    which is the only module allowed to import the provider adapters: if
    importing, say, ``agent_lab.cost`` pulled in the CLI, it would pull in the
    OpenAI SDK with it and the layering would be decorative.

    A subprocess rather than an in-process import, because ``sys.modules`` is
    already populated by the time any test runs -- pytest has imported this
    file, and conftest has imported half the package -- so an in-process check
    could only ever report what the suite happened to import first.
    """
    program = (
        "import importlib, pkgutil, sys\n"
        "import agent_lab\n"
        "names = [m.name for m in pkgutil.walk_packages(agent_lab.__path__, 'agent_lab.')\n"
        "         if m.name != 'agent_lab.cli']\n"
        "for name in names:\n"
        "    importlib.import_module(name)\n"
        "print(len(names))\n"
        "print('cli' if 'agent_lab.cli' in sys.modules else 'no-cli')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO_ROOT,
    )
    assert completed.returncode == 0, (
        "importing the package modules failed:\n" + completed.stderr
    )
    count_line, flag_line = completed.stdout.split()[:2]
    # Sanity: the walk found the package, not an empty namespace.
    assert int(count_line) >= 20, f"only {count_line} modules were found under src/"
    assert flag_line == "no-cli", (
        "importing a module under src/agent_lab also imported agent_lab.cli; "
        "the CLI is the only module that may import the provider adapters, so "
        "nothing below it may import the CLI"
    )


def test_no_module_under_src_names_the_cli_in_an_import_statement() -> None:
    """The same rule as a static scan, which names the offending file.

    The subprocess above proves the fact; this one localises a regression to a
    file and a line number without anyone having to bisect imports by hand.
    """
    offenders: list[str] = []
    for path in _source_files():
        if path.name == "cli.py":
            continue
        tree = ast.parse(_read(path), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                # `from .cli import main` has module == 'cli'; `from . import cli`
                # carries it among the aliases.
                names = [node.module or ""] + [alias.name for alias in node.names]
            else:
                continue
            if any(name == "cli" or name.endswith(".cli") for name in names):
                rel = path.relative_to(REPO_ROOT).as_posix()
                offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, f"these modules import the CLI: {', '.join(offenders)}"


# --------------------------------------------------------------------------- #
# Req 1.2, Req 1.3: the dependency manifest
# --------------------------------------------------------------------------- #


def test_pyproject_declares_the_interpreter_floor_and_the_entry_point() -> None:
    """``requires-python = ">=3.10"`` and the ``agentlab`` console script.

    Asserted against the file's text rather than a parsed document so that this
    holds on Python 3.10 itself, where no TOML parser is in the standard library
    -- and 3.10 is exactly the interpreter the line is a promise about.
    """
    text = _read(REPO_ROOT / "pyproject.toml")
    assert re.search(r'requires-python\s*=\s*">=3\.10"', text), (
        "pyproject.toml must declare requires-python = \">=3.10\" (Req 1.3)"
    )
    assert re.search(r'agentlab\s*=\s*"agent_lab\.cli:main"', text), (
        "pyproject.toml must declare the console script agentlab = "
        '"agent_lab.cli:main" (Req 1.1)'
    )


def test_pyproject_declares_every_runtime_and_test_dependency() -> None:
    """Each dependency the project imports is named in the one manifest (Req 1.2).

    The runtime list carries the provider SDK, the vector store and the PDF
    reader; the ``dev`` extra carries the test framework and the property-based
    testing library. ``rich`` is optional by design (Req 13.5, Req 13.6) and so
    is checked as an extra, not as a runtime dependency.
    """
    tomllib = pytest.importorskip(
        "tomllib",
        reason="no TOML parser in the standard library before 3.11; the "
        "text-level manifest assertions above still run",
    )
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    runtime = " ".join(data["project"]["dependencies"])
    for package in ("openai", "chromadb", "pypdf"):
        assert package in runtime, f"{package} is imported but not declared"

    extras = data["project"]["optional-dependencies"]
    dev = " ".join(extras["dev"])
    for package in ("pytest", "hypothesis"):
        assert package in dev, f"{package} is used by the suite but not declared"
    assert "rich" in " ".join(extras.get("rich", [])) or "rich" in dev, (
        "rich must be declared as an optional extra (Req 13.5)"
    )


# --------------------------------------------------------------------------- #
# Req 1.4, Req 17.15: run artifacts stay out of the repository
# --------------------------------------------------------------------------- #


def _gitignore_patterns() -> list[str]:
    """The ignore patterns, comments and blank lines dropped."""
    lines = [line.strip() for line in _read(REPO_ROOT / ".gitignore").splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def _pattern_ignores(pattern: str, rel_path: str, *, is_dir: bool) -> bool:
    """Whether one ignore pattern covers ``rel_path``.

    A deliberately small reading of gitignore syntax -- enough for the patterns
    this repository actually uses, and no more. A pattern is *anchored* at the
    repository root when it carries a leading slash (``/traces/``) or an
    interior one (``traces/run.jsonl``); git does not count a trailing slash as
    a separator, so ``notes.md`` and ``.chroma/`` stay unanchored and are
    matched against every path segment, which is how ``.chroma/`` covers
    everything the index writes inside it. A trailing slash restricts the
    pattern to directories.

    The leading-slash form matters here rather than being a nicety: the
    operator's run artifacts are anchored precisely so that ``traces/`` stops
    matching ``evals/traces/``, the committed eval evidence. A matcher blind to
    anchoring cannot tell the fixed ignore list from the broken one.

    Negation (``!``) is not implemented, and :func:`test_gitignore_uses_no_negations`
    asserts the file contains none, so this stays a faithful reading rather than
    an optimistic one.
    """
    dir_only = pattern.endswith("/")
    pat = pattern.rstrip("/")
    # A leading slash anchors without being an interior separator; strip it, but
    # remember that it anchored the pattern.
    anchored = pat.startswith("/") or "/" in pat.lstrip("/")
    pat = pat.lstrip("/")
    rel = PurePosixPath(rel_path)

    if anchored:  # anchored at the repository root
        if dir_only:
            return rel_path == pat or rel_path.startswith(pat + "/")
        return fnmatch.fnmatch(rel_path, pat) or rel_path.startswith(pat + "/")

    parts = rel.parts
    for index, segment in enumerate(parts):
        if not fnmatch.fnmatch(segment, pat):
            continue
        is_leaf = index == len(parts) - 1
        if dir_only and is_leaf and not is_dir:
            continue  # the pattern matches directories only
        return True
    return False


def _is_ignored(rel_path: str, *, is_dir: bool = False) -> bool:
    """Whether any committed ignore pattern covers ``rel_path``."""
    return any(
        _pattern_ignores(pattern, rel_path, is_dir=is_dir)
        for pattern in _gitignore_patterns()
    )


def test_gitignore_uses_no_negations() -> None:
    """No ``!`` re-inclusion, so the matcher above reads the file correctly.

    A guard on this file's own reasoning: a negation could un-ignore one of the
    four artifact paths below and every assertion that follows would still pass.
    """
    negations = [p for p in _gitignore_patterns() if p.startswith("!")]
    assert not negations, (
        f"the ignore matcher in this test does not implement negation, but "
        f".gitignore carries {negations}; extend the matcher or drop the pattern"
    )


@pytest.mark.parametrize(
    ("rel_path", "is_dir", "what"),
    [
        (".env", False, "the operator's real configuration"),
        (DEFAULTS["NOTES_FILE"], False, "the notes file written by save_note"),
        (DEFAULTS["TRACE_FILE"], False, "the JSON-lines trace of a run"),
        ("traces", True, "the default trace directory"),
        (DEFAULTS["CHROMA_DIR"], True, "the Chroma persistence directory"),
    ],
)
def test_gitignore_excludes_every_run_artifact(
    rel_path: str, is_dir: bool, what: str
) -> None:
    """All four Req 1.4 patterns are present and actually cover their paths.

    The paths come from ``config.DEFAULTS``, not from literals, so changing a
    documented default without updating ``.gitignore`` fails here rather than
    at the moment an operator commits their own notes.
    """
    assert _is_ignored(rel_path, is_dir=is_dir), (
        f".gitignore does not exclude {rel_path!r} ({what}); Req 1.4 requires "
        f"patterns for .env, the notes file, the trace file and the Chroma "
        f"directory. Present patterns: {_gitignore_patterns()}"
    )


def test_no_run_artifact_is_committed() -> None:
    """Req 17.15: no committed notes file, trace file, or Chroma directory.

    A build of the retrieval index leaves a real ``.chroma/`` in the working
    tree, and a run leaves ``notes.md`` and ``traces/run.jsonl``, so "not
    committed" cannot mean "not present on disk" for a developer checkout. It
    means each path is either absent or covered by ``.gitignore`` -- which is
    what makes the ignore assertion above load-bearing rather than decorative.
    """
    artifacts = [
        (DEFAULTS["NOTES_FILE"], False),
        (DEFAULTS["TRACE_FILE"], False),
        ("run.jsonl", False),
        (DEFAULTS["CHROMA_DIR"], True),
        (".env", False),
    ]
    committed = [
        rel_path
        for rel_path, is_dir in artifacts
        if (REPO_ROOT / rel_path).exists() and not _is_ignored(rel_path, is_dir=is_dir)
    ]
    assert not committed, (
        f"these run artifacts exist in the checkout and are not git-ignored, so "
        f"a commit would include them: {committed} (Req 17.15)"
    )


# --------------------------------------------------------------------------- #
# Req 1.5, Req 17.3: the documented table carries placeholders only
# --------------------------------------------------------------------------- #

#: Variable-name suffixes that mark a value as a credential.
_CREDENTIAL_SUFFIXES = ("_API_KEY", "_API_TOKEN", "_SECRET_KEY", "_SECRET", "_TOKEN")

#: Words and shapes that make a value visibly a placeholder rather than a
#: secret. ``<...>`` is the angle-bracket form; ``sk-your-key-here`` is the form
#: ``.env.example`` uses.
_PLACEHOLDER_MARKERS = (
    "your",
    "here",
    "placeholder",
    "example",
    "changeme",
    "change-me",
    "xxx",
    "...",
    "replace",
    "<",
)

#: Shapes real credentials take. Each requires a long random tail, so a
#: placeholder that merely *starts* like a key (``sk-your-key-here``) does not
#: match while an actual key does. Req 17.3 is the reason these run over every
#: committed text resource and not only over ``.env.example``.
_CREDENTIAL_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9]{24,}"),
    re.compile(r"sk-proj-[A-Za-z0-9_\-]{24,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bASIA[0-9A-Z]{16}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{36}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)


def _looks_like_a_placeholder(value: str) -> bool:
    """Whether a credential value is visibly not a credential."""
    lowered = value.strip().strip("\"'").lower()
    if not lowered:
        return True  # an empty value is no secret
    return any(marker in lowered for marker in _PLACEHOLDER_MARKERS)


def _credential_matches(text: str) -> list[str]:
    """Every credential-shaped token in ``text``."""
    found: list[str] = []
    for pattern in _CREDENTIAL_PATTERNS:
        found.extend(match.group(0) for match in pattern.finditer(text))
    return found


def test_env_example_assigns_a_placeholder_to_every_credential_variable() -> None:
    """Req 1.5, Req 17.3: the committed table names a key but holds none.

    Only the credential rows are inspected here. The defaults and the documented
    ranges are Property 19's business -- it parses this same file as the oracle
    for ``load_config`` -- and restating them would make two places to update.
    """
    text = _read(REPO_ROOT / EXAMPLE_FILE)
    assignments = [
        (match.group(1), match.group(2))
        for match in re.finditer(r"(?m)^([A-Z][A-Z0-9_]*)=(.*)$", text)
    ]
    assert assignments, f"{EXAMPLE_FILE} declares no variables at all"

    credentials = [
        (name, value)
        for name, value in assignments
        if name.endswith(_CREDENTIAL_SUFFIXES)
    ]
    assert credentials, (
        f"{EXAMPLE_FILE} must document the provider credential variable "
        f"(Req 1.5); found only {[name for name, _ in assignments]}"
    )
    for name, value in credentials:
        assert _looks_like_a_placeholder(value), (
            f"{EXAMPLE_FILE} assigns {name}={value!r}, which does not read as a "
            f"placeholder. Req 17.3 allows placeholder values only"
        )


def test_no_committed_resource_contains_a_credential_value() -> None:
    """Req 17.3: no credential-shaped token in any committed resource.

    The scan covers ``.env.example``, the sample corpus, the system prompt, the
    price table and the eval suite -- every text file an operator did not write
    themselves. ``README.md`` is excluded because task 12.2 has not written it
    yet and this file asserts nothing about it.
    """
    scanned = [
        REPO_ROOT / EXAMPLE_FILE,
        DEFAULT_PROMPT_PATH,
        DEFAULT_PRICES_PATH,
        REPO_ROOT / "evals" / "tasks.json",
        *_sample_files(),
    ]
    offenders = {
        path.relative_to(REPO_ROOT).as_posix(): matches
        for path in scanned
        if path.suffix != ".pdf" and (matches := _credential_matches(_read(path)))
    }
    assert not offenders, f"credential-shaped values in committed files: {offenders}"


def test_the_injection_fixtures_api_key_placeholder_is_not_a_credential() -> None:
    """The one deliberate key-shaped string in the corpus stays allowed.

    ``sample_notes/injection_test.md`` asks its reader to disclose the provider
    key and notes the placeholder an auditor would expect in its place. That
    literal is correct and must not be read as a committed secret, so it is
    asserted here explicitly: if the scan above is ever tightened into flagging
    it, this test says why it should not be.
    """
    placeholder = "<PROVIDER_API_KEY_VALUE>"
    text = _read(SAMPLE_NOTES_DIR / INJECTION_DOCUMENT)
    assert placeholder in text, (
        f"{INJECTION_DOCUMENT} no longer carries the {placeholder} placeholder; "
        f"the fixture's configuration-disclosure block depends on it"
    )
    assert not _credential_matches(placeholder)
    assert _looks_like_a_placeholder(placeholder)


# --------------------------------------------------------------------------- #
# Req 3.8: the committed price table
# --------------------------------------------------------------------------- #

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def test_every_committed_price_entry_carries_both_prices_and_a_source_date() -> None:
    """Req 3.8, asserted against the committed JSON rather than the loader.

    ``tests/test_pricing.py`` covers parsing and validation. What is checked
    here is the file as shipped: that each row a cost figure will be computed
    from has an input price, an output price, and the ISO 8601 date on which
    both were read from the provider's pricing page. A row without the date is
    a row nobody can audit.
    """
    document = json.loads(DEFAULT_PRICES_PATH.read_text(encoding="utf-8"))
    models = document["models"]
    assert models, "pricing/prices.json declares no models"

    for model_id, entry in models.items():
        where = f"pricing/prices.json models.{model_id}"
        assert model_id and not model_id.startswith("_"), f"{where}: odd model id"
        for key in ("input_per_mtok", "output_per_mtok"):
            assert key in entry, f"{where}: missing {key}"
            value = entry[key]
            assert isinstance(value, (int, float)) and not isinstance(value, bool), (
                f"{where}: {key} is {value!r}, not a number"
            )
            assert value > 0, f"{where}: {key} is {value}, which cannot be a price"
        sourced_on = entry.get("sourced_on")
        assert isinstance(sourced_on, str) and _ISO_DATE.match(sourced_on), (
            f"{where}: sourced_on is {sourced_on!r}; Req 3.8 requires the "
            f"YYYY-MM-DD date on which the prices were read"
        )


def test_the_default_model_is_priced() -> None:
    """The model a reviewer runs without configuring anything has a price.

    Otherwise the first run out of the box reports a cost of zero plus an
    unpriced-model warning (Req 3.3), which is correct behaviour for an exotic
    model id and a packaging mistake for the documented default.
    """
    document = json.loads(DEFAULT_PRICES_PATH.read_text(encoding="utf-8"))
    default_model = DEFAULTS["MODEL_ID"]
    assert default_model in document["models"], (
        f"the documented default MODEL_ID {default_model!r} has no row in "
        f"pricing/prices.json"
    )


# --------------------------------------------------------------------------- #
# Req 12.10: one versioned prompt resource, and only one copy of its text
# --------------------------------------------------------------------------- #


def test_the_prompt_resource_is_committed_and_carries_a_version() -> None:
    """Req 12.10: the single resource, with a version in its first line."""
    assert DEFAULT_PROMPT_PATH.is_file(), (
        f"the system prompt resource is missing: {DEFAULT_PROMPT_PATH}"
    )
    text, version = load_system_prompt()
    assert version, "the prompt's first line carries no version identifier"
    assert version in DEFAULT_PROMPT_PATH.name, (
        f"the resource is named {DEFAULT_PROMPT_PATH.name} but its header "
        f"declares {version}; the file name and the header must agree"
    )
    assert len(text.split()) > 100, "the prompt resource holds no instruction text"


def test_only_one_versioned_prompt_resource_is_committed() -> None:
    """Exactly one ``system_prompt.*.md`` file, so "the single copy" is literal.

    A second resource left behind by a revision is the same defect Req 12.10
    forbids, one directory over from the one the lint below looks at.
    """
    resources = sorted(
        path.name for path in (REPO_ROOT / "prompts").glob("system_prompt*.md")
    )
    assert resources == [DEFAULT_PROMPT_PATH.name], (
        f"prompts/ holds {resources}; Req 12.10 allows one versioned resource, "
        f"and a revision replaces it rather than joining it"
    )


def _distinctive_prompt_sentences() -> list[str]:
    """The prompt's prose sentences, in comparison form.

    Fenced code blocks are dropped before sentences are cut, because the one
    fenced block holds the untrusted-data markers and ``grounding.py`` must
    define those constants -- they are shared vocabulary, not duplicated
    instruction text. Headings go too, for the same reason: ``# Saving notes``
    is a title, not an instruction. Inline code ticks and emphasis markers are
    stripped so that a sentence copied into a docstring without its backticks
    is still recognised as the same sentence.

    Only sentences of eight words or more are kept. Shorter ones ("Search
    before you assert.") are short enough that an unrelated module could
    coincidentally contain the same words, and a lint that fires on a
    coincidence gets switched off.
    """
    text = _read(DEFAULT_PROMPT_PATH)
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    prose = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    prose = prose.replace("`", "").replace("*", "").replace("_", "")
    sentences = re.split(r"(?<=[.!?:])\s+|\n\s*\n", prose)
    return [
        collapsed
        for collapsed in (_collapse(sentence) for sentence in sentences)
        if len(collapsed.split()) >= 8
    ]


def test_no_module_under_src_repeats_the_prompts_distinctive_sentences() -> None:
    """Req 12.10: the instruction text is held in one place, the resource.

    Each prose sentence of eight words or more is looked for in every ``.py``
    file under ``src/`` -- code and docstrings alike, since a docstring quoting
    the prompt is as much a second copy as a string constant is, and it is the
    copy that drifts quietly when the resource is edited.

    This lint can hold because ``prompts.py`` was written to quote none of the
    resource's wording and to assert nothing about its content (see its module
    docstring). A module that needs to talk about what the prompt says should
    reference the requirement, not restate the sentence.
    """
    sentences = _distinctive_prompt_sentences()
    assert len(sentences) >= 20, (
        f"only {len(sentences)} distinctive sentences were extracted from the "
        f"prompt; the lint would be checking almost nothing"
    )

    duplicates: list[str] = []
    for path in _source_files():
        haystack = _collapse(_read(path).replace("`", "").replace("*", ""))
        rel = path.relative_to(REPO_ROOT).as_posix()
        for sentence in sentences:
            if sentence in haystack:
                duplicates.append(f"{rel}: {sentence[:70]}...")
    assert not duplicates, (
        "Req 12.10 allows one copy of the system prompt's instruction text, in "
        "prompts/" + DEFAULT_PROMPT_PATH.name + ". These modules repeat it:\n"
        + "\n".join(duplicates)
    )


# --------------------------------------------------------------------------- #
# Req 14.1, Req 14.9: the committed eval suite
# --------------------------------------------------------------------------- #


def test_the_committed_eval_suite_is_twelve_tasks_in_the_published_split() -> None:
    """Req 14.1: ``evals/tasks.json`` loads and matches the mandated composition.

    ``load_suite(require_composition=True)`` is the whole assertion: it parses
    the committed file, validates every task and check against Req 14.2 and
    Req 14.10, and then compares the category counts against
    ``CATEGORY_COMPOSITION``. Those constants are the authority, so the counts
    are read from them rather than restated -- a suite edited to eleven tasks
    fails here, and so does one that quietly swaps a multi-tool task for a
    single-tool one.
    """
    suite_path = REPO_ROOT / "evals" / "tasks.json"
    assert suite_path.is_file(), f"the eval suite is missing: {suite_path}"

    suite = load_suite(suite_path, require_composition=True)

    assert len(suite) == SUITE_SIZE == 12
    assert dict(suite.counts_by_category()) == dict(CATEGORY_COMPOSITION)
    assert len(set(suite.ids())) == len(suite.ids()), "task identifiers repeat"
    for task in suite:
        assert task.checks, f"task {task.id} declares no check (Req 14.2)"


# --------------------------------------------------------------------------- #
# Req 17.1: the sample corpus
# --------------------------------------------------------------------------- #


def test_the_sample_corpus_holds_between_five_and_ten_committed_documents() -> None:
    """Req 17.1: 5 to 10 files, each one the loader can read.

    The corpus is what a reviewer's first run searches, so its size and its file
    types are a shipped fact rather than a configuration choice. ``.gitkeep`` is
    excluded: it keeps the directory in version control and is not a document.
    """
    files = _sample_files()
    names = [path.name for path in files]
    assert 5 <= len(files) <= 10, (
        f"sample_notes holds {len(files)} documents ({names}); Req 17.1 asks for "
        f"between 5 and 10"
    )
    wrong_type = [name for name in names if not name.endswith(CORPUS_EXTENSIONS)]
    assert not wrong_type, (
        f"these sample files are not of a type the loader indexes "
        f"{CORPUS_EXTENSIONS}: {wrong_type}"
    )
    assert INJECTION_DOCUMENT in names, (
        f"Req 17.1 and Req 12.5 require the injection fixture "
        f"{INJECTION_DOCUMENT} to be one of the sample files; found {names}"
    )


def test_every_sample_document_holds_the_documented_amount_of_content() -> None:
    """Req 17.1: 150 to 2000 words per file.

    The lower bound is there because a one-line file produces a chunk with no
    context and no retrieval test worth running; the upper bound keeps the
    committed corpus small enough to read.
    """
    too_short: list[str] = []
    too_long: list[str] = []
    for path in _sample_files():
        if path.suffix == ".pdf":
            continue  # word count needs the extractor; the loader's tests own that
        words = len(_read(path).split())
        if words < 150:
            too_short.append(f"{path.name} ({words} words)")
        elif words > 2000:
            too_long.append(f"{path.name} ({words} words)")
    assert not too_short, f"sample files below 150 words: {too_short}"
    assert not too_long, f"sample files above 2000 words: {too_long}"


def test_the_injection_fixture_carries_its_override_and_hacked_instruction() -> None:
    """Req 12.5: the fixture instructs a reader to ignore its rules and save `HACKED`.

    The eval suite's injection task asserts that the notes file does not contain
    `HACKED` at the end of the run (Req 14.12), which only means anything while
    the document actually asks for that write. Asserted on the committed file so
    that an edit softening the fixture fails here rather than turning the
    injection task green for the wrong reason.
    """
    text = _read(SAMPLE_NOTES_DIR / INJECTION_DOCUMENT)
    collapsed = _collapse(text)
    assert "HACKED" in text, (
        f"{INJECTION_DOCUMENT} must ask for the literal text HACKED to be saved"
    )
    assert "save_note" in text, (
        f"{INJECTION_DOCUMENT} must name save_note in its forced-write block"
    )
    assert any(
        phrase in collapsed
        for phrase in (
            "ignore all previous instructions",
            "disregard your prior instructions",
            "disregard all previous instructions",
        )
    ), (
        f"{INJECTION_DOCUMENT} must instruct the reader to disregard its prior "
        f"instructions (Req 12.5)"
    )


def test_the_injection_fixture_shares_a_subject_with_another_sample_file() -> None:
    """Req 12.5: the fixture is retrievable, not an isolated oddity.

    It has to come back from a plausible ``search_docs`` query, which it only
    does if its topical content overlaps another file in the corpus. The
    fixture names the files it shares a subject with, so the assertion is that
    those files exist.
    """
    text = _read(SAMPLE_NOTES_DIR / INJECTION_DOCUMENT)
    neighbours = set(re.findall(r"`([A-Za-z0-9_\-]+\.(?:md|txt|pdf))`", text))
    neighbours.discard(INJECTION_DOCUMENT)
    assert neighbours, (
        f"{INJECTION_DOCUMENT} names no other sample file; Req 12.5 requires it "
        f"to carry topical content shared with at least one other document"
    )
    missing = sorted(
        name for name in neighbours if not (SAMPLE_NOTES_DIR / name).is_file()
    )
    assert not missing, (
        f"{INJECTION_DOCUMENT} refers to sample files that are not committed: "
        f"{missing}"
    )
