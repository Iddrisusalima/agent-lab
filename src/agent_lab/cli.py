"""The command-line entry point: parse, wire, run, report, exit (Req 1.1, 13.4).

This is the one module that imports adapters. ``agent_loop.py`` receives a
``ModelClient``, a dispatcher, a registry, a cost tracker, a trace sink and a
renderer, and never learns which implementation it was given; the knowledge of
*which* provider adapter, *which* retriever and *which* gate a run uses is
collected here and nowhere else. That is what makes Req 1.1 ("every module
importable without importing the CLI entry point") hold by construction rather
than by discipline, and it is why the whole of Requirements 4 through 13 can be
tested offline against fakes.

**What this module owns, and what it does not.** The loop narrates its own steps
and prints its own abnormal-termination message (Req 4.7, 4.8, 4.12, 4.13); the
CLI prints the *final answer* (Req 13.4) and the *run summary* (Req 3.9),
because those two are the CLI's under the requirements and because the CLI is
what holds the cost tracker. ``RunResult.error_message`` carries the same text
the loop already printed, so :func:`run_command` reports it through the exit
code and leaves the line alone rather than doubling it.

**The summary is printed on every termination path.** There is exactly one call
to ``renderer.run_summary`` on the normal path, placed after
:meth:`AgentLoop.run` returns, and :meth:`AgentLoop.run` returns on all four
outcomes rather than raising on three of them. So the summary is unconditional
by structure, not by four parallel branches that could drift (Property 14,
Req 3.9). The invalid-task path, which starts no run at all, prints a zeroed
summary from the same helper.

**Startup order is load-bearing.** Configuration, the system prompt and the
price table are all read before the task string is validated, and the task
string is validated before the trace logger, the model client, the gate and the
tool registry are built. Two requirements turn on that order: a configuration
fault must stop the process before any provider request *and* before any trace
line (Req 1.8, 1.9, 2.9), and an invalid task must start no run and issue no
model call (Req 4.12) -- so neither should leave a freshly created trace file
behind.

**Exit codes.** Every outcome's code comes from ``agent_loop.exit_code_for`` via
``RunResult.exit_code``, so the CLI does not re-derive the mapping (Req 4.10,
4.12). A configuration fault is caught in :func:`main` as the single
:class:`~agent_lab.config.ConfigError` type -- ``PromptError``,
``PriceTableError``, ``DisplayLimitError`` and ``ChunkingError`` are all
subclasses, so one ``except`` covers every startup refusal -- and exits 6
(Req 1.8, 1.9, 2.9). An argparse usage error exits 6 as well, through
:class:`_Parser`, because argparse's own default of 2 is this project's
step-limit code and a usage error must not be mistaken for one.

**Adding a subcommand.** A subcommand is one entry in :data:`SUBCOMMANDS`: a
name, a help line, a function that adds the command's own arguments, and a
handler taking ``(args, context)``. Nothing in :func:`build_parser` or
:func:`main` needs to change to add one. ``index`` and ``eval`` were both added
that way.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import IO, Any, Callable, Mapping, Sequence

from agent_lab import __version__, exit_codes
from agent_lab.agent_loop import AgentLoop, validate_task
from agent_lab.config import Config, ConfigError, load_config
from agent_lab.confirmation import build_gate
from agent_lab.cost import CostTracker
from agent_lab.evals.runner import run_eval
from agent_lab.pricing import price_table
from agent_lab.prompts import load_system_prompt
from agent_lab.providers import build_model_client
from agent_lab.redaction import redact
from agent_lab.render import AnswerStreamer, Renderer, build_renderer
from agent_lab.retrieval.base import Retriever
from agent_lab.retrieval.chroma_index import (
    ChromaUnavailable,
    build_index,
    select_retriever,
)
from agent_lab.tools.dispatcher import ToolDispatcher
from agent_lab.tools.factory import build_tool_registry
from agent_lab.trace import JsonlTraceLogger

PROGRAM = "agentlab"

DESCRIPTION = (
    "A command-line research agent that answers a task by calling tools in a "
    "bounded think-act-observe loop over your own notes and documents."
)

#: The status ``agentlab index`` exits with when the corpus was read but the
#: index could not be written -- ``chromadb`` absent, the store unopenable, or
#: the bundled embedding model not yet downloaded on a machine with no network
#: (:class:`~agent_lab.retrieval.chroma_index.ChromaUnavailable`).
#:
#: Not one of :mod:`agent_lab.exit_codes`'s values, on purpose. Those codes name
#: *run outcomes* (Req 4.10), and an index build performs no run: reporting this
#: as 2 would claim a step limit was reached and as 6 would claim the operator's
#: configuration is wrong, which it is not -- ``ChromaUnavailable`` says the
#: vector path is unavailable on this machine, and the product still answers
#: through keyword fallback search (Req 7.6). 1 is the shell's generic failure,
#: and it is the only code here that cannot be mistaken for an outcome.
INDEX_FAILURE = 1

#: Req 18.9: what the operator is told when streaming output was asked for and
#: the configured provider sent the reply complete rather than token by token.
#: Printed to stderr, as every other warning is (Req 13.8), so stdout still
#: carries exactly the heading, the answer and the summary it carries with
#: streaming off.
STREAMING_UNAVAILABLE_MESSAGE = (
    "warning: streaming output was unavailable for the configured provider, so "
    "the final answer was printed in a single write. Set STREAM_OUTPUT=off to "
    "stop asking for it."
)

#: Printed when tokens were streamed under the final-answer heading and the
#: answer the loop returned was nevertheless a different string. The loop is the
#: source of truth for the answer (Req 4.3), so the answer is printed again in
#: full and this line says why the operator is seeing it twice. Reached only
#: through the ordering :class:`~agent_lab.render.AnswerStreamer` documents and
#: cannot undo.
STREAM_MISMATCH_MESSAGE = (
    "warning: the streamed text differed from the answer the run returned; the "
    "answer below is the one the run returned."
)


# --------------------------------------------------------------------------- #
# The injection points a handler needs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CliContext:
    """The process state a subcommand handler reads, passed in rather than taken.

    ``load_config`` takes the environment as a parameter and the renderer, the
    gate and the trace logger all accept explicit streams, so a test can drive
    the whole command with a dictionary and three ``StringIO`` objects and
    assert on what each stream received. Defaulting every field to ``None``
    here, and resolving to the real process objects only where they are used,
    keeps ``main`` free of that plumbing.
    """

    env: Mapping[str, str] = field(default_factory=lambda: os.environ)
    stdin: IO[str] | None = None
    stdout: IO[str] | None = None
    stderr: IO[str] | None = None

    @property
    def err(self) -> IO[str]:
        """The stream a startup refusal is written to (Req 13.8)."""
        return self.stderr if self.stderr is not None else sys.stderr


# --------------------------------------------------------------------------- #
# Retriever selection
# --------------------------------------------------------------------------- #


def build_retriever(config: Config) -> Retriever:
    """The retriever ``search_docs`` searches through (Req 7.6, 7.7).

    One delegation, and it is the whole of the CLI's knowledge of retrieval:
    :func:`~agent_lab.retrieval.chroma_index.select_retriever` returns the
    Chroma-backed retriever when ``config.chroma_dir`` holds a non-empty
    collection -- which is what ``agentlab index`` builds -- and the keyword
    retriever otherwise, which needs no index, no embedding model and no extra
    dependency, and states in every result that fallback search was used
    (Req 7.6).

    Selection lives there rather than here because it has to answer the same
    question for the eval runner and for any other caller, and because it is the
    one function that may touch the store to answer it. What stays here is the
    fact that a run gets its retriever through the
    :class:`~agent_lab.retrieval.base.Retriever` protocol, so neither
    ``build_tool_registry``, nor ``search_docs``, nor the loop learns which
    implementation answered (Req 7.7).
    """
    return select_retriever(config)


# --------------------------------------------------------------------------- #
# The `run` subcommand
# --------------------------------------------------------------------------- #


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Declare the arguments of ``agentlab run`` (Req 4.1, 9.11)."""
    parser.add_argument(
        "task",
        metavar="TASK",
        help=(
            "the task to perform, as one quoted string of 1 to 4000 characters "
            "after trimming. Put '--' before a task that begins with a dash."
        ),
    )
    # Req 9.11: this option is the *only* thing that selects non-interactive
    # mode. No environment variable is read for it and no condition is inferred
    # from the process -- in particular, a non-tty stdin does not select it. The
    # `--yes` spelling is the familiar one for "do not ask me"; both set the
    # same flag, so `Config.interactive` has one source.
    parser.add_argument(
        "--non-interactive",
        "--yes",
        "-y",
        dest="non_interactive",
        action="store_true",
        help=(
            "approve note writes without prompting. Without this option every "
            "save_note call waits for your confirmation."
        ),
    )


def run_command(args: argparse.Namespace, context: CliContext) -> int:
    """Wire one run, perform it, report it, and return its exit code.

    The body is the startup order of the module docstring. Three groups, in
    this sequence:

    1. **Startup refusals.** Configuration, the system prompt and the price
       table. Each raises :class:`~agent_lab.config.ConfigError` on a fault and
       :func:`main` turns that into exit 6, before a provider request is issued
       and before a trace line exists (Req 1.6, 1.8, 1.9, 2.9). The access
       probe inside ``load_config`` warns and continues (Req 1.10).
    2. **The task check.** Applied here, through the loop's own
       :func:`~agent_lab.agent_loop.validate_task`, so that a task which starts
       no run also builds no model client and creates no trace file (Req 4.12).
    3. **The run.** Everything else is constructed, the loop is called, and the
       answer and the summary are printed.

    **Streaming output (Req 18.2, Req 18.9) is optional and sits in one place.**
    When ``config.streaming_enabled`` is false -- which it is unless the
    operator set ``STREAM_OUTPUT=on`` -- this function builds the adapter with
    no extra arguments and prints the answer in the one
    ``renderer.final_answer`` write it always printed (Req 18.6, Req 18.7).
    When it is true, an :class:`~agent_lab.render.AnswerStreamer` is given to
    the adapter as a token sink, and the single write is skipped only if what
    was streamed equals the answer the loop returned. The loop is the source of
    truth either way: the answer under the heading is always
    ``result.final_answer``, whether it got there one token at a time or all at
    once.
    """
    # --- 1. startup refusals ------------------------------------------------
    config = load_config(
        context.env,
        interactive=not args.non_interactive,
        stream=context.stderr,
    )
    system_prompt, _prompt_version = load_system_prompt()
    prices = price_table()

    renderer = build_renderer(config, stdout=context.stdout, stderr=context.stderr)
    tracker = CostTracker(prices=prices)

    # --- 2. the task check (Req 4.1, 4.12) ----------------------------------
    task, task_error = validate_task(args.task)
    if task_error is not None:
        renderer.error(task_error)
        _summarise(renderer, tracker, config, ())
        return exit_codes.INVALID_TASK

    # --- 3. the run ---------------------------------------------------------
    # The gate before the registry: `save_note` is given the gate at
    # construction, so no model-supplied value can reach it (Req 9.11, 12.6).
    gate = build_gate(
        config,
        input_stream=context.stdin,
        output_stream=context.stdout,
        secret=config.api_key,
    )
    registry = build_tool_registry(config, gate=gate, retriever=build_retriever(config))
    dispatcher = ToolDispatcher(registry, config)

    # Req 18.2, Req 18.6: the whole of the streaming feature's wiring, and it
    # is empty unless the operator set STREAM_OUTPUT=on. With it off,
    # `stream_overrides` is `{}`, the adapter is built with the arguments it has
    # always been built with, and `streamer` stays None so every branch below
    # that mentions it is skipped.
    streamer: AnswerStreamer | None = None
    stream_overrides: dict[str, Any] = {}
    if config.streaming_enabled:
        streamer = AnswerStreamer(renderer)
        stream_overrides = {
            "stream": True,
            "token_sink": streamer.token,
            "on_retract": streamer.retract,
        }

    # The adapter, and the only line in the project that chooses one (Req 2.8,
    # 2.9). One client per run: it holds the run's issued tool-call ids.
    model_client = build_model_client(config, **stream_overrides)

    trace_sink = JsonlTraceLogger(
        config.trace_file,
        api_key=config.api_key,
        stream=context.stderr,
    )

    loop = AgentLoop(
        model_client,
        dispatcher,
        registry,
        tracker,
        trace_sink=trace_sink,
        renderer=renderer,
        system_prompt=system_prompt,
        # Req 4.11: the configured values, which default to the loop's own
        # MAX_STEPS and COST_CAP constants. No per-run flag overrides either.
        max_steps=config.max_steps,
        cost_cap=config.cost_cap,
        notes_file=config.notes_file,
        # Req 18.4, Req 18.6: None unless MEMORY_FILE is set, which is the
        # feature's disabled state.
        memory_file=config.memory_path,
        memory_max_chars=config.tool_result_max_chars,
    )

    result = loop.run(task)

    # Req 18.2: close any open stream and read back what was written. `None`
    # here means streaming was never enabled, which is the only case that has to
    # behave exactly as it did before (Req 18.6, Req 18.7).
    streamed = streamer.finish() if streamer is not None else None

    # Req 18.9: said before the answer, so the operator knows why the answer
    # arrived all at once rather than after it has already read it.
    if streamer is not None and getattr(model_client, "streaming_unavailable", False):
        renderer.warning(STREAMING_UNAVAILABLE_MESSAGE)

    # Req 13.4: the final answer, to stdout, under its labelled heading. Present
    # only on the final-answer outcome -- a step-limited or cost-capped run
    # produced no answer and the loop has already said so (Req 4.7).
    if result.final_answer is not None:
        if streamed is not None and streamed == result.final_answer:
            # Req 18.2: the answer is already on stdout, written token by token
            # under this same heading and followed by the same newline. Printing
            # it again would double it, and the concatenation of what was
            # written equals what the loop returned, which is the guarantee.
            pass
        else:
            if streamer is not None and streamer.wrote_answer:
                renderer.warning(STREAM_MISMATCH_MESSAGE)
            renderer.final_answer(result.final_answer)

    # Req 3.9: one unconditional summary, reached on all four outcomes because
    # `run` returns on all four. `result.error_message` is deliberately not
    # reprinted: the loop printed it through this same renderer.
    _summarise(renderer, tracker, config, result.warnings)

    # Req 4.10, 4.12: the documented code for this outcome, from the loop's own
    # mapping.
    return result.exit_code


def _summarise(
    renderer: Renderer,
    tracker: CostTracker,
    config: Config,
    run_warnings: Sequence[str],
) -> None:
    """Print the run summary and every accumulated warning once (Req 3.9).

    ``run_warnings`` is ``RunResult.warnings``, which already begins with the
    cost tracker's own lines -- the unpriced-model warning of Req 3.3 and the
    unavailable-token-count warning of Req 3.4. The configuration access
    warnings of Req 1.10 are appended so the operator sees them again at the
    end of a run that printed many lines since startup; they were printed once
    already by the probe, so a line appearing in both is emitted once here.
    """
    seen: set[str] = set()
    lines: list[str] = []
    for line in (*run_warnings, *config.access_warnings):
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
    renderer.run_summary(tracker, warnings=lines)


# --------------------------------------------------------------------------- #
# The `index` subcommand (Req 7.3, 7.4, 7.10)
# --------------------------------------------------------------------------- #


def add_index_arguments(parser: argparse.ArgumentParser) -> None:
    """Declare the arguments of ``agentlab index``: there are none (Req 7.4).

    Req 7.4 speaks of "the single CLI command that builds the Retrieval_Index",
    and every input that command needs -- the corpus directory, the persistence
    directory, the target chunk length and the overlap -- is already an
    environment variable with a documented default (Req 1.6, 1.7). Adding flags
    for them would create a second, undocumented source for the same four
    values and let an index be built with settings that no later run reproduces.
    So the function exists to keep the shape of a :class:`Subcommand` entry
    uniform, and deliberately adds nothing.
    """


def index_command(args: argparse.Namespace, context: CliContext) -> int:
    """Rebuild the document index over the configured corpus (Req 7.3, 7.4).

    The three numbers Req 7.4 asks for -- files ingested, files skipped, chunks
    indexed -- are the fields of the
    :class:`~agent_lab.retrieval.loader.BuildReport` that
    :func:`~agent_lab.retrieval.chroma_index.build_index` returns, so this
    handler prints a report rather than counting anything itself. The loader
    prints one line per skipped file naming the file and the reason as it reads
    (Req 7.9); ``stream`` is stdout here, which is why those lines appear above
    the summary and in corpus order.

    **Three failure shapes, three answers.**

    * An out-of-range chunk length or overlap raises
      :class:`~agent_lab.retrieval.chunking.ChunkingError`, which is a
      :class:`~agent_lab.config.ConfigError`, so it is already caught in
      :func:`main` and reported as exit 6 with the offending value and the
      permitted range. ``build_index`` validates before it opens the store, and
      ``load_config`` validates before that, so a rejected build leaves an
      existing collection exactly as it was (Req 7.10).
    * :class:`~agent_lab.retrieval.chroma_index.ChromaUnavailable` -- no
      ``chromadb``, an unopenable store, or the bundled embedding model not yet
      downloaded on a machine with no network -- is caught here and reported as
      one legible stderr line with :data:`INDEX_FAILURE`, not as a traceback.
      It is not a configuration fault and not fatal to the product: without an
      index, ``search_docs`` answers through keyword fallback search and says so
      (Req 7.6).
    * Anything else propagates, because an unexpected exception during a build
      should not be reported as a tidy failure of a known kind.

    **No provider call is made and none is needed.** The index is built from
    local files with Chroma's bundled embedding function. The credential is
    nonetheless required, because :func:`~agent_lab.config.load_config` is the
    one validated path to the corpus and chunk settings and it refuses a missing
    or empty API key (Req 1.8). Nothing here contacts the provider.
    """
    config = load_config(context.env, stream=context.stderr)
    renderer = build_renderer(
        config,
        stdout=context.stdout,
        stderr=context.stderr,
        narrate=False,
    )
    out = context.stdout if context.stdout is not None else sys.stdout

    _print(out, f"Indexing {config.docs_dir} into {config.chroma_dir}", config.api_key)
    _print(
        out,
        f"Chunk settings: size={config.chunk_size}  overlap={config.chunk_overlap}",
        config.api_key,
    )

    try:
        # Req 7.9: the loader's skipped-file lines go to the same stream the
        # counts do, so the operator reads the reasons and the totals together.
        report = build_index(config, stream=out)
    except ChromaUnavailable as exc:
        renderer.error(f"the document index was not built: {exc}")
        return INDEX_FAILURE

    # Req 7.4: the three counts, in one line, in the shape of the run summary.
    _print(
        out,
        "Index summary: "
        f"files_ingested={report.files_ingested}  "
        f"files_skipped={len(report.files_skipped)}  "
        f"chunks_indexed={report.chunks_indexed}",
        config.api_key,
    )
    if report.chunks_indexed == 0:
        # Not an error: an empty corpus produces an empty collection, and
        # `select_retriever` then sends searches to the keyword fallback
        # (Req 7.6). Said plainly, because an operator who expected their
        # documents to be searchable has learned nothing from "0" alone.
        renderer.warning(
            f"warning: no chunks were indexed, so searches will use keyword "
            f"fallback over {config.docs_dir}. Check that the directory exists "
            f"and holds .txt, .md or .pdf files."
        )
    return exit_codes.FINAL_ANSWER


def _print(stream: IO[str], text: str, secret: str | None) -> None:
    """Write one redacted line to ``stream`` and flush it (Req 1.11, 13.9).

    The ``index`` command reports counts and configured paths rather than model
    output, so there is nothing here that should carry the credential -- but
    "should not" is not "cannot", and every other write path in this project
    passes through :func:`~agent_lab.redaction.redact`. This one does too, so
    there is no sink left to audit separately.

    A stream that cannot be written to does not fail the build: the collection
    is already on disk by the time the summary is printed, and a closed stdout
    must not turn a completed index into a non-zero exit.
    """
    try:
        stream.write(redact(text, secret) + "\n")
        stream.flush()
    except (OSError, ValueError):  # a closed or broken stream is not a build failure
        pass


# --------------------------------------------------------------------------- #
# The `eval` subcommand (Req 15.1, 15.2, 15.13)
# --------------------------------------------------------------------------- #


def add_eval_arguments(parser: argparse.ArgumentParser) -> None:
    """Declare the arguments of ``agentlab eval``: one optional identifier.

    Req 15.1 asks for a single command that, invoked with no identifier, runs
    every task in declaration order; Req 15.13 asks that the same command with
    exactly one identifier run only that task. One optional positional is
    exactly that pair of invocations and nothing else, which is why ``nargs="?"``
    rather than a ``--task`` option: ``agentlab eval`` and
    ``agentlab eval calc_chain`` are the two documented forms.

    No ``--non-interactive`` flag is offered here, deliberately. Req 15.2 makes
    non-interactive mode a property of the suite rather than a choice the
    operator makes per invocation, so
    :func:`~agent_lab.evals.runner.run_eval` selects it unconditionally. A flag
    would imply the mode could be turned off, and a suite that stopped on a
    confirmation prompt half way through would not be a suite.

    Everything else the suite needs -- the cost ceiling of Req 15.8, the corpus,
    the model, the operator's own notes and trace paths that the evaluation
    paths are checked against -- is already configuration with a documented
    default (Req 1.6, 1.7), so, as with ``index``, no flag duplicates it.
    """
    parser.add_argument(
        "task_id",
        metavar="TASK_ID",
        nargs="?",
        default=None,
        help=(
            "run only the evaluation task with this identifier. Omit it to run "
            "the whole suite in declaration order. An identifier no task "
            "carries is refused before anything runs."
        ),
    )


def eval_command(args: argparse.Namespace, context: CliContext) -> int:
    """Run the eval suite, or one task of it, and return its exit code (Req 15).

    One delegation, and that is the point. Everything a handler would otherwise
    assemble here -- loading configuration in non-interactive mode (Req 15.2),
    selecting the tasks (Req 15.1, 15.13), redirecting the notes and trace
    paths away from the operator's and refusing a collision (Req 15.3, 15.4),
    honouring ``config.eval_cost_ceiling`` (Req 15.8), writing each task's own
    trace file, printing the table and the aggregates (Req 15.9, 15.10),
    writing ``evals/results.md`` while keeping its hand-maintained section
    (Req 15.11, 15.12), and returning 0 only for an all-pass suite
    (Req 15.15) -- belongs to :func:`~agent_lab.evals.runner.run_eval`. Req 1.1
    is why it is arranged this way round: the runner may not import the CLI, so
    the one entry point it exposes for a command to call is the thing to call,
    and the suite stays runnable as ``agent_lab.evals.runner`` by anything else.

    Two error shapes reach a caller, and neither is handled here.
    :class:`~agent_lab.evals.runner.EvalRunnerError` (an unknown identifier,
    Req 15.14, or a path collision, Req 15.4) and
    :class:`~agent_lab.evals.schema.EvalSuiteError` (a defective suite,
    Req 14.13) are already caught inside ``run_eval``, which prints the message
    and returns :data:`~agent_lab.exit_codes.EVAL_FAILURE`.
    :class:`~agent_lab.config.ConfigError` is deliberately left to propagate --
    a missing credential or an out-of-range setting is exit 6 and :func:`main`
    has the project's single handler for it (Req 1.8).

    **This command spends money.** It issues one real run per task against the
    configured provider, which is what Req 15.11's cost figure measures and
    what the ceiling of Req 15.8 bounds.
    """
    return run_eval(
        args.task_id,
        env=context.env,
        stdout=context.stdout,
        stderr=context.stderr,
    )


# --------------------------------------------------------------------------- #
# The subcommand table
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Subcommand:
    """One ``agentlab`` subcommand: its name, its help, its arguments, its code."""

    name: str
    help: str
    configure: Callable[[argparse.ArgumentParser], None]
    handler: Callable[[argparse.Namespace, CliContext], int]


RUN = Subcommand(
    name="run",
    help="perform one task and print the answer, the steps and the run summary",
    configure=add_run_arguments,
    handler=run_command,
)

INDEX = Subcommand(
    name="index",
    help=(
        "rebuild the document index over the configured corpus and report "
        "files ingested, files skipped and chunks indexed"
    ),
    configure=add_index_arguments,
    handler=index_command,
)

EVAL = Subcommand(
    name="eval",
    help=(
        "run the evaluation suite, or one task of it, without an operator at "
        "the keyboard, then print the results table and write evals/results.md"
    ),
    configure=add_eval_arguments,
    handler=eval_command,
)

#: Every subcommand, in the order ``--help`` lists them. Each is one entry, with
#: no change to :func:`build_parser` or :func:`main`.
SUBCOMMANDS: tuple[Subcommand, ...] = (RUN, INDEX, EVAL)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


class _Parser(argparse.ArgumentParser):
    """An ``ArgumentParser`` whose usage errors exit 6, not 2.

    argparse exits with code 2 on a bad command line, and 2 is this project's
    documented "step limit reached" code (Req 4.10). A script checking the exit
    code of a run must not read a typo in the command line as a run that used
    its whole step budget, so a usage error is reported as what it is: the
    operator's configuration of this invocation was wrong (exit 6).
    """

    def error(self, message: str) -> None:  # type: ignore[override]
        self.print_usage(sys.stderr)
        self.exit(exit_codes.CONFIGURATION_ERROR, f"{self.prog}: error: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    """The argument parser for every subcommand in :data:`SUBCOMMANDS`."""
    parser = _Parser(prog=PROGRAM, description=DESCRIPTION)
    parser.add_argument(
        "--version",
        action="version",
        version=f"{PROGRAM} {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    for command in SUBCOMMANDS:
        sub = subparsers.add_parser(
            command.name,
            help=command.help,
            description=command.help,
        )
        command.configure(sub)
        sub.set_defaults(handler=command.handler)
    return parser


# --------------------------------------------------------------------------- #
# The entry point
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None, context: CliContext | None = None) -> int:
    """Parse ``argv``, dispatch to a subcommand, and return its exit code.

    Declared in ``pyproject.toml`` as ``agentlab = "agent_lab.cli:main"``, so
    the return value becomes the process exit status.

    ``--help`` and ``--version`` raise ``SystemExit`` from inside argparse;
    those are caught and turned into a return value so that this function
    always returns an ``int`` and can be called directly from a test. A
    subcommand handler's own exceptions are not caught here, with one
    exception: :class:`~agent_lab.config.ConfigError` and its subclasses, every
    startup refusal in the project, become exit 6 with the message on stderr
    (Req 1.8, 1.9, 2.9).
    """
    ctx = context if context is not None else CliContext()
    parser = build_parser()

    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return exit_codes.FINAL_ANSWER
        return code if isinstance(code, int) else exit_codes.CONFIGURATION_ERROR

    handler = getattr(args, "handler", None)
    if handler is None:
        # No subcommand. argparse can require one, but its message names the
        # argument rather than saying what to do, so the help text is better.
        parser.print_help(ctx.err)
        return exit_codes.CONFIGURATION_ERROR

    try:
        return handler(args, ctx)
    except ConfigError as exc:
        # Req 1.8, 1.9, 2.9: one type covers a missing credential, an
        # out-of-range value, an unknown provider, an unreadable system prompt,
        # a malformed price table and an out-of-range display limit. Nothing has
        # been requested from the provider and no trace line has been written.
        print(str(exc), file=ctx.err)
        return exit_codes.CONFIGURATION_ERROR


__all__ = [
    "EVAL",
    "INDEX",
    "INDEX_FAILURE",
    "PROGRAM",
    "RUN",
    "STREAMING_UNAVAILABLE_MESSAGE",
    "STREAM_MISMATCH_MESSAGE",
    "SUBCOMMANDS",
    "CliContext",
    "Subcommand",
    "add_eval_arguments",
    "add_index_arguments",
    "add_run_arguments",
    "build_parser",
    "build_retriever",
    "eval_command",
    "index_command",
    "main",
    "run_command",
]


if __name__ == "__main__":  # pragma: no cover - module execution convenience
    raise SystemExit(main())
