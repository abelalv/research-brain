"""Command-line interface: ``rbrain <command>``.

Run ``rbrain --help`` or ``rbrain <command> --help`` for details.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .agent import agent_env, export_lines, launch
from .bootstrap import doctor, init_vault
from .config import find_vault, load_config
from .connect import Wizard, list_connections
from .llm import LLMError
from .logic import FormulaError, check_argument, classify, truth_table
from .providers import ollama_pull, ollama_status
from .skills import list_skills
from .workflows import Brain


def _rel(brain: Brain, path: Path) -> str:
    try:
        return str(path.relative_to(brain.vault.root))
    except ValueError:
        return str(path)


def _brain(args) -> Brain:
    cfg = load_config(find_vault(args.vault))
    cfg.forced_profile = args.profile
    return Brain(cfg)


def _print_created(brain: Brain, notes) -> None:
    for n in notes:
        print(f"  + {_rel(brain, n.path)}")


# ------------------------------------------------------------------ command handlers
def cmd_init(args) -> None:
    created = init_vault(Path(args.path), force=args.force, link_skills=args.link_skills,
                         set_default=not args.no_default)
    root = Path(args.path).expanduser().resolve()
    print(f"Vault ready at {root} ({len(created)} files created).")
    print("Next steps:\n"
          f"  1. Open {root} as a vault in Obsidian.\n"
          f"  2. Edit {root}/.rbrain/config.toml (models), then run: rbrain doctor\n"
          "  3. Try: rbrain explain \"physics-informed neural network\"\n"
          f"  4. Claude Code (optional): cd {root} && claude")


def cmd_doctor(args) -> None:
    cfg = load_config(find_vault(args.vault))
    ok_all = True
    for ok, msg in doctor(cfg, live=args.live):
        ok_all &= ok
        print(f"  {'✔' if ok else '✘'} {msg}")
    if not ok_all:
        print("\n  Fix connections interactively with: rbrain connect")
    sys.exit(0 if ok_all else 1)


def cmd_connect(args) -> None:
    vault = find_vault(args.vault)
    if args.list or args.test:
        list_connections(vault, live=args.test)
        return
    skills = [s.strip() for s in args.skills.split(",")] if args.skills else None
    use = "skills" if skills else ("default" if args.default else ("optional" if args.optional else None))
    interactive = sys.stdin.isatty() and not args.yes
    Wizard(vault, interactive=interactive).run(
        provider=args.provider, name=args.name, model=args.model, base_url=args.base_url,
        key_env=args.key_env, use=use, skills=skills, test=not args.no_test, embeddings=args.embeddings)


def cmd_agent(args) -> None:
    vault = find_vault(args.vault)
    cfg = load_config(vault)
    ollama_default = next((p.model for p in cfg.all_profiles().values() if p.provider == "ollama"), "")
    env, shown, notes = agent_env(args.backend, model=args.model, base_url=args.base_url,
                                  token_env=args.token_env, default_ollama_model=ollama_default)
    if args.print_env:
        print(export_lines(shown) or "# nothing to set: plain `claude` uses your Claude login")
        print(f"cd {vault} && claude")
        return
    for note in notes:
        print(f"  · {note}")
    for k, v in shown.items():
        print(f"  {k}={v}")
    print(f"Starting Claude Code in {vault} …")
    launch(vault, env, args.claude_args)


def cmd_ollama(args) -> None:
    st = ollama_status()
    if not st.running:
        where = f"binary at {st.binary}" if st.binary else "no ollama binary on this machine"
        print(f"  ✘ Ollama not reachable at {st.url} ({where}).")
        print("    Start it: sudo systemctl start ollama   (or: ollama serve)")
        print("    WSL with Ollama on Windows: start the Windows app; with mirrored networking it is")
        print("    reachable at localhost, otherwise export OLLAMA_HOST=<windows-ip>:11434")
        sys.exit(1)
    print(f"  ✔ Ollama v{st.version} at {st.url}" + ("" if st.binary else " (remote/Windows server)"))
    for m in st.models:
        print(f"      {m}")
    wanted: list[str] = list(args.pull or [])
    try:
        cfg = load_config(find_vault(args.vault))
        needed = {p.model for p in cfg.all_profiles().values() if p.provider == "ollama"}
        if cfg.embeddings.enabled and cfg.embeddings.provider == "ollama":
            needed.add(cfg.embeddings.model)
        missing = sorted(m for m in needed if not st.has(m))
        print("  Models used by this vault: " + ", ".join(sorted(needed)))
        print("  Missing: " + (", ".join(missing) if missing else "none"))
        if args.pull_missing:
            wanted += missing
    except FileNotFoundError:
        pass
    for model in wanted:
        if not args.yes and sys.stdin.isatty() and input(f"  Pull {model}? [Y/n] ").strip().lower() == "n":
            continue
        ollama_pull(model, st.url)
        print(f"  ✔ {model}")


def cmd_ingest(args) -> None:
    brain = _brain(args)
    print(f"Ingesting {args.file} …")
    notes = brain.ingest(args.file, title=args.title, segment=not args.no_segment,
                         max_chunks=args.max_chunks, use_marker=args.marker)
    _print_created(brain, notes)


def cmd_segment(args) -> None:
    brain = _brain(args)
    source = None
    if args.text:
        text, label = args.text, "inline text"
    else:
        note = brain.vault.resolve(args.target)
        if note is not None:
            text, label = note.body, note.title
            source = note if note.type in {"source", "summary"} else None
        else:
            path = Path(args.target).expanduser()
            text, label = path.read_text(encoding="utf-8"), path.stem
    print(f"Segmenting '{label}' …")
    notes = brain.segment(text, source=source, max_chunks=args.max_chunks, label=label)
    _print_created(brain, notes)


def cmd_synthesize(args) -> None:
    brain = _brain(args)
    print("Synthesizing new ideas …")
    _print_created(brain, brain.synthesize(args.notes, query=args.query, n=args.n))


def cmd_explain(args) -> None:
    brain = _brain(args)
    print(f"Explaining '{args.concept}' ({args.level}) …")
    _print_created(brain, [brain.explain(args.concept, level=args.level)])


def cmd_deduce(args) -> None:
    brain = _brain(args)
    print("Reasoning …")
    note, check = brain.deduce(args.premise, args.goal, rounds=args.rounds)
    print(f"  {check.verdict}")
    _print_created(brain, [note])


def _print_answer(ans, show_context: bool = False) -> None:
    print("\n" + ans.text.strip() + "\n")
    print("  " + ans.diagnostics)
    if show_context:
        print("\n--- context sent to the model ---\n" + (ans.context.text or "(empty)"))


def cmd_ask(args) -> None:
    brain = _brain(args)
    ans = brain.ask(args.question, save=args.save, k=args.k, scope=args.scope,
                    expand=args.expand, rerank=args.rerank)
    _print_answer(ans, args.show_context)
    if ans.note:
        _print_created(brain, [ans.note])


def cmd_chat(args) -> None:
    brain = _brain(args)
    history: list[tuple[str, str]] = []
    last = None
    scope, expand, rerank = args.scope, args.expand, args.rerank
    print("Research-brain chat. Ask about your vault; follow-up questions keep the context.\n"
          "Commands: /sources  /save  /scope all|wiki|sources  /expand  /rerank  /clear  /exit")
    while True:
        try:
            msg = input("\n› ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not msg:
            continue
        cmd, _, arg = msg.partition(" ")
        if cmd in {"/exit", "/quit", "/q"}:
            break
        if cmd == "/clear":
            history.clear()
            print("  (conversation cleared)")
            continue
        if cmd == "/scope":
            scope = arg.strip() if arg.strip() in {"all", "wiki", "sources"} else scope
            print(f"  scope = {scope or brain.cfg.rag.scope}")
            continue
        if cmd in {"/expand", "/rerank"}:
            if cmd == "/expand":
                expand = not (expand if expand is not None else brain.cfg.rag.expand_queries)
                print(f"  query expansion {'on' if expand else 'off'}")
            else:
                rerank = not (rerank if rerank is not None else brain.cfg.rag.rerank)
                print(f"  reranking {'on' if rerank else 'off'}")
            continue
        if cmd == "/sources":
            print(last.context.text if last else "  (no answer yet)")
            continue
        if cmd == "/save":
            if last is None:
                print("  (no answer yet)")
            else:
                _print_created(brain, [brain.save_answer(last)])
            continue
        question = brain.condense(msg, history)
        if question != msg:
            print(f"  · searching for: {question}")
        last = brain.ask(question, scope=scope, expand=expand, rerank=rerank, history=history, k=args.k)
        _print_answer(last)
        history.append((msg, last.raw))


def cmd_mcp(args) -> None:
    from .mcp_server import install, serve
    if args.install:
        path = install(args.vault)
        print(f"  ✔ wrote {path}\n  Claude Code started in the vault will offer the 'research-brain' tools "
              "(approve them once).\n  User-wide instead: claude mcp add research-brain -- rbrain --vault "
              f"{path.parent} mcp")
        return
    serve(args.vault)


def cmd_rag(args) -> None:
    brain = _brain(args)
    if args.rag_cmd == "index":
        stats = brain.refresh_rag(force=True, rebuild=args.rebuild)
        print(f"  {stats}")
        return
    if args.rag_cmd == "stats":
        brain.refresh_rag()
        for key, value in brain.rag().stats().items():
            print(f"  {key:15} {value}")
        return
    hits = brain.retrieve(args.query, k=args.k, scope=args.scope, expand=args.expand, rerank=args.rerank)
    if not hits:
        print("  no passages found")
    for h in hits:
        c = h.chunk
        print(f"\n[{h.rank}] {c.location()}   ({c.note_type}, score {h.score:.4f})\n    {c.link()}")
        text = c.text if args.full else c.text[:300].replace("\n", " ") + ("…" if len(c.text) > 300 else "")
        print("    " + text.replace("\n", "\n    "))


def cmd_search(args) -> None:
    brain = _brain(args)
    index = brain.search_index(embeddings=not args.keyword)
    types = set(args.type) if args.type else None
    for note, score in index.search(args.query, k=args.k, types=types):
        print(f"  {score:.4f}  [{note.type:8}] {_rel(brain, note.path)} — {note.title}")
    if index.embedding_error:
        print(f"  (embeddings unavailable, keyword search only: {index.embedding_error})")


def cmd_index(args) -> None:
    brain = _brain(args)
    path = brain.vault.update_index()
    print(f"  index → {_rel(brain, path)}")
    if args.embed:
        index = brain.search_index()
        vecs = index.ensure_embeddings()
        print("  embeddings:", "up to date" if vecs is not None else f"unavailable ({index.embedding_error})")


def cmd_lint(args) -> None:
    brain = _brain(args)
    r = brain.lint(use_llm=args.llm)
    print(f"  missing pages: {len(r.broken)} · orphans: {len(r.orphans)} · unsourced: {len(r.unsourced)}"
          f" · duplicates: {len(r.duplicates)} · invalid arguments: {len(r.invalid_arguments)}")
    print(f"  report → {_rel(brain, r.path)}")


def cmd_logic(args) -> None:
    try:
        if args.logic_cmd == "check":
            r = check_argument(args.premises, args.goal)
            print(r.verdict)
            if r.counterexample:
                print("Counter-model:", ", ".join(f"{k}={'T' if v else 'F'}" for k, v in sorted(r.counterexample.items())))
            if r.valid and r.essential:
                print("Essential premises:", ", ".join(f"P{i + 1}: {args.premises[i]}" for i in r.essential))
        elif args.logic_cmd == "table":
            print(truth_table(args.formula))
        elif args.logic_cmd == "classify":
            print(classify(args.formula))
    except FormulaError as exc:
        sys.exit(f"Formula error: {exc}")


def cmd_skills(args) -> None:
    vault = None
    try:
        vault = find_vault(args.vault)
    except FileNotFoundError:
        pass
    for s in list_skills(vault):
        print(f"  {s.name:22} {s.description[:100]}")


# ------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rbrain", description="Research second brain: Obsidian + local LLM or Claude.")
    p.add_argument("--version", action="version", version=f"research-brain {__version__}")
    p.add_argument("--vault", help="vault path (default: auto-detect)")
    p.add_argument("--profile", help="force an LLM profile from config.toml for this command (e.g. claude)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="create a new vault")
    s.add_argument("path")
    s.add_argument("--force", action="store_true", help="overwrite template files and skills")
    s.add_argument("--link-skills", action="store_true", help="symlink skills instead of copying (development)")
    s.add_argument("--no-default", action="store_true", help="do not record this vault as the default one")
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("doctor", help="check configuration, models and embeddings")
    s.add_argument("--live", action="store_true", help="also send a one-line test prompt to each model")
    s.set_defaults(func=cmd_doctor)

    s = sub.add_parser("connect", help="connect a model: Ollama, Claude (subscription or API), DeepSeek, ...")
    s.add_argument("provider", nargs="?", help="ollama, lmstudio, claude_code, anthropic, deepseek, openrouter, "
                                              "gemini, groq, mistral, openai, together, openai_compat, litellm")
    s.add_argument("--name", help="profile name (default: provider name)")
    s.add_argument("--model")
    s.add_argument("--base-url")
    s.add_argument("--key-env", help="environment variable holding the API key")
    g = s.add_mutually_exclusive_group()
    g.add_argument("--default", action="store_true", help="make it the default for every skill")
    g.add_argument("--skills", help="comma-separated skills that should use it")
    g.add_argument("--optional", action="store_true", help="only on demand with --profile")
    s.add_argument("--embeddings", action="store_true", help="configure it as the embedding model instead")
    s.add_argument("--no-test", action="store_true", help="skip the test prompt")
    s.add_argument("--yes", "-y", action="store_true", help="non-interactive: accept defaults")
    s.add_argument("--list", action="store_true", help="show configured profiles and their status")
    s.add_argument("--test", action="store_true", help="like --list, and send a test prompt to each")
    s.set_defaults(func=cmd_connect)

    s = sub.add_parser("agent", help="open Claude Code in the vault (Claude, DeepSeek or Ollama backend)")
    s.add_argument("--backend", default="claude", choices=["claude", "deepseek", "ollama", "custom"])
    s.add_argument("--model")
    s.add_argument("--base-url", help="custom backend: Anthropic-compatible endpoint")
    s.add_argument("--token-env", help="custom backend: variable holding the token")
    s.add_argument("--print-env", action="store_true", help="print the export lines instead of launching")
    s.add_argument("claude_args", nargs=argparse.REMAINDER, help="arguments after -- go to claude")
    s.set_defaults(func=cmd_agent)

    s = sub.add_parser("ollama", help="Ollama status: server, installed models, models this vault needs")
    s.add_argument("--pull-missing", action="store_true", help="pull the models the config uses but lacks")
    s.add_argument("--pull", action="append", help="pull a specific model (repeatable)")
    s.add_argument("--yes", "-y", action="store_true")
    s.set_defaults(func=cmd_ollama)

    s = sub.add_parser("ingest", help="add a source (md, txt, tex, html, pdf): summary + atoms")
    s.add_argument("file")
    s.add_argument("--title")
    s.add_argument("--no-segment", action="store_true", help="only write the summary page")
    s.add_argument("--max-chunks", type=int, default=6)
    s.add_argument("--marker", action="store_true", help="use Marker for PDF extraction (better math)")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("segment", help="idea segmentation: split a note/file/text into atomic ideas")
    s.add_argument("target", nargs="?", help="note name, [[link]] or file path")
    s.add_argument("--text", help="segment this text instead")
    s.add_argument("--max-chunks", type=int, default=6)
    s.set_defaults(func=cmd_segment)

    s = sub.add_parser("synthesize", help="construct new ideas from existing notes")
    s.add_argument("notes", nargs="*", help="parent notes (names or [[links]])")
    s.add_argument("--query", "-q", help="research focus; also pulls related notes by search")
    s.add_argument("--n", type=int, default=3, help="number of ideas")
    s.set_defaults(func=cmd_synthesize)

    s = sub.add_parser("explain", help="write a layered concept page")
    s.add_argument("concept")
    s.add_argument("--level", default="graduate", choices=["intro", "undergraduate", "graduate", "expert"])
    s.set_defaults(func=cmd_explain)

    s = sub.add_parser("deduce", help="deductive reasoning with symbolic validity check")
    s.add_argument("--premise", "-p", action="append", required=True,
                   help="premise text or [[note]] (repeatable)")
    s.add_argument("--goal", "-g", required=True, help="conclusion to establish")
    s.add_argument("--rounds", type=int, default=2, help="max repair rounds if invalid")
    s.set_defaults(func=cmd_deduce)

    def rag_flags(parser) -> None:
        parser.add_argument("-k", type=int, default=None, help="passages to retrieve ([rag] top_k)")
        parser.add_argument("--scope", choices=["all", "wiki", "sources"], default=None,
                            help="all notes, only your wiki notes, or only raw sources")
        parser.add_argument("--expand", action=argparse.BooleanOptionalAction, default=None,
                            help="LLM query variants incl. English<->Spanish")
        parser.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=None,
                            help="LLM reranking of the candidates")

    s = sub.add_parser("ask", help="RAG: answer from the vault with passage citations")
    s.add_argument("question")
    s.add_argument("--save", action="store_true", help="file the answer back into 70-answers")
    s.add_argument("--show-context", action="store_true", help="print the passages sent to the model")
    rag_flags(s)
    s.set_defaults(func=cmd_ask)

    s = sub.add_parser("chat", help="RAG chat with follow-up questions (/sources, /save, /scope...)")
    rag_flags(s)
    s.set_defaults(func=cmd_chat)

    s = sub.add_parser("mcp", help="MCP server: the vault's RAG and tools for Claude Code / Claude Desktop")
    s.add_argument("--install", action="store_true", help="write <vault>/.mcp.json for Claude Code")
    s.set_defaults(func=cmd_mcp)

    s = sub.add_parser("rag", help="RAG index: build, inspect, search passages")
    rsub = s.add_subparsers(dest="rag_cmd", required=True)
    c = rsub.add_parser("index", help="index new/changed notes (incremental)")
    c.add_argument("--rebuild", action="store_true", help="re-chunk and re-embed everything")
    c = rsub.add_parser("search", help="show the passages a question would retrieve")
    c.add_argument("query")
    c.add_argument("--full", action="store_true", help="print whole passages")
    rag_flags(c)
    rsub.add_parser("stats", help="passages per note type, model, backend")
    s.set_defaults(func=cmd_rag)

    s = sub.add_parser("search", help="hybrid search (BM25 + embeddings)")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=10)
    s.add_argument("--type", action="append", help="filter by note type (repeatable)")
    s.add_argument("--keyword", action="store_true", help="BM25 only")
    s.set_defaults(func=cmd_search)

    s = sub.add_parser("index", help="rebuild index.md (and the embedding cache with --embed)")
    s.add_argument("--embed", action="store_true")
    s.set_defaults(func=cmd_index)

    s = sub.add_parser("lint", help="health check: missing pages, orphans, duplicates, unsourced claims")
    s.add_argument("--llm", action="store_true", help="also ask the LLM for research questions")
    s.set_defaults(func=cmd_lint)

    s = sub.add_parser("logic", help="propositional logic tools (no LLM)")
    lsub = s.add_subparsers(dest="logic_cmd", required=True)
    c = lsub.add_parser("check", help="is GOAL a consequence of the premises?")
    c.add_argument("premises", nargs="+")
    c.add_argument("--goal", "-g", required=True)
    c = lsub.add_parser("table", help="truth table of a formula")
    c.add_argument("formula")
    c = lsub.add_parser("classify", help="tautology / contradiction / contingent")
    c.add_argument("formula")
    s.set_defaults(func=cmd_logic)

    s = sub.add_parser("skills", help="list available skills")
    s.set_defaults(func=cmd_skills)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.cmd == "segment" and not (args.target or args.text):
        sys.exit("segment: give a note/file or --text")
    if args.cmd == "agent" and args.claude_args[:1] == ["--"]:
        args.claude_args = args.claude_args[1:]
    try:
        args.func(args)
    except (FileNotFoundError, ValueError, LLMError, KeyError) as exc:
        sys.exit(f"Error: {exc}")
    except KeyboardInterrupt:
        sys.exit("\nCancelled.")


if __name__ == "__main__":
    main()
