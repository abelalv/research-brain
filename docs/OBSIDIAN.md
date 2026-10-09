# Obsidian setup

## 1. Open the vault

Obsidian → *Open folder as vault* → choose the folder created by `rbrain init`
(e.g. `~/ResearchBrain`). The vault ships with settings in `.obsidian/`:

| Setting | Value | Why |
|---|---|---|
| Attachment folder | `10-sources/assets` | images and PDFs stay next to the immutable sources |
| New notes | `00-inbox` | everything new starts in the inbox |
| Links | `[[wikilinks]]`, shortest path, auto-update on rename | slugs are unique, so short links are unambiguous |
| Templates folder | `90-templates` | atom, idea, concept, argument, summary, fleeting thought |
| Graph colours | atoms blue, concepts green, ideas orange, arguments purple, summaries teal, answers pink | see the knowledge pipeline at a glance |

If the **Templates** core plugin is off, enable it in *Settings → Core plugins*.

## 2. Community plugins

*Settings → Community plugins → Turn on community plugins → Browse.*

- **Dataview** (needed for `Dashboard.md`): ideas ranked by score, arguments that need
  work, low-confidence atoms, atoms by kind, recent concepts and sources. All notes carry
  YAML frontmatter, so you can write your own queries, e.g.
  ```dataview
  LIST FROM "30-atoms" WHERE kind = "assumption" AND contains(source, "porous")
  ```
- **Obsidian Git** (recommended): automatic commits of the vault. The wiki is a git
  repository of Markdown, so history and diffs come for free; `.gitignore` already excludes
  runtime caches.
- **Copilot** (optional): chat with a local model inside Obsidian — see section 3.
- **Smart Connections** (optional): a "related notes" side panel computed with local
  embeddings; complements `rbrain search`.
- **Marp Slides** (optional): turn concept pages into slides for class.

Browser: the **Obsidian Web Clipper** extension saves web pages as Markdown — point it to
`00-inbox/`, then `rbrain ingest` the clipped file.

## 3. Copilot with a local model (Ollama)

This is the setup described in the XDA article, adapted to the research skills. Copilot is
for quick, interactive work on a selection; the `rbrain` CLI is for work that must create
and link notes.

**3.1 Allow Obsidian to call Ollama (CORS).** Ollama must accept requests from the
Obsidian app origin. On Ubuntu, where Ollama runs as a systemd service:

```bash
sudo systemctl edit ollama
```

add

```ini
[Service]
Environment="OLLAMA_ORIGINS=app://obsidian.md*"
```

then

```bash
sudo systemctl daemon-reload && sudo systemctl restart ollama
```

(If you start Ollama by hand: `OLLAMA_ORIGINS="app://obsidian.md*" ollama serve`.) Set the
variable **before** Ollama starts; setting it later is the most common cause of CORS errors.

**3.2 A model with a larger context.** Plugins may not set Ollama's context length, and the
default is short. Create a variant with a larger context once:

```bash
cat > Modelfile <<'EOF'
FROM qwen2.5:7b-instruct-q4_K_M
PARAMETER num_ctx 16384
EOF
ollama create qwen2.5-7b-16k -f Modelfile
```

**3.3 Configure Copilot.** In Copilot's settings, add a custom chat model with provider
*Ollama*, model name `qwen2.5-7b-16k` and base URL `http://localhost:11434`, and make it the
default. For vault-wide questions, add `bge-m3` as an Ollama embedding model and let Copilot
index the vault. Restart Obsidian if the model does not appear.

**3.4 Research prompts.** `rbrain init` installs four custom prompts in
`copilot-custom-prompts/`:

- *RB – Segment into atomic ideas*
- *RB – Construct new ideas*
- *RB – Explain concept*
- *RB – Check deductive argument* (ends with the `rbrain logic check` command to verify)

Point Copilot's custom-prompts folder setting to `copilot-custom-prompts` if it is not
already the default. Select text in a note and apply a prompt from the command palette or
the context menu. In the prompts, `{}` stands for the selected text; Copilot's template
syntax has changed between versions, so check its documentation if a prompt receives no
text.

## 4. Useful hotkeys

- *Templates: Insert template* — e.g. `Ctrl+T`.
- *Download attachments for current file* (Karpathy's tip) — e.g. `Ctrl+Shift+D` after
  clipping an article, so images are stored locally in `10-sources/assets/`.
- *Open graph view* and *Open local graph* — to see what a new note connected to.

## 5. Several computers

The vault is a plain folder. Git (with Obsidian Git) is the safest way to sync between
machines; a synced folder (Obsidian Sync, Syncthing, rclone) also works. Exclude
`.rbrain/embeddings.json` and `.obsidian/workspace*.json` from sync — they are per-machine
caches and are rebuilt automatically.
