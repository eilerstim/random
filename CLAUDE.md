# Notes for working in this repo

## What this repo is

Source-level deep dives into how ML frameworks structure their model files and what happens, step by step, when a model is loaded. Each framework gets its own directory with the same set of artifacts so the deep dives can be compared side by side. The audience is technical (people who read framework source and care about the security of model loading), so precision and code references matter more than polish.

## Layout

```
README.md                      index of frameworks
CLAUDE.md                      this file: method, conventions, backlog
tools/structure_figure.py      JSON spec -> SVG "model structure" figure, shared style for all frameworks
tools/render_png.cjs           SVG -> PNG with headless Chromium (playwright); optional
<framework>/README.md          the deep dive (structure + load path + security notes + source map)
<framework>/figures/           <framework>_structure.json (spec), .svg (source of truth), .png (rendered)
<framework>/scripts/           reproducible experiments; stdlib + the framework only
<framework>/scripts/*_output_<framework>-<version>.txt   captured output, kept as evidence for the version studied
```

Done so far: `pytorch/` (torch 2.14.0, tag v2.14.0).

## Method for a deep dive

1. **Pin a version.** Install the framework (`pip install torch --index-url https://download.pytorch.org/whl/cpu` worked for PyTorch, no GPU needed). Sparse-clone the upstream repo at the matching tag for the parts the wheel does not ship (C/C++):
   `git clone --filter=blob:none --no-checkout --depth 1 --branch vX.Y.Z <url> && git sparse-checkout init --no-cone && git sparse-checkout set <paths> && git checkout`.
   Diff the wheel's Python files against the checkout for every file you cite and say so in the deep dive.
2. **Read the loader top-down.** Start at the public load function, follow every call into the container parser (zip/tar/protobuf/HDF5/...), the object reconstruction (pickle, JSON config, protobuf graph), and the device/dtype handling. Read the C/C++ layer too; that is where the container format is actually defined.
3. **Read the saver as well.** The writer tells you exactly which records exist and in which order; the reader often tolerates more than the writer produces.
4. **Confirm empirically.** Write `scripts/inspect_<something>.py` that saves a tiny model and dumps the container table, the serialized structure (e.g. `pickletools.dis`), the bookkeeping records, and the behaviour of the safety switches. Commit the script and its output for the pinned version. Never trust a claim about bytes on disk that the script did not show.
5. **Date the features.** For each record/flag, find the release it appeared in by fetching old tags into the sparse clone (`git fetch --depth 1 origin tag vA.B.0`) and grepping `git show vA.B.0:path`. Put the result in a "Since" column.
6. **Cite `path:line` at the pinned tag.** Re-grep every line number before committing; they drift while writing.
7. **Cross-check with the official docs**, but describe behaviour from the code. Note where docs and code disagree.

## Deep dive document template

Use these sections in this order so the frameworks line up:

1. Header: version studied, citation format, pointer to the evidence script/output.
2. TL;DR (5 or so numbered facts).
3. Part 1, structure: in-memory model object; what gets serialized (and the alternatives, e.g. weights-only vs whole model); on-disk container with the structure figure and a record table (entry, written by, content, since); a real dump from the script; the serialized-structure internals; legacy formats; other files the loader accepts.
4. Part 2, loading: the text flow chart, then step-by-step narration with code references, then focused subsections on the security-relevant component (the deserializer), device mapping, memory mapping, and the second stage that binds weights to the model object.
5. Part 3, security-relevant observations: what executes code, what the mitigation is, how it gets switched off, what is parsed natively before any safe layer runs, how to inspect a file statically, and the guidance that follows from the format.
6. Part 4, reproduce: exact commands.
7. Part 5, source map table.

Writing rules: short sentences, one fact per sentence, no hedging where the code is unambiguous, tables for parallel facts, fenced blocks for dumps and code. No em dashes.

## Figure conventions

Two figures per framework.

**Figure 1: load-path flow chart (text, inside the README).** Match the density of the owner's Keras example, not more; the PyTorch one was first written far too detailed and had to be cut down.
- The file tree at the top with bare entry names (`├──`, `└──`), then `│ / ▼` connectors, then numbered steps with circled digits `① ② ③ ...`, ending in `Final model`.
- One short label per step: a function call (`torch.load()`) or a three-to-five-word action (`Open the .pt ZIP`). No trailing explanations, argument lists or line references in the figure; those belong in the step-by-step prose, whose step numbers must match the figure.
- Sub-bullets only where a step fans out (look-ups, branches): two or three short phrases at an 8-space indent with `├──` / `└──`.
- Unnumbered result nodes between steps (`Create each Tensor`, `Complete state_dict`), optionally with an `e.g.:` list of three or four items.
- About nine steps. The only extra the PyTorch figure carries relative to the Keras example is the `weights_only` branch, because the safety switch is the point of these deep dives.

**Figure 2: model-structure figure (SVG via `tools/structure_figure.py`).** Reference style is the Keras v3 figure the repo owner provided: dark navy panel, title inside the panel, one rounded container with a green border, gray-bordered boxes inside, red text for the component that can execute code (Keras: the Lambda layer; PyTorch: the pickle `GLOBAL + REDUCE / NEWOBJ / BUILD` box).
- Spec lives in `<framework>/figures/<framework>_structure.json`; keep the SVG committed next to it and a 2x PNG for viewers without SVG support.
- Layout: labelled column = a section box with stacked boxes (the "metadata / program" half); unlabelled column = one tall box (the "weights" half); optional footer row of small boxes for bookkeeping records; optional `container_label` (monospace, top-left) for a path prefix; `caption` is supported but off by default (the PyTorch figure has none, by request).
- The generator prints a warning when text is likely to overflow a box: shorten the text or widen the column rather than ignoring it, and always look at the rendered PNG once.
- Commands:
  `python tools/structure_figure.py <fw>/figures/<fw>_structure.json -o <fw>/figures/<fw>_structure.svg`
  `NODE_PATH=/opt/node22/lib/node_modules node tools/render_png.cjs <fw>/figures/<fw>_structure.svg <fw>/figures/<fw>_structure.png 2`
  (the `NODE_PATH` is for the sandbox this repo was started in, where playwright is installed globally; adjust locally.)

## Comparison seeds (fill one row per framework)

| Framework | Container | Structure/metadata encoding | Weights encoding | Code-execution surface | Default safety | mmap-able |
|---|---|---|---|---|---|---|
| PyTorch `torch.save` | uncompressed ZIP64 with archive-name prefix | pickle protocol 2 (`data.pkl`) | one raw blob per storage (`data/N`), 64-byte aligned | pickle `GLOBAL`/`REDUCE`/`NEWOBJ`/`BUILD`; whole-module saves need class imports | `weights_only=True` since 2.6 (allowlist unpickler) | yes (`mmap=True`, one `mmap` of the file, sliced) |

## Backlog

- Keras v3 `.keras` (zip: `config.json`, `metadata.json`, `model.weights.h5`; `Lambda` layers and `safe_mode`) - the owner already has a reference structure figure for this one.
- safetensors (8-byte header length + JSON header + raw tensor bytes; no code path).
- TensorFlow SavedModel (`saved_model.pb` protobuf graph + `variables/` checkpoint shards + `assets/`).
- ONNX (protobuf `ModelProto`, initializers inline or external data).
- GGUF (llama.cpp: header, KV metadata, tensor infos, aligned data).
- Pickle-based sklearn/joblib artifacts, for contrast with PyTorch's `weights_only` interpreter.

## Environment notes

- Python 3.11; scripts must run with the standard library plus the framework under study.
- PyTorch CPU wheel: `pip install torch --index-url https://download.pytorch.org/whl/cpu` (about 200 MB).
- PNG rendering needs playwright's Chromium; in the original sandbox it was preinstalled (`PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers`). The SVG is the source of truth; the PNG can be regenerated any time.
