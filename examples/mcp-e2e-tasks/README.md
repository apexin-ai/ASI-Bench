# MCP E2E fake tasks

Fake tasks that prove an agent in the normal `asibench run` → `asibench score`
pipeline really calls a given MCP tool and uses its result. They are not
benchmark tasks: they sit outside `tasks/`, have `status: test`, and are only
found with `--tasks-dir examples/mcp-e2e-tasks --include-test`.

There are two separate questions:

- `asibench score`: is the answer right?
- [`scripts/mcp/e2e/verify_run.py`](../../scripts/mcp/e2e/verify_run.py): did
  the answer come from the MCP tool?

An E2E pass needs both. The verifier reads the run artefacts and the task's
`e2e_check.json`:

| Check | Passes when |
|---|---|
| `mcp_connected` | the server connected and offered every required tool (Codex reports neither, so it must show successful calls instead) |
| `tool_called` | every required tool was called |
| `tool_correct` | each required call returned the reference result (value, JSON field or image type) |
| `tool_chain` | a call's input is the output of an earlier call, where configured |
| `answer_from_tool` | each answer in the output file equals a value a tool returned |
| `no_bypass` | the agent did not reach the backend through other routes (imports, installs, web tools) |

## Tasks

| Task | Server | Tools | Reference in `generate_gt.py` |
|---|---|---|---|
| `mcp_e2e.pyscf_rhf_energy` | `pyscf` | `pyscf_rhf_energy` | PySCF RHF |
| `mcp_e2e.pyscf_bond_stretch` | `pyscf` | `run_bond_stretch_calculation_mcp` → `plot_energy_scan_image_mcp` (image) | seeded RDKit/UFF + PySCF |
| `mcp_e2e.arxiv_search_snippets` | `arxiv` | `ArXiv_search_papers` → `ArXiv_get_pdf_snippets` | raw arXiv API + MarkItDown (needs network; generate just before the run, papers can get new PDF versions) |
| `mcp_e2e.jsbsim_engine_run` | `jsbsim` | `create_session` → `set_*` → `step` → `get_property`/`get_telemetry` (one session) | JSBSim driven directly |
| `mcp_e2e.s4_grating_spectrum` | `s4` | `simulate_stack_spectrum` | numpy 1D RCWA |
| `mcp_e2e.psi4_opt_freq` | `psi4` | `optimize` → `frequency` at the returned geometry | PySCF DF-RHF + geomeTRIC |
| `mcp_e2e.rdkit_conformer` | `rdkit` | `smiles_to_mol` → `EmbedMolecule` → `mol_to_sdf` (writes the file), `Max`/`MinPartialCharge` | RDKit ETKDGv3 at the instance's seed + Gasteiger charges |
| `mcp_e2e.build123d_plate_measure` | `build123d` | `execute` → `validate` → `measure` → `find_holes`/`find_hole_patterns` → `export` (STEP+STL) → `import_cad_file` (one session) | closed form: box − bore − corner slivers − bolt circle, extruded (stdlib only) |
| `mcp_e2e.gpaw_mos2_bandgap` | `gpaw` | `fetch_structure` → `relax_structure` → `check_convergence` → `calc_band_dos` → `verify_run` → `get_run_artifacts` (one run) | DFT **measured** against the pinned server by [`measure_gpaw_table.py`](../../scripts/mcp/e2e/measure_gpaw_table.py); k-grid, convergence recommendation and verification verdict recomputed in stdlib |
| `mcp_e2e.ncbi_gene_protein_card` | `ncbi` | `NCBIGene_search` → `NCBIGene_get_summary` (id from the search) + `NCBIProtein_get_summary` (GI from the input file) | raw E-utilities, cross-checked against `gene_table`, protein FASTA and the Datasets API (needs network; the GI freezes the protein facts, the gene's band and assembly accession track the current build, so generate just before the run) |
| `mcp_e2e.qe_si_bandstructure` | `quantum_espresso` | `qe_list_pseudopotentials` + `qe_suggest_kpoints` → `qe_workflow_bandstructure` (suggested grid) → `qe_list_files` (its run directory) → `qe_read_bands` (the listed band file) | DFT **measured** against the pinned server by [`measure_qe_table.py`](../../scripts/mcp/e2e/measure_qe_table.py); k-grid rule, band and point counts derived in stdlib |
| `mcp_e2e.alphafold_isoform_profile` | `alphafold_db` | `alphafold_get_prediction` + `alphafold_get_summary` + `alphafold_get_annotations` (one accession) | raw AlphaFold DB API; every per-residue score checked against the AlphaMissense CSV mean (needs network; the input file lists the entry ids to rank, so a newly added isoform model does not change the answer) |
| `mcp_e2e.atomictoolkit_vacancy` | `atomictoolkit` | `build_structure_workflow` → `manipulate_structure_workflow` (`supercell`, then `vacancy`, each on the file the previous call wrote) → `single_point_workflow` ×2 + `analyze_structure_workflow` + `estimate_elastic_workflow` (chained by the written files' names) | ASE EMT on the same extxyz files (bit-identical to the server over 41 seeds); coordination with the server's 0.3 Å neighbour-list skin; the unrelaxed vacancy formation energy is derived, so only the scorer grades it |
| `mcp_e2e.openroad_tiny_floorplan` | `openroad` | `create_interactive_session` → `interactive_openroad_exec` (`read_lef`, `read_def`, `set_cmd_units`, die box and HPWL in Tcl) + `interactive_openroad_query` (`report_design_area`) → `grep_session_output` (one session) | closed form of our own LEF/DEF (counts, die, whole-um² cell area over the row box, HPWL from pin-shape centres; stdlib only); with the L1 constants the generator writes the smoke's host-validated fixture byte for byte |

How each task picks its instances, scores answers and keeps the tool the only
practical source of the numbers is described in its `generate_gt.py` and
`task_eval.yaml`. Server setup and quirks are in
[`scripts/mcp/e2e/README.md`](../../scripts/mcp/e2e/README.md).

## Results

All runs: AWS Linux amd64, seed 31415, `--sandbox none`. Claude Code used
`claude-opus-5-5`; Codex CLI used `gpt-5.6-sol` at effort `medium`. Each
cell gives local score / verifier PASS. One run per level shows that the path
works; it is not a pass rate.

| Task | Date | Claude Code B1–B4 | Codex CLI B1–B4 |
|---|---|---|---|
| `pyscf_rhf_energy` | 2026-09-29/30 | 3/3 per level (B1 2/3 → 5/5 after the prompt fix below) | 4 × 100, 4/4 |
| `pyscf_bond_stretch` | 2026-09-30 | 12 × 100, 12/12 (3 per level) | 4 × 100, 4/4 |
| `arxiv_search_snippets` | 2026-10-02 | 4 × 100, 4/4 | 4 × 100, 4/4 |
| `jsbsim_engine_run` | 2026-10-02 | 4 × 100, 4/4 | 4 × 100, 4/4 |
| `s4_grating_spectrum` | 2026-10-02 | 4 × 100, 4/4 | 4 × 100, 4/4 |
| `psi4_opt_freq` | 2026-10-02 | 4 × 100, 4/4 | 4 × 100, 4/4 |
| `rdkit_conformer` | 2026-10-03 | 4 × 100, 4/4 | 4 × 100, 4/4 |
| `build123d_plate_measure` | 2026-10-03 | 4 × 100, 4/4 | 396/400 (b1 96), 4/4 |
| `gpaw_mos2_bandgap` | 2026-10-07 | 4 × 100, 4/4 | 4 × 100, 4/4 |
| `ncbi_gene_protein_card` | 2026-10-07 | 4 × 100, 4/4 | 4 × 100, 4/4 |
| `alphafold_isoform_profile` | 2026-10-07 | 4 × 100, 4/4 | 4 × 100, 4/4 |
| `qe_si_bandstructure` | 2026-10-07 | 4 × 100, 4/4 | 4 × 100, 4/4 |
| `atomictoolkit_vacancy` | 2026-10-07 | 4 × 100, 4/4 | 4 × 100, 4/4 |
| `openroad_tiny_floorplan` | 2026-10-07 | 4 × 100, 4/4 | 4 × 100, 4/4 |

In every run at B3/B4 the agent found the tools without being told their names.

The one deduction: Codex's `build123d_plate_measure` b1 reported the bolt *circle*
diameter as `bolt_hole_diameter_mm` (the input file carries both numbers), which costs a
fifth of the feature weight. Every verifier check still passed — the value came from a
tool, it was just the wrong one. The output description now says "one bolt hole … not the
diameter of the circle the holes sit on"; the run above predates that wording.

## Run

Prerequisites: the server is installed and its smoke test passes, and the agent
CLI is logged in for the E2E user. Example for one task; for the others, change
the task and the `--mcp-config` file.

```sh
cd ~/ASI-Bench
T=mcp_e2e.pyscf_rhf_energy; MCP=~/mcp/pyscf.mcp.json; OUT=~/e2e/out-claude

# 1. Instance + reference (seed 31415 is required for local scoring)
uv run asibench generate --task $T --params '{"seed": 31415}' \
  --sandbox task --tasks-dir examples/mcp-e2e-tasks --output-dir ~/e2e/instances

# 2. Agent run (host only: --sandbox os is rejected together with --mcp-config)
uv run asibench run --agent claude_code_cli \
  --agent-config '{"model": "claude-opus-5-5", "permission_mode": "bypassPermissions"}' \
  --mcp-config $MCP --tasks $T --include-test --tasks-dir examples/mcp-e2e-tasks \
  --instances-dir ~/e2e/instances --prompt-levels b1,b2,b3,b4 \
  --sandbox none --timeout 900 --output-dir $OUT

# 3. Score, 4. verify
uv run asibench score --repo seed31415 --results-dir $OUT \
  --instances-dir ~/e2e/instances --tasks-dir examples/mcp-e2e-tasks
python3 scripts/mcp/e2e/verify_run.py --results-dir $OUT \
  --instances-dir ~/e2e/instances --tasks-dir examples/mcp-e2e-tasks
```

For Codex, use `--agent codex_cli --agent-config '{"model": "gpt-5.6-sol",
"effort": "medium", "codex_home": "/abs/path"}'`. With a custom gateway, put
`config.toml` and `auth.json` in a directory of their own and point both
`CODEX_HOME` and `codex_home` at it: the isolated run home copies `config.toml`
from the `codex_home` of `--agent-config` but `auth.json` from the *shell's*
`CODEX_HOME` (else `~/.codex`). Miss either and Codex starts, finds no usable
credentials and finishes without a single model call. A gateway that
authenticates through `OPENAI_API_KEY` does not work either: without an
`api_key` in `--agent-config` the adapter removes that variable from the
agent's environment, so use `auth.json` or name the variable something else in
`config.toml`.

Host runs have no filesystem isolation, so use a dedicated unprivileged user.
The agent can still find the backend library elsewhere on the host; catching
that is what `no_bypass` is for. `--mcp-config` also turns on web search
(Claude: WebSearch/WebFetch, Codex: web_search).

`gpaw_mos2_bandgap` is the slow one: a correct chain is about 430 s of
single-threaded plane-wave DFT (the hard-coded convergence sweep alone is
~60%), so it needs `--timeout 1800`. Its reference is measured rather than
recomputed — GPAW exists only inside the server's conda environment — so after
bumping the server revision or the conda lock, re-run
`scripts/mcp/e2e/measure_gpaw_table.py` and replace the `MEASURED` table in
`generate_gt.py` wholesale.

`qe_si_bandstructure` is measured the same way (`measure_qe_table.py`, about
three minutes), but its band-structure workflow takes seconds, so
`--timeout 1200` is ample. Before a round, run `measure_qe_table.py --check`
on the run host (it also confirms the host indexes the Si pseudopotential the
table was measured with) and clear `~/mcp/quantum_espresso/qe_calculations`.

`atomictoolkit_vacancy` needs `--timeout 900` at most (every tool answers in well
under a second). The server's cwd is its checkout, and relative paths, default
output names and every in-band error's `tool_errors/` log land there; the prompts
ask for absolute paths in the working directory, but clear the known names before a
round (`structure.extxyz*`, `manipulated.extxyz*`, `analysis_outputs/`,
`tool_errors/` in `~/mcp/atomictoolkit`) — never `git clean`, which deletes `.venv`.

`openroad_tiny_floorplan` needs `--timeout 900` at most (each session command
answers in well under a second; all four levels finished quickly). Answers count only if a session command
*printed* them (`openroad_output` extractor): the prompts ask for in-session Tcl
arithmetic, so a value read from the DEF or converted by hand fails
`answer_from_tool`. Tcl `exec` sent through `interactive_openroad_exec` or
`_query` is a `no_bypass` FAIL (upstream allows it, so the session is a shell).

## Adding a task

1. Add the server to `scripts/mcp/e2e/manifest.json` and make its smoke test pass.
2. Copy the closest existing task. Keep `status: test` and seed 31415, and
   compute the reference independently of the server.
3. Write `e2e_check.json` (schema 2: `calls`, `answers`, bypass patterns). The
   format is documented in the `verify_run.py` docstring and checked strictly
   when loaded (a typo fails with `invalid_spec`). A new non-numeric result
   type is a named extractor in `scripts/mcp/e2e/e2e_verify/extractors.py`;
   `json_scalars` already covers a record a REST wrapper keys by the identifier
   that was requested (`data.result.<uid>.slen`), which no static dotted path
   reaches — use it with `match: "member"`.
   Copy from an existing task:

   | Pattern | Task |
   |---|---|
   | single tool, scalar answer | `pyscf_rhf_energy` |
   | tool chain, image result | `pyscf_bond_stretch` |
   | non-numeric extractors, web bypass | `arxiv_search_snippets` |
   | session chain, optional/grouped calls | `jsbsim_engine_run` |
   | answers selected from returned arrays | `s4_grating_spectrum` |
   | nested keys, geometry chain | `psi4_opt_freq` |
   | opaque strings a tool chain passes on | `rdkit_conformer` |
   | path arguments, binary artefacts, closed-form reference | `build123d_plate_measure` |
   | measured reference, forbidden one-call shortcut, listing and status extractors | `gpaw_mos2_bandgap` |
   | records a REST wrapper keys by the requested uid, in-band failures | `ncbi_gene_protein_card` |
   | answers that need a ranking or a count over a returned list (scorer-only) | `alphafold_isoform_profile` |
   | result → argument chains through host paths, measured table with `--check` | `qe_si_bandstructure` |
   | file chains through path arguments the trajectory does not keep (compared by file name), a scorer-only derived answer | `atomictoolkit_vacancy` |
   | facts and numbers read out of an interactive shell's text output, Tcl `exec` through the session as a bypass | `openroad_tiny_floorplan` |

4. Prompts name the server and the tool (e.g. "`pyscf_rhf_energy` of the
   `pyscf` server"). Never use a harness-specific name such as
   `mcp__pyscf__…`.

Why the prompt rule exists: when the B1 prompt used the name
`mcp__pyscf__pyscf_rhf_energy`, 2 of 4 Claude runs decided that the tool was
unavailable. They tried `ToolSearch`, then got "No MCP servers configured" from
`claude mcp list`, and gave up. After the prompt was changed to name only the
server and the tool, B1 passed 5/5.

## Evidence quirks

The verifier handles these; they matter only if you are debugging it.

- **Claude, tool results:** persisted stream-json redacts user events,
  including tool results. The verifier fills results in from
  `*.trajectory.json` by `tool_call_id`.
- **Claude, images:** trajectories never keep image data. The extractors
  record `content_types`/`image_media_types` instead.
- **Claude, structured output:** for FastMCP tools that declare an
  `outputSchema`, Claude Code shows `structuredContent` (`{"result": …}`) and
  not the text. The verifier unwraps it.
- **Both, scrubbed paths:** the persisted stdout replaces host paths with
  placeholders — `<home>`, `<workspace>`, `<run_output_dir>`, `<repo_root>` by
  name and every other absolute path with `<abs_path>` — which would hide
  path-based bypass patterns and break path arguments and returned paths. The
  verifier restores shell commands, tool results **and** path-like tool
  arguments from `*.trajectory.json`, matched by call id; the trajectory keeps
  them because `ai4sci_bench.core.trajectory.KEY_ARG_NAMES` lists the arguments
  an extractor preserves (`filename`, `path`, `file_path`, `output_dir`, `command`, …).
  What it cannot restore is reported as a coverage gap, never as a failure: a
  scrubbed command is a `no_bypass` WARN, and a comparison that only a
  placeholder made fail becomes "not observable in this evidence". An argument
  outside `KEY_ARG_NAMES` (`atomictoolkit`'s `input_filepath`) is only in the
  persisted log, as `<workspace>/<name>`; a link that compares file names
  (`output_file`) still compares it, so a wrong file there is a FAIL, while a bare
  `<abs_path>` (a path outside the workspace and home) stays a coverage gap.
- **Codex:** each call is an `mcp_tool_call` item that keeps its inputs and
  results, but Codex does not list servers or offered tools. Its own
  `list_mcp_resources*` calls are not counted as required tools.
