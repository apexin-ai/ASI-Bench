# Tiny placed design: counts, die, area, utilisation and HPWL

`data/task.json` describes a small placed design, `{{design}}` (LEF + DEF), and the id of the OpenROAD session to use. Characterise it with OpenROAD through the tools available in this session: how many instances, nets and I/O pins it has, the size of its die, how much cell area it holds and how full its core is, and how much wire it needs by the half-perimeter estimate. Afterwards, recover the area report from the session's retained output by searching it (pattern in the task file) rather than re-running the report, close the session, and write `result.json`.

Background: a LEF file describes the technology and the cell library, a DEF file the design itself — die, rows, placed components, pins and nets. Utilisation is the summed cell area over the core area spanned by the rows. The half-perimeter wirelength (HPWL) of a net is the half perimeter of the box around its pins, a standard placement-quality estimate.

Things to know about these tools:

- The `interactive_openroad_query` tool accepts read-only commands only (`report_*`, `get_*`, `check_*`, …) and refuses any command containing a nested `[...]` call. Reading files, `set_*` commands and Tcl that walks the database must go through `interactive_openroad_exec`.
- No Liberty library is loaded, so OpenROAD's distance unit defaults to metres and `report_design_area` would print `0 u^2`. Run `set_cmd_units -distance um` (exec) before reporting the area.
- Give OpenROAD absolute file paths: the session process does not necessarily start in your working directory.

## Output

Write `result.json` in the current working directory:

```json
{
  "design": "{{design}}",
  "session_id": "<the session id every call used>",
  "instance_count": <int>,
  "net_count": <int>,
  "io_pin_count": <int>,
  "die_width_dbu": <int>,
  "die_height_dbu": <int>,
  "design_area_um2": <number>,
  "utilization_percent": <number>,
  "hpwl_total_dbu": <int>,
  "area_report_line": "<the line>"
}
```

- `instance_count`, `net_count`, `io_pin_count`: the components, nets and I/O pins of the design as OpenROAD reads it.
- `die_width_dbu`, `die_height_dbu`: the width and height of the die area in database units (the die starts at the origin).
- `design_area_um2`, `utilization_percent`: the design area in µm² and the core utilisation in percent, as `report_design_area` prints them.
- `hpwl_total_dbu`: the total half-perimeter wirelength in database units. For every net take the centre of the bounding box of each of its pin shapes (instance pins and I/O pins alike); the net's HPWL is (largest x − smallest x) + (largest y − smallest y) over those centres; sum over all nets.
- `area_report_line`: the area report line exactly as `grep_session_output` returns it (its `line` field).

Copy every value exactly as OpenROAD printed it in the session; do not round, convert or recompute it.

## Rules

- OpenROAD is available **only** through the `openroad` MCP server attached to this session. Do not run the `openroad` binary or any other EDA tool (KLayout, …) yourself, do not install or import OpenROAD's Python modules (`odb`, `openroad`), and do not start the MCP server yourself.
- Inside the session, do not use Tcl `exec` (or `open` / `file`): the session is for OpenROAD commands, not a shell.
- You may look at the LEF/DEF files, but **every number in `result.json` must be one that OpenROAD printed in the session**. If a value needs arithmetic, do it in the session with Tcl (`expr`) and print the result; do not compute values outside it, from memory or by hand.
- Pass the same `session_id` to every session call: a call without one silently starts a new, empty session.
- Do not modify anything under `data/`.
