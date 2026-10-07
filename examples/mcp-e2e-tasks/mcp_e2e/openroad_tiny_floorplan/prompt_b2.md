# Tiny placed design: counts, die, area, utilisation and HPWL via the OpenROAD MCP server

`data/task.json` names the design `{{design}}`: a technology/library LEF (`{{lef_file}}`) and a placed DEF (`{{def_file}}`) — a few inverters and buffers in standard-cell rows, wired from an input pin to an output pin. Read it into an interactive OpenROAD session of the `openroad` MCP server and report what OpenROAD says about it.

## Steps

1. Read `data/task.json`. Open a session with `create_interactive_session` (`session_id` = `{{session_id}}`) and use that session for everything below.
2. With `interactive_openroad_exec`, read the LEF and then the DEF (`read_lef`, `read_def`, absolute paths). The DEF reader's INFO lines give the I/O pin, component and net counts.
3. With `interactive_openroad_exec`, set the distance unit to microns (`set_cmd_units -distance um`; no Liberty library is loaded, so otherwise areas are reported in m² and print as 0). Then run `report_design_area` with `interactive_openroad_query` for the design area and utilisation.
4. With `interactive_openroad_exec`, print the die box from the database: `[ord::get_db_block]` is the block, its `getDieArea` returns a rectangle with `xMin` / `yMin` / `xMax` / `yMax` in database units.
5. With `interactive_openroad_exec`, compute and print the total HPWL in Tcl from the database: the block's `getNets`; per net its instance terminals (`getITerms`) and I/O terminals (`getBTerms`), each with `getBBox` (a rectangle as above). Use the bounding-box centres, integer database units. A procedure can be defined in one exec call and called in the next.
6. With `grep_session_output` (`pattern` = `{{grep_pattern}}`), find the area report line in the session's retained output.
7. Terminate the session with `terminate_interactive_session`, then write `result.json`.

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
