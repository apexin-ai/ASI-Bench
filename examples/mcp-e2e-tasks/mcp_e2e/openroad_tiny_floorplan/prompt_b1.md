# Tiny placed design: counts, die, area, utilisation and HPWL via the OpenROAD MCP server

`data/task.json` names the design `{{design}}`: a technology/library LEF (`{{lef_file}}`) and a placed DEF (`{{def_file}}`) — a few inverters and buffers in standard-cell rows, wired from an input pin to an output pin. Read it into an interactive OpenROAD session of the `openroad` MCP server and report what OpenROAD says about it.

## Steps

1. Read `data/task.json` and run `pwd`; below, `<workdir>` is that absolute path.
2. Call `create_interactive_session` of the `openroad` MCP server with `session_id` = `{{session_id}}`. Pass `session_id` = `{{session_id}}` to **every** later session call.
3. Call `interactive_openroad_exec` with `command` = `read_lef <workdir>/{{lef_file}}`. OpenROAD answers `... created 1 layers, 2 library cells`.
4. Call `interactive_openroad_exec` with `command` = `read_def <workdir>/{{def_file}}`. Take `io_pin_count` from `Created N pins.`, `instance_count` from `Created N components and ...` and `net_count` from `Created N nets and ...`.
5. Call `interactive_openroad_exec` with `command` = `set_cmd_units -distance um` (without it the area below is reported as 0, because no Liberty library sets the units).
6. Call `interactive_openroad_query` with `command` = `report_design_area`. It prints `Design area A u^2 U% utilization.`; take `design_area_um2` = A and `utilization_percent` = U.
7. Call `interactive_openroad_exec` with this `command` (one line), which prints the die box in database units as `DIE xMin yMin xMax yMax`; the die starts at the origin, so `die_width_dbu` = xMax and `die_height_dbu` = yMax:

   ```tcl
   puts "DIE [[[ord::get_db_block] getDieArea] xMin] [[[ord::get_db_block] getDieArea] yMin] [[[ord::get_db_block] getDieArea] xMax] [[[ord::get_db_block] getDieArea] yMax]"
   ```

8. Call `interactive_openroad_exec` with this `command` (one line), which defines a Tcl procedure computing the total HPWL from the pin-shape centres in the database:

   ```tcl
   proc or_hpwl {} { set total 0; foreach net [[ord::get_db_block] getNets] { set xs {}; set ys {}; foreach it [$net getITerms] { set b [$it getBBox]; lappend xs [expr {([$b xMin] + [$b xMax]) / 2}]; lappend ys [expr {([$b yMin] + [$b yMax]) / 2}] }; foreach bt [$net getBTerms] { set b [$bt getBBox]; lappend xs [expr {([$b xMin] + [$b xMax]) / 2}]; lappend ys [expr {([$b yMin] + [$b yMax]) / 2}] }; set xs [lsort -integer $xs]; set ys [lsort -integer $ys]; incr total [expr {[lindex $xs end] - [lindex $xs 0] + [lindex $ys end] - [lindex $ys 0]}] }; return $total }
   ```

9. Call `interactive_openroad_exec` with `command` = `puts "HPWL_TOTAL [or_hpwl]"` and take `hpwl_total_dbu` from the printed `HPWL_TOTAL` line.
10. Call `grep_session_output` with `session_id` = `{{session_id}}` and `pattern` = `{{grep_pattern}}`. Take `area_report_line` from the `line` field of the match (the line step 6 printed).
11. Call `terminate_interactive_session` with `session_id` = `{{session_id}}`.
12. Write `result.json` as described below.

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
