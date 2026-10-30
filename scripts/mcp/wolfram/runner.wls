(* Per-job native execution. User WL has the privileges of this container. *)
Module[{source, output, seconds, expression, value, timedOut = False, aborted = False,
        status, form, exported, code},
  source = Environment["ASIBENCH_WOLFRAM_INPUT"];
  output = Environment["ASIBENCH_WOLFRAM_OUTPUT"];
  seconds = ToExpression[Environment["ASIBENCH_WOLFRAM_TIMEOUT"]];
  code = Import[source, "Text"];
  If[StringStartsQ[code, "#!"], code = StringReplace[code, StartOfString ~~ "#!" ~~ Shortest[___] ~~ "\n" -> ""]];
  expression = Quiet[ToExpression[code, InputForm, HoldComplete]];
  If[expression === $Failed,
    Export[FileNameJoin[{output, "result.json"}], <|"state" -> "failed", "reason" -> "syntax_error"|>, "RawJSON"]; Exit[2]];
  value = CheckAbort[
    TimeConstrained[ReleaseHold[expression], seconds, timedOut = True; $Aborted],
    aborted = True; $Aborted
  ];
  status = Which[timedOut, "timeout", aborted || value === $Aborted, "aborted",
                 value === $Failed || FailureQ[value], "failed", True, "completed"];
  form = ToString[value, InputForm, PageWidth -> Infinity];
  Export[FileNameJoin[{output, "result.inputform.txt"}], form, "Text"];
  exported = Quiet[Check[Export[FileNameJoin[{output, "result.wxf"}], value, "WXF"], $Failed]];
  Export[FileNameJoin[{output, "result.json"}],
    <|"state" -> status, "inputform_preview" -> StringTake[form, UpTo[16000]],
      "truncated" -> (StringLength[form] > 16000), "result_wxf" -> (exported =!= $Failed),
      "note" -> "Messages and Print output are in stdout.log/stderr.log; unevaluated expressions are preserved, not certified as solved."|>, "RawJSON"];
  Exit[If[status === "completed", 0, 2]]
];
