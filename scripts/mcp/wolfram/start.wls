(* Load the pinned image's official paclet. No client config installation. *)
$Messages = Streams["stderr"];
If[DirectoryQ["/opt/AgentTools"], PacletDirectoryLoad["/opt/AgentTools"]];
Needs["Wolfram`AgentTools`"];
If[!NameQ["Wolfram`AgentTools`StartMCPServer"], Exit[70]];
Wolfram`AgentTools`StartMCPServer[];
