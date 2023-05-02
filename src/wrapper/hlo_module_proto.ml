open! Import
open! Base
include Wrappers.HloModuleProto

let read_text_file filename = Stdio.In_channel.read_all filename |> parse_text

let read_proto_file filename ~binary =
  Stdio.In_channel.read_all filename |> parse_proto ~binary
