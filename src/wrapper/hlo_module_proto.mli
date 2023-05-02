open! Import
include module type of Wrappers.HloModuleProto

val read_text_file : string -> t
val read_proto_file : string -> binary:bool -> t
