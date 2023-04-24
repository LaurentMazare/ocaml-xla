open! Import
open! Base

val read_literal : ?only:string list -> string -> (string * Literal.t) list

val read_buffer
  :  ?only:string list
  -> string
  -> device:Device.t
  -> (string * Wrappers.PjRtBuffer.t) list
