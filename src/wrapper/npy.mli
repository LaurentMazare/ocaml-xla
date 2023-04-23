open! Base

(** [write ?header_len bigarray filename] writes a npy file [filename]
    with the content of [bigarray].
    [header_len] can be used to override the npy header length. This is
    only useful for testing.
*)
val write : ?header_len:int -> Literal.t -> string -> unit

val read_literal : string -> Literal.t
val read_buffer : string -> device:Device.t -> Wrappers.PjRtBuffer.t

module Npz : sig
  type in_file

  val open_in : string -> in_file
  val read_literal : ?suffix:string -> in_file -> string -> Literal.t

  val read_buffer
    :  ?suffix:string
    -> in_file
    -> string
    -> device:Device.t
    -> Wrappers.PjRtBuffer.t

  val entries : in_file -> string list
  val close_in : in_file -> unit

  (** Reads all the tensors in a npz file. *)
  val read_all_literal : string -> (string, Literal.t) Hashtbl.t

  val read_all_buffer
    :  string
    -> device:Device.t
    -> (string, Wrappers.PjRtBuffer.t) Hashtbl.t

  type out_file

  val open_out : string -> out_file
  val write : ?suffix:string -> out_file -> string -> Literal.t -> unit
  val close_out : out_file -> unit
end
