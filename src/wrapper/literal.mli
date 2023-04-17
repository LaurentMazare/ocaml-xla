open! Import
include module type of Wrappers.Literal

val dims : t -> int list

val to_bigarray
  :  t
  -> ('a, 'b) Bigarray.kind
  -> ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t
