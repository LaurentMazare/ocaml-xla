open! Import
include module type of Wrappers.Literal

val dims : t -> int array

val to_bigarray
  :  t
  -> kind:('a, 'b) Bigarray.kind
  -> ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t
