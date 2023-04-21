open! Base

val bigarray_to_array0 : ('a, 'b, 'c) Bigarray.Genarray.t -> 'a option
val bigarray_to_array1 : ('a, 'b, 'c) Bigarray.Genarray.t -> 'a array option
val bigarray_to_array2 : ('a, 'b, 'c) Bigarray.Genarray.t -> 'a array array option
val bigarray_to_array3 : ('a, 'b, 'c) Bigarray.Genarray.t -> 'a array array array option
val bigarray_to_array0_exn : ('a, 'b, 'c) Bigarray.Genarray.t -> 'a
val bigarray_to_array1_exn : ('a, 'b, 'c) Bigarray.Genarray.t -> 'a array
val bigarray_to_array2_exn : ('a, 'b, 'c) Bigarray.Genarray.t -> 'a array array
val bigarray_to_array3_exn : ('a, 'b, 'c) Bigarray.Genarray.t -> 'a array array array
