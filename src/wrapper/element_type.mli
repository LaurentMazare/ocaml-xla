type t =
  | Invalid
  | Pred
  | S8
  | S16
  | S32
  | S64
  | U8
  | U16
  | U32
  | U64
  | Bf16
  | F16
  | F32
  | F64
  | C64
  | C128
  | Tuple
  | OpaqueType
  | Token
[@@deriving sexp, compare, equal]

val of_c_int : int -> t
val to_c_int : t -> int
val to_string : t -> string
val size_in_bytes : t -> int option
val check_exn : t -> (_, _) Bigarray.kind -> unit

type ba_kind = P : (_, _) Bigarray.kind -> ba_kind

val ba_kind : t -> ba_kind option
val is_tensor : t -> bool
val is_tuple : t -> bool
