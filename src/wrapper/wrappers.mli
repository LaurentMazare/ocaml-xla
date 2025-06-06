(* A low-level but hopefully type safe version of the API. *)
open! Base

module Shape : sig
  type t

  val rank : t -> int
  val tuple_shapes_size : t -> int
  val dimensions : t -> int array
  val ty : t -> Element_type.t
end

module Builder : sig
  type t

  val create : name:string -> t
  val first_error : t -> unit Or_error.t
  val current_status : t -> unit Or_error.t
end

module Literal : sig
  type t

  val create : ty:Element_type.t -> dims:int array -> t
  val clone : t -> t
  val reshape : t -> dims:int array -> t
  val convert : t -> ty:Element_type.t -> t
  val ty : t -> Element_type.t
  val size_bytes : t -> int
  val element_count : t -> int
  val shape : t -> Shape.t

  (* This consumes the input literal t *)
  val decompose_tuple : t -> t array

  (* Bigarray interop. *)
  val of_bigarray_bytes
    :  src:(int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Genarray.t
    -> ty:Element_type.t
    -> dims:int array
    -> t

  val of_bigarray : (_, _, Bigarray.c_layout) Bigarray.Genarray.t -> t
  val copy_from_bigarray : t -> src:(_, _, Bigarray.c_layout) Bigarray.Genarray.t -> unit
  val copy_to_bigarray : t -> dst:(_, _, Bigarray.c_layout) Bigarray.Genarray.t -> unit
  val r0_i32 : int -> t
  val r0_i64 : int -> t
  val r0_u32 : int -> t
  val r0_u64 : int -> t
  val r0_f32 : float -> t
  val r0_f64 : float -> t
end

module Op : sig
  type t
  type computation

  val builder : t -> Builder.t
  val rank : t -> int
  val ty : t -> Element_type.t
  val dims : t -> int array
  val constant : Literal.t -> builder:Builder.t -> t
  val normalize_index : rank:int -> dim_index:int -> int

  val parameter
    :  string
    -> id:int
    -> ty:Element_type.t
    -> dims:int array
    -> builder:Builder.t
    -> t

  val r0_i32 : int -> builder:Builder.t -> t
  val r0_i64 : int -> builder:Builder.t -> t
  val r0_u32 : int -> builder:Builder.t -> t
  val r0_u64 : int -> builder:Builder.t -> t
  val r0_f32 : float -> builder:Builder.t -> t
  val r0_f64 : float -> builder:Builder.t -> t
  val r1_i32 : int array -> builder:Builder.t -> t
  val r1_i64 : int array -> builder:Builder.t -> t
  val r1_u32 : int array -> builder:Builder.t -> t
  val r1_u64 : int array -> builder:Builder.t -> t
  val r1_f32 : float array -> builder:Builder.t -> t
  val r1_f64 : float array -> builder:Builder.t -> t
  val min_value : ty:Element_type.t -> builder:Builder.t -> t
  val max_value : ty:Element_type.t -> builder:Builder.t -> t
  val iota1 : ty:Element_type.t -> size:int -> builder:Builder.t -> t

  val iota
    :  ty:Element_type.t
    -> dims:int array
    -> iota_dimension:int
    -> builder:Builder.t
    -> t

  val tuple : t list -> builder:Builder.t -> t

  (* Unary. *)
  val not_ : t -> t
  val abs : t -> t
  val exp : t -> t
  val expm1 : t -> t
  val floor : t -> t
  val ceil : t -> t
  val round : t -> t
  val log : t -> t
  val log1p : t -> t
  val logistic : t -> t
  val sign : t -> t
  val clz : t -> t
  val cos : t -> t
  val sin : t -> t
  val tanh : t -> t
  val real : t -> t
  val imag : t -> t
  val sqrt : t -> t
  val rsqrt : t -> t
  val cbrt : t -> t
  val is_finite : t -> t
  val neg : t -> t
  val lower_triangle : t -> t
  val upper_triangle : t -> t
  val copy : t -> t
  val clone : t -> t
  val zeros_like : t -> t
  val zero_like : t -> t
  val einsum1 : t -> string -> t
  val reshape : t -> dims:int array -> t
  val broadcast : t -> dims:int array -> t
  val broadcast_in_dim : t -> out_dims:int array -> broadcast_dims:int array -> t
  val collapse : t -> dim_indexes:int array -> t
  val transpose : t -> dim_indexes:int array -> t
  val swap_dims : t -> dim_index1:int -> dim_index2:int -> t
  val convert : t -> ty:Element_type.t -> t
  val dimensions_size : t -> dim_index:int -> t

  (* Binary. *)
  val add : t -> t -> t
  val sub : t -> t -> t
  val mul : t -> t -> t
  val div : t -> t -> t
  val rem : t -> t -> t
  val max : t -> t -> t
  val min : t -> t -> t
  val and_ : t -> t -> t
  val or_ : t -> t -> t
  val xor : t -> t -> t
  val atan2 : t -> t -> t
  val pow : t -> t -> t
  val dot : t -> t -> t
  val eq : t -> t -> t
  val ne : t -> t -> t
  val ge : t -> t -> t
  val gt : t -> t -> t
  val le : t -> t -> t
  val lt : t -> t -> t

  val dot_general
    :  t
    -> t
    -> lhs_c:int array
    -> rhs_c:int array
    -> lhs_b:int array
    -> rhs_b:int array
    -> t

  val einsum2 : t -> t -> string -> t

  (* Ternary *)
  val clamp : t -> min:t -> max:t -> t
  val select : mask:t -> on_true:t -> on_false:t -> t

  val gather
    :  t
    -> start_indices:t
    -> offset_dims:int array
    -> collapsed_slice_dims:int array
    -> start_index_map:int array
    -> set_index_vector_dim:int option
    -> slice_sizes:int array
    -> t

  val slice_in_dim
    :  ?stride:int
    -> ?start_index:int
    -> t
    -> stop_index:int
    -> dim_index:int
    -> t

  val concat_in_dim : t -> t list -> dim_index:int -> t
  val reduce : t -> init:t -> f:computation -> dims:int array -> keep_dims:bool -> t
end

module HloModuleProto : sig
  type t

  val computation : t -> Op.computation
  val to_string : t -> string
  val parse_proto : string -> binary:bool -> t
  val parse_text : string -> t
end

module Computation : sig
  type t = Op.computation

  val name : t -> string
  val build : root:Op.t -> t
  val proto : t -> HloModuleProto.t
end

module PjRtDevice : sig
  type t

  val id : t -> int
  val process_index : t -> int
  val local_hardware_id : t -> int
  val kind : t -> string
  val debug_string : t -> string
  val to_string : t -> string
end

module PjRtClient : sig
  type t

  val cpu : unit -> t
  val gpu : memory_fraction:float -> preallocate:bool -> t
  val tpu : max_inflight_computations:int -> t
  val device_count : t -> int
  val devices : t -> PjRtDevice.t list
  val addressable_device_count : t -> int
  val addressable_devices : t -> PjRtDevice.t list
  val platform_name : t -> string
  val platform_version : t -> string
end

module PjRtBuffer : sig
  type t

  val of_host_literal : Literal.t -> device:PjRtDevice.t -> t
  val on_device_shape : t -> Shape.t
  val to_literal_sync : t -> Literal.t
  val copy_to_device : t -> device:PjRtDevice.t -> t

  val of_bigarray_bytes
    :  src:(int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Genarray.t
    -> ty:Element_type.t
    -> dims:int array
    -> device:PjRtDevice.t
    -> t

  val of_bigarray
    :  (_, _, Bigarray.c_layout) Bigarray.Genarray.t
    -> device:PjRtDevice.t
    -> t
end

module PjRtLoadedExecutable : sig
  type t

  val compile : PjRtClient.t -> Computation.t -> t
  val execute : t -> Literal.t array -> PjRtBuffer.t array array
  val execute_b : t -> PjRtBuffer.t array -> PjRtBuffer.t array array
end
