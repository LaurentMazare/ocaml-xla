(* A low-level but hopefully type safe version of the API. *)
open! Base

module Shape : sig
  type t

  val dimensions : t -> int list
  val element_type : t -> Element_type.t
end

module Builder : sig
  type t

  val create : name:string -> t
  val first_error : t -> unit Or_error.t
  val current_status : t -> unit Or_error.t
end

module Literal : sig
  type t

  val create : element_type:Element_type.t -> dims:int list -> t
  val clone : t -> t
  val reshape : t -> dims:int list -> t
  val convert : t -> element_type:Element_type.t -> t
  val element_type : t -> Element_type.t
  val size_bytes : t -> int
  val element_count : t -> int
  val shape : t -> Shape.t

  (* Bigarray interop. *)
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
  val element_type : t -> Element_type.t
  val dims : t -> int list
  val constant : Literal.t -> builder:Builder.t -> t
  val normalize_index : rank:int -> dim_index:int -> int

  val parameter
    :  string
    -> id:int
    -> element_type:Element_type.t
    -> dims:int list
    -> builder:Builder.t
    -> t

  val r0_i32 : int -> builder:Builder.t -> t
  val r0_i64 : int -> builder:Builder.t -> t
  val r0_u32 : int -> builder:Builder.t -> t
  val r0_u64 : int -> builder:Builder.t -> t
  val r0_f32 : float -> builder:Builder.t -> t
  val r0_f64 : float -> builder:Builder.t -> t
  val min_value : element_type:Element_type.t -> builder:Builder.t -> t
  val max_value : element_type:Element_type.t -> builder:Builder.t -> t

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
  val reshape : t -> dims:int list -> t
  val broadcast : t -> dims:int list -> t
  val collapse : t -> dim_indexes:int list -> t
  val transpose : t -> dim_indexes:int list -> t
  val swap_dims : t -> dim_index1:int -> dim_index2:int -> t
  val convert : t -> element_type:Element_type.t -> t
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
    -> lhs_c:int list
    -> rhs_c:int list
    -> lhs_b:int list
    -> rhs_b:int list
    -> t

  val einsum2 : t -> t -> string -> t

  (* Ternary *)
  val clamp : t -> min:t -> max:t -> t
  val select : mask:t -> on_true:t -> on_false:t -> t

  val gather
    :  t
    -> start_indices:t
    -> offset_dims:int list
    -> collapsed_slice_dims:int list
    -> start_index_map:int list
    -> set_index_vector_dim:int option
    -> slice_sizes:int list
    -> t

  val slice_in_dim
    :  ?stride:int
    -> ?start_index:int
    -> t
    -> stop_index:int
    -> dim:int
    -> t

  val reduce : t -> init:t -> f:computation -> dims:int list -> keep_dims:bool -> t
end

module Computation : sig
  type t = Op.computation

  val name : t -> string
  val build : root:Op.t -> t
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
end

module PjRtLoadedExecutable : sig
  type t

  val compile : PjRtClient.t -> Computation.t -> t
  val execute : t -> Literal.t list -> PjRtBuffer.t array array
end
