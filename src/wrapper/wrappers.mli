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

  val create : Element_type.t -> int list -> t
  val clone : t -> t
  val reshape : t -> int list -> t
  val convert : t -> Element_type.t -> t
  val element_type : t -> Element_type.t
  val size_bytes : t -> int
  val element_count : t -> int
  val shape : t -> Shape.t

  (* Bigarray interop. *)
  val of_bigarray : (_, _, Bigarray.c_layout) Bigarray.Genarray.t -> t
  val copy_from_bigarray : t -> src:(_, _, Bigarray.c_layout) Bigarray.Genarray.t -> unit
  val copy_to_bigarray : t -> dst:(_, _, Bigarray.c_layout) Bigarray.Genarray.t -> unit
end

module Op : sig
  type t

  val constant : Literal.t -> builder:Builder.t -> t
  val r0_f32 : float -> builder:Builder.t -> t
  val r0_f64 : float -> builder:Builder.t -> t
  val add : t -> t -> t
end

module Computation : sig
  type t

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
