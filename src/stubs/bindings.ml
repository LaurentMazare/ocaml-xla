open! Ctypes

module C (F : Cstubs.FOREIGN) = struct
  open! F

  module Status = struct
    type modl
    type struct_ = modl Ctypes.structure
    type t = struct_ ptr

    let struct_ : struct_ typ = structure "_status"
    let t : t typ = ptr struct_
    let error_message = foreign "status_error_message" (t @-> returning string)
    let release = foreign "status_free" (t @-> returning void)
  end

  module Shape = struct
    type modl
    type struct_ = modl Ctypes.structure
    type t = struct_ ptr

    let struct_ : struct_ typ = structure "_shape"
    let t : t typ = ptr struct_
    let dimensions_size = foreign "shape_dimensions_size" (t @-> returning int)
    let element_type = foreign "shape_element_type" (t @-> returning int)
    let dimensions = foreign "shape_dimensions" (t @-> int @-> returning int)
    let release = foreign "shape_free" (t @-> returning void)
  end

  module Builder = struct
    type modl
    type struct_ = modl Ctypes.structure
    type t = struct_ ptr

    let struct_ : struct_ typ = structure "_xla_builder"
    let t : t typ = ptr struct_
    let create = foreign "xla_builder_create" (string @-> returning t)
    let first_error = foreign "first_error" (t @-> returning Status.t)
    let get_current_status = foreign "get_current_status" (t @-> returning Status.t)
    let release = foreign "xla_builder_free" (t @-> returning void)
  end

  module Literal = struct
    type modl
    type struct_ = modl Ctypes.structure
    type t = struct_ ptr

    let struct_ : struct_ typ = structure "_literal"
    let t : t typ = ptr struct_

    let create_from_shape =
      foreign "literal_create_from_shape" (int @-> ptr int64_t @-> size_t @-> returning t)

    let create_from_shape_and_data =
      foreign
        "literal_create_from_shape_and_data"
        (int @-> ptr int64_t @-> size_t @-> ptr void @-> size_t @-> returning t)

    let clone = foreign "literal_clone" (t @-> returning t)

    let reshape =
      foreign
        "literal_reshape"
        (t @-> ptr int64_t @-> size_t @-> ptr t @-> returning Status.t)

    let copy_to = foreign "literal_copy_to" (t @-> ptr void @-> size_t @-> returning void)

    let copy_from =
      foreign "literal_copy_from" (t @-> ptr void @-> size_t @-> returning void)

    let convert = foreign "literal_convert" (t @-> int @-> ptr t @-> returning Status.t)
    let shape = foreign "literal_shape" (t @-> ptr Shape.t @-> returning void)
    let element_type = foreign "literal_element_type" (t @-> returning int)
    let element_count = foreign "literal_element_count" (t @-> returning int64_t)
    let size_bytes = foreign "literal_size_bytes" (t @-> returning int64_t)
    let release = foreign "literal_free" (t @-> returning void)
  end

  module Op = struct
    type modl
    type struct_ = modl Ctypes.structure
    type t = struct_ ptr

    let struct_ : struct_ typ = structure "_xla_op"
    let t : t typ = ptr struct_

    let constant_literal =
      foreign "constant_literal" (Builder.t @-> Literal.t @-> returning t)

    let parameter =
      foreign
        "parameter"
        (Builder.t @-> int64_t @-> int @-> int @-> ptr long @-> string @-> returning t)

    (* Binary functions. *)
    let add = foreign "op_add" (t @-> t @-> returning t)
    let sub = foreign "op_sub" (t @-> t @-> returning t)
    let mul = foreign "op_mul" (t @-> t @-> returning t)
    let div = foreign "op_div" (t @-> t @-> returning t)
    let rem = foreign "op_rem" (t @-> t @-> returning t)
    let max = foreign "op_max" (t @-> t @-> returning t)
    let min = foreign "op_min" (t @-> t @-> returning t)
    let and_ = foreign "op_and" (t @-> t @-> returning t)
    let or_ = foreign "op_or" (t @-> t @-> returning t)
    let xor = foreign "op_xor" (t @-> t @-> returning t)
    let atan2 = foreign "op_atan2" (t @-> t @-> returning t)
    let pow = foreign "op_pow" (t @-> t @-> returning t)
    let dot = foreign "op_dot" (t @-> t @-> returning t)
    let eq = foreign "op_eq" (t @-> t @-> returning t)
    let ne = foreign "op_ne" (t @-> t @-> returning t)
    let ge = foreign "op_ge" (t @-> t @-> returning t)
    let gt = foreign "op_gt" (t @-> t @-> returning t)
    let le = foreign "op_le" (t @-> t @-> returning t)
    let lt = foreign "op_lt" (t @-> t @-> returning t)

    let dot_general =
      foreign
        "op_dot_general"
        (t
         @-> t
         @-> ptr int64_t
         @-> size_t
         @-> ptr int64_t
         @-> size_t
         @-> ptr int64_t
         @-> size_t
         @-> ptr int64_t
         @-> size_t
         @-> returning t)

    (* Unary functions. *)
    let not_ = foreign "op_not" (t @-> returning t)
    let abs = foreign "op_abs" (t @-> returning t)
    let exp = foreign "op_exp" (t @-> returning t)
    let expm1 = foreign "op_expm1" (t @-> returning t)
    let floor = foreign "op_floor" (t @-> returning t)
    let ceil = foreign "op_ceil" (t @-> returning t)
    let round = foreign "op_round" (t @-> returning t)
    let log = foreign "op_log" (t @-> returning t)
    let log1p = foreign "op_log1p" (t @-> returning t)
    let logistic = foreign "op_logistic" (t @-> returning t)
    let sign = foreign "op_sign" (t @-> returning t)
    let clz = foreign "op_clz" (t @-> returning t)
    let cos = foreign "op_cos" (t @-> returning t)
    let sin = foreign "op_sin" (t @-> returning t)
    let tanh = foreign "op_tanh" (t @-> returning t)
    let real = foreign "op_real" (t @-> returning t)
    let imag = foreign "op_imag" (t @-> returning t)
    let sqrt = foreign "op_sqrt" (t @-> returning t)
    let rsqrt = foreign "op_rsqrt" (t @-> returning t)
    let cbrt = foreign "op_cbrt" (t @-> returning t)
    let is_finite = foreign "op_is_finite" (t @-> returning t)
    let neg = foreign "op_neg" (t @-> returning t)
    let lower_triangle = foreign "op_lower_triangle" (t @-> returning t)
    let upper_triangle = foreign "op_upper_triangle" (t @-> returning t)
    let copy = foreign "op_copy" (t @-> returning t)
    let clone = foreign "op_clone" (t @-> returning t)
    let zeros_like = foreign "op_zeros_like" (t @-> returning t)
    let zero_like = foreign "op_zero_like" (t @-> returning t)

    (* Ternary functions. *)
    let clamp = foreign "op_clamp" (t @-> t @-> t @-> returning t)
    let select = foreign "op_select" (t @-> t @-> t @-> returning t)
    let einsum1 = foreign "op_einsum1" (t @-> string @-> returning t)
    let einsum2 = foreign "op_einsum2" (t @-> t @-> string @-> returning t)

    let convert_element_types =
      foreign "op_convert_element_type" (t @-> int @-> returning t)

    let gather =
      foreign
        "op_gather"
        (t
         @-> t
         @-> ptr int64_t
         @-> size_t
         @-> ptr int64_t
         @-> size_t
         @-> ptr int64_t
         @-> size_t
         @-> ptr int64_t
         @-> ptr int64_t
         @-> size_t
         @-> returning t)

    let reshape = foreign "op_reshape" (t @-> size_t @-> ptr int64_t @-> returning t)
    let broadcast = foreign "op_broadcast" (t @-> size_t @-> ptr int64_t @-> returning t)
    let collapse = foreign "op_collapse" (t @-> size_t @-> ptr int64_t @-> returning t)
    let transpose = foreign "op_transpose" (t @-> size_t @-> ptr int64_t @-> returning t)
    let dimensions_size = foreign "op_dimensions_size" (t @-> int64_t @-> returning t)
    let internal_error = foreign "op_internal_error" (Builder.t @-> string @-> returning t)
    let unknown_error = foreign "op_unknown_error" (Builder.t @-> string @-> returning t)

    let invalid_argument_error =
      foreign "op_invalid_argument_error" (Builder.t @-> string @-> returning t)

    let builder = foreign "op_builder" (t @-> returning Builder.t)
    let valid = foreign "xla_op_valid" (t @-> returning int)
    let release = foreign "xla_op_free" (t @-> returning void)

    let get_shape =
      foreign "get_shape" (Builder.t @-> t @-> ptr Shape.t @-> returning Status.t)

    let get_element_type =
      foreign "get_element_type" (Builder.t @-> t @-> ptr int @-> returning Status.t)

    let get_dimensions_size =
      foreign "get_dimensions_size" (Builder.t @-> t @-> ptr int @-> returning Status.t)

    let get_dimensions =
      foreign "get_dimensions" (Builder.t @-> t @-> ptr size_t @-> returning Status.t)

    let r0_f32 = foreign "constant_r0_float" (Builder.t @-> float @-> returning t)
    let r0_f64 = foreign "constant_r0_double" (Builder.t @-> double @-> returning t)
  end

  module Computation = struct
    type modl
    type struct_ = modl Ctypes.structure
    type t = struct_ ptr

    let struct_ : struct_ typ = structure "_xla_computation"
    let t : t typ = ptr struct_
    let name = foreign "xla_computation_name" (t @-> returning string)
    let build = foreign "build" (Builder.t @-> Op.t @-> ptr t @-> returning Status.t)
    let release = foreign "xla_computation_free" (t @-> returning void)
  end

  module PjRtDevice = struct
    type modl
    type struct_ = modl Ctypes.structure
    type t = struct_ ptr

    let struct_ : struct_ typ = structure "_pjrt_device"
    let t : t typ = ptr struct_
    let id = foreign "pjrt_device_id" (t @-> returning int)
    let process_index = foreign "pjrt_device_process_index" (t @-> returning int)
    let local_hardware_id = foreign "pjrt_device_local_hardware_id" (t @-> returning int)
    let kind = foreign "pjrt_device_kind" (t @-> returning string)
    let debug_string = foreign "pjrt_device_debug_string" (t @-> returning string)
    let to_string = foreign "pjrt_device_to_string" (t @-> returning string)
    (* No release here as the object is always owned by the client. *)
  end

  module PjRtClient = struct
    type modl
    type struct_ = modl Ctypes.structure
    type t = struct_ ptr

    let struct_ : struct_ typ = structure "_pjrt_client"
    let t : t typ = ptr struct_
    let cpu = foreign "pjrt_cpu_client_create" (ptr t @-> returning Status.t)

    let gpu =
      foreign "pjrt_gpu_client_create" (ptr t @-> double @-> bool @-> returning Status.t)

    let device_count = foreign "pjrt_client_device_count" (t @-> returning int)

    let addressable_device_count =
      foreign "pjrt_client_addressable_device_count" (t @-> returning int)

    let devices = foreign "pjrt_client_devices" (t @-> ptr PjRtDevice.t @-> returning void)

    let addressable_devices =
      foreign "pjrt_client_addressable_devices" (t @-> ptr PjRtDevice.t @-> returning void)

    let platform_name = foreign "pjrt_client_platform_name" (t @-> returning string)
    let platform_version = foreign "pjrt_client_platform_version" (t @-> returning string)
    let release = foreign "pjrt_client_free" (t @-> returning void)
  end

  module PjRtBuffer = struct
    type modl
    type struct_ = modl Ctypes.structure
    type t = struct_ ptr

    let struct_ : struct_ typ = structure "_pjrt_buffer"
    let t : t typ = ptr struct_

    let from_host_literal =
      foreign
        "pjrt_buffer_from_host_literal"
        (PjRtClient.t @-> PjRtDevice.t @-> Literal.t @-> ptr t @-> returning Status.t)

    let from_host_buffer =
      foreign
        "pjrt_buffer_from_host_buffer"
        (PjRtClient.t
         @-> PjRtDevice.t
         @-> ptr void
         @-> int
         @-> int
         @-> ptr int64_t
         @-> ptr t
         @-> returning Status.t)

    let to_literal_sync =
      foreign "pjrt_buffer_to_literal_sync" (t @-> ptr Literal.t @-> returning Status.t)

    let copy_raw_to_host_sync =
      foreign
        "pjrt_buffer_copy_raw_to_host_sync"
        (t @-> ptr void @-> size_t @-> size_t @-> returning Status.t)

    let on_device_shape = foreign "pjrt_buffer_on_device_shape" (t @-> returning Shape.t)

    let copy_to_device =
      foreign
        "pjrt_buffer_copy_to_device"
        (t @-> PjRtDevice.t @-> ptr t @-> returning Status.t)

    let release = foreign "pjrt_buffer_free" (t @-> returning void)
  end

  module PjRtLoadedExecutable = struct
    type modl
    type struct_ = modl Ctypes.structure
    type t = struct_ ptr

    let struct_ : struct_ typ = structure "_pjrt_loaded_executable"
    let t : t typ = ptr struct_

    let compile =
      foreign "compile" (PjRtClient.t @-> Computation.t @-> ptr t @-> returning Status.t)

    let execute =
      foreign
        "execute"
        (t
         @-> ptr Literal.t
         @-> int
         @-> ptr (ptr (ptr PjRtBuffer.t))
         @-> returning Status.t)

    let release = foreign "pjrt_loaded_executable_free" (t @-> returning void)
  end
end
