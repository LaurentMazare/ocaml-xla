open! Base
open! Import
module CArray = Ctypes.CArray

let _add_compact =
  match Sys.getenv "OCAML_XLA_ADD_COMPACT" with
  | None | Some "false" | Some "0" -> false
  | Some _ -> true

module Status = struct
  let check_and_release status =
    if Ctypes.is_null status
    then Ok ()
    else (
      let error_message = W.Status.error_message status in
      W.Status.release status;
      Or_error.error_string error_message)

  let ok_exn status = check_and_release status |> Or_error.ok_exn
end

module Shape = struct
  type t = W.Shape.t

  (* This takes ownership of the pointer. *)
  let of_ptr ptr =
    if Ctypes.is_null ptr then failwith "null shape pointer";
    Caml.Gc.finalise W.Shape.release ptr;
    ptr

  let dimensions t =
    let dsize = W.Shape.dimensions_size t in
    List.init dsize ~f:(W.Shape.dimensions t)

  let element_type t = W.Shape.element_type t |> Element_type.of_c_int
end

module Builder = struct
  type t = W.Builder.t

  let create ~name =
    let ptr = W.Builder.create name in
    if Ctypes.is_null ptr then failwith "null builder pointer";
    Caml.Gc.finalise W.Builder.release ptr;
    ptr

  let first_error t = W.Builder.first_error t |> Status.check_and_release
  let current_status t = W.Builder.get_current_status t |> Status.check_and_release
end

module Literal = struct
  type t = W.Literal.t

  let of_ptr ptr =
    if Ctypes.is_null ptr then failwith "null literal pointer";
    Caml.Gc.finalise W.Literal.release ptr;
    ptr

  let create element_type dims =
    let dims = List.map dims ~f:Int64.of_int |> CArray.of_list Ctypes.int64_t in
    let t =
      W.Literal.create_from_shape
        (Element_type.to_c_int element_type)
        (CArray.start dims)
        (CArray.length dims |> Unsigned.Size_t.of_int)
    in
    keep_alive dims;
    of_ptr t

  let clone t = W.Literal.clone t |> of_ptr

  let reshape t dims =
    let dims = List.map dims ~f:Int64.of_int |> CArray.of_list Ctypes.int64_t in
    let ptr = Ctypes.(allocate_n (ptr W.Literal.struct_) ~count:1) in
    let status =
      W.Literal.reshape
        t
        (CArray.start dims)
        (CArray.length dims |> Unsigned.Size_t.of_int)
        ptr
    in
    keep_alive dims;
    Status.ok_exn status;
    Ctypes.( !@ ) ptr |> of_ptr

  let convert t element_type =
    let ptr = Ctypes.(allocate_n (ptr W.Literal.struct_) ~count:1) in
    let status = W.Literal.convert t (Element_type.to_c_int element_type) ptr in
    Status.ok_exn status;
    Ctypes.( !@ ) ptr |> of_ptr

  let element_type t = W.Literal.element_type t |> Element_type.of_c_int
  let size_bytes t = W.Literal.size_bytes t |> Int64.to_int_exn
  let element_count t = W.Literal.element_count t |> Int64.to_int_exn

  let shape t =
    let ptr = Ctypes.(allocate_n (ptr W.Shape.struct_) ~count:1) in
    W.Literal.shape t ptr;
    Ctypes.( !@ ) ptr |> Shape.of_ptr
end

module Op = struct
  type t = W.Op.t

  let of_ptr ptr =
    if Ctypes.is_null ptr then failwith "null op pointer";
    Caml.Gc.finalise W.Op.release ptr;
    ptr

  let constant builder literal = W.Op.constant_literal builder literal |> of_ptr
  let add t1 t2 = W.Op.add t1 t2 |> of_ptr
end

module Computation = struct
  type t = W.Computation.t

  let of_ptr ptr =
    if Ctypes.is_null ptr then failwith "null computation pointer";
    Caml.Gc.finalise W.Computation.release ptr;
    ptr

  let name = W.Computation.name

  let build builder ~root =
    let ptr = Ctypes.(allocate_n (ptr W.Computation.struct_) ~count:1) in
    let status = W.Computation.build builder root ptr in
    Status.ok_exn status;
    Ctypes.( !@ ) ptr |> of_ptr
end

module PjRtDevice = struct
  type t = W.PjRtDevice.t

  let id = W.PjRtDevice.id
  let process_index = W.PjRtDevice.process_index
  let local_hardware_id = W.PjRtDevice.local_hardware_id
  let kind = W.PjRtDevice.kind
  let debug_string = W.PjRtDevice.debug_string
  let to_string = W.PjRtDevice.to_string
end

module PjRtClient = struct
  type t = W.PjRtClient.t

  let of_ptr ptr =
    if Ctypes.is_null ptr then failwith "null client pointer";
    Caml.Gc.finalise W.PjRtClient.release ptr;
    ptr

  let cpu () =
    let ptr = Ctypes.(allocate_n (ptr W.PjRtClient.struct_) ~count:1) in
    let status = W.PjRtClient.cpu ptr in
    Status.ok_exn status;
    Ctypes.( !@ ) ptr |> of_ptr

  let gpu ~memory_fraction ~preallocate =
    let ptr = Ctypes.(allocate_n (ptr W.PjRtClient.struct_) ~count:1) in
    let status = W.PjRtClient.gpu ptr memory_fraction preallocate in
    Status.ok_exn status;
    Ctypes.( !@ ) ptr |> of_ptr

  let device_count = W.PjRtClient.device_count
  let addressable_device_count = W.PjRtClient.addressable_device_count
  let platform_name = W.PjRtClient.platform_name
  let platform_version = W.PjRtClient.platform_version
end

module PjRtBuffer = struct
  type t = W.PjRtBuffer.t
end

module PjRtLoadedExecutable = struct
  type t = W.PjRtLoadedExecutable.t

  let of_ptr ptr =
    if Ctypes.is_null ptr then failwith "null PjRtLoadedExecutable pointer";
    Caml.Gc.finalise W.PjRtLoadedExecutable.release ptr;
    ptr

  let compile client computation =
    let ptr = Ctypes.(allocate_n (ptr W.PjRtLoadedExecutable.struct_) ~count:1) in
    let status = W.PjRtLoadedExecutable.compile client computation ptr in
    Status.ok_exn status;
    Ctypes.( !@ ) ptr |> of_ptr
end
