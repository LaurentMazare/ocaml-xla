open! Base
open! Import
module CArray = Ctypes.CArray

let carray_i64 v = List.map v ~f:Int64.of_int |> CArray.of_list Ctypes.int64_t

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
    if Ctypes.is_null ptr then failwith "null Shape pointer";
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
    if Ctypes.is_null ptr then failwith "null Builder pointer";
    Caml.Gc.finalise W.Builder.release ptr;
    ptr

  let first_error t = W.Builder.first_error t |> Status.check_and_release
  let current_status t = W.Builder.get_current_status t |> Status.check_and_release
end

module Literal = struct
  type t = W.Literal.t

  let of_ptr ptr =
    if Ctypes.is_null ptr then failwith "null Literal pointer";
    Caml.Gc.finalise W.Literal.release ptr;
    ptr

  let create ~element_type ~dims =
    let dims = carray_i64 dims in
    let t =
      W.Literal.create_from_shape
        (Element_type.to_c_int element_type)
        (CArray.start dims)
        (CArray.length dims |> Unsigned.Size_t.of_int)
    in
    keep_alive dims;
    of_ptr t

  let clone t = W.Literal.clone t |> of_ptr

  let reshape t ~dims =
    let dims = carray_i64 dims in
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

  let convert t ~element_type =
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

  let check_same_dims_and_kind (type a b) t (ba : (a, b, _) Bigarray.Genarray.t) =
    (match Bigarray.Genarray.layout ba with
     | C_layout -> ()
     | _ -> .);
    let shape = shape t in
    let dims = Shape.dimensions shape |> Array.of_list in
    let ba_dims = Bigarray.Genarray.dims ba in
    if not (Array.equal Int.equal ba_dims dims)
    then
      failwith_s [%message "dims do not match" (ba_dims : int array) (dims : int array)];
    Element_type.check_exn (Shape.element_type shape) (Bigarray.Genarray.kind ba)

  let copy_from_bigarray t ~src =
    check_same_dims_and_kind t src;
    let size_in_bytes = Bigarray.Genarray.size_in_bytes src in
    W.Literal.copy_from
      t
      (Ctypes.bigarray_start Ctypes.genarray src |> Ctypes.to_voidp)
      (Unsigned.Size_t.of_int size_in_bytes);
    keep_alive src

  let copy_to_bigarray t ~dst =
    check_same_dims_and_kind t dst;
    let size_in_bytes = Bigarray.Genarray.size_in_bytes dst in
    W.Literal.copy_to
      t
      (Ctypes.bigarray_start Ctypes.genarray dst |> Ctypes.to_voidp)
      (Unsigned.Size_t.of_int size_in_bytes);
    keep_alive dst

  let of_bigarray (type a b) (src : (a, b, Bigarray.c_layout) Bigarray.Genarray.t) =
    let element_type : Element_type.t =
      match Bigarray.Genarray.kind src with
      | Char | Int8_unsigned -> U8
      | Int16_unsigned -> U16
      | Int8_signed -> S8
      | Int16_signed -> S16
      | Int32 -> S32
      | Int64 -> S64
      | Float32 -> F32
      | Float64 -> F64
      | _ba_kind -> failwith_s [%message "unsupported bigarray type"]
    in
    let dims = Bigarray.Genarray.dims src |> Array.to_list |> carray_i64 in
    let t =
      W.Literal.create_from_shape_and_data
        (Element_type.to_c_int element_type)
        (CArray.start dims)
        (CArray.length dims |> Unsigned.Size_t.of_int)
        (Ctypes.bigarray_start Ctypes.genarray src |> Ctypes.to_voidp)
        (Bigarray.Genarray.size_in_bytes src |> Unsigned.Size_t.of_int)
    in
    keep_alive src;
    keep_alive dims;
    t

  let r0_i32 v = W.Literal.r0_i32 (Int32.of_int_exn v)
  let r0_i64 v = W.Literal.r0_i64 (Int64.of_int_exn v)
  let r0_u32 v = W.Literal.r0_u32 (Unsigned.UInt32.of_int v)
  let r0_u64 v = W.Literal.r0_u64 (Unsigned.UInt64.of_int v)
  let r0_f32 = W.Literal.r0_f32
  let r0_f64 = W.Literal.r0_f64
end

module Op = struct
  type t =
    { ptr : W.Op.t
    ; builder : Builder.t
    }

  type computation = W.Computation.t

  let of_ptr ptr ~builder =
    if Ctypes.is_null ptr then failwith "null Op pointer";
    Caml.Gc.finalise W.Op.release ptr;
    Builder.current_status builder |> Or_error.ok_exn;
    { ptr; builder }

  let rank t =
    let ptr = Ctypes.(allocate_n int ~count:1) in
    let status = W.Op.get_dimensions_size t.builder t.ptr ptr in
    Status.ok_exn status;
    Ctypes.( !@ ) ptr

  let normalize_index ~rank ~dim_index =
    if 0 <= dim_index && dim_index < rank
    then dim_index
    else if dim_index + rank >= 0
    then dim_index + rank
    else [%message "ind out of bounds" (dim_index : int) (rank : int)] |> failwith_s

  let normalize_indexes t ~dim_indexes =
    let rank = rank t in
    List.map dim_indexes ~f:(fun dim_index ->
      normalize_index ~rank ~dim_index |> Int64.of_int_exn)

  let dimensions_size t ~dim_index =
    let rank = rank t in
    let dim_index = normalize_index ~rank ~dim_index in
    W.Op.dimensions_size t.ptr (Int64.of_int_exn dim_index) |> of_ptr ~builder:t.builder

  let element_type t =
    let ptr = Ctypes.(allocate_n int ~count:1) in
    let status = W.Op.get_element_type t.builder t.ptr ptr in
    Status.ok_exn status;
    Ctypes.( !@ ) ptr |> Element_type.of_c_int

  let dims t =
    let rank = rank t in
    let ptr = Ctypes.(allocate_n size_t ~count:rank) in
    let status = W.Op.get_dimensions t.builder t.ptr ptr in
    Status.ok_exn status;
    List.init rank ~f:(fun i ->
      Ctypes.( +@ ) ptr i |> Ctypes.( !@ ) |> Unsigned.Size_t.to_int)

  let constant literal ~builder = W.Op.constant_literal builder literal |> of_ptr ~builder

  let parameter name ~id ~element_type ~dims ~builder =
    let dims = List.map dims ~f:Signed.Long.of_int |> CArray.of_list Ctypes.long in
    let t =
      W.Op.parameter
        builder
        (Int64.of_int_exn id)
        (Element_type.to_c_int element_type)
        (CArray.length dims)
        (CArray.start dims)
        name
      |> of_ptr ~builder
    in
    keep_alive dims;
    t

  let not_ t = W.Op.not_ t.ptr |> of_ptr ~builder:t.builder
  let abs t = W.Op.abs t.ptr |> of_ptr ~builder:t.builder
  let exp t = W.Op.exp t.ptr |> of_ptr ~builder:t.builder
  let expm1 t = W.Op.expm1 t.ptr |> of_ptr ~builder:t.builder
  let floor t = W.Op.floor t.ptr |> of_ptr ~builder:t.builder
  let ceil t = W.Op.ceil t.ptr |> of_ptr ~builder:t.builder
  let round t = W.Op.round t.ptr |> of_ptr ~builder:t.builder
  let log t = W.Op.log t.ptr |> of_ptr ~builder:t.builder
  let log1p t = W.Op.log1p t.ptr |> of_ptr ~builder:t.builder
  let logistic t = W.Op.logistic t.ptr |> of_ptr ~builder:t.builder
  let sign t = W.Op.sign t.ptr |> of_ptr ~builder:t.builder
  let clz t = W.Op.clz t.ptr |> of_ptr ~builder:t.builder
  let cos t = W.Op.cos t.ptr |> of_ptr ~builder:t.builder
  let sin t = W.Op.sin t.ptr |> of_ptr ~builder:t.builder
  let tanh t = W.Op.tanh t.ptr |> of_ptr ~builder:t.builder
  let real t = W.Op.real t.ptr |> of_ptr ~builder:t.builder
  let imag t = W.Op.imag t.ptr |> of_ptr ~builder:t.builder
  let sqrt t = W.Op.sqrt t.ptr |> of_ptr ~builder:t.builder
  let rsqrt t = W.Op.rsqrt t.ptr |> of_ptr ~builder:t.builder
  let cbrt t = W.Op.cbrt t.ptr |> of_ptr ~builder:t.builder
  let is_finite t = W.Op.is_finite t.ptr |> of_ptr ~builder:t.builder
  let neg t = W.Op.neg t.ptr |> of_ptr ~builder:t.builder
  let lower_triangle t = W.Op.lower_triangle t.ptr |> of_ptr ~builder:t.builder
  let upper_triangle t = W.Op.upper_triangle t.ptr |> of_ptr ~builder:t.builder
  let copy t = W.Op.copy t.ptr |> of_ptr ~builder:t.builder
  let clone t = W.Op.clone t.ptr |> of_ptr ~builder:t.builder
  let zeros_like t = W.Op.zeros_like t.ptr |> of_ptr ~builder:t.builder
  let zero_like t = W.Op.zero_like t.ptr |> of_ptr ~builder:t.builder
  let add t1 t2 = W.Op.add t1.ptr t2.ptr |> of_ptr ~builder:t1.builder
  let sub t1 t2 = W.Op.sub t1.ptr t2.ptr |> of_ptr ~builder:t1.builder
  let mul t1 t2 = W.Op.mul t1.ptr t2.ptr |> of_ptr ~builder:t1.builder
  let div t1 t2 = W.Op.div t1.ptr t2.ptr |> of_ptr ~builder:t1.builder
  let rem t1 t2 = W.Op.rem t1.ptr t2.ptr |> of_ptr ~builder:t1.builder
  let max t1 t2 = W.Op.max t1.ptr t2.ptr |> of_ptr ~builder:t1.builder
  let min t1 t2 = W.Op.min t1.ptr t2.ptr |> of_ptr ~builder:t1.builder
  let and_ t1 t2 = W.Op.and_ t1.ptr t2.ptr |> of_ptr ~builder:t1.builder
  let or_ t1 t2 = W.Op.or_ t1.ptr t2.ptr |> of_ptr ~builder:t1.builder
  let xor t1 t2 = W.Op.xor t1.ptr t2.ptr |> of_ptr ~builder:t1.builder
  let atan2 t1 t2 = W.Op.atan2 t1.ptr t2.ptr |> of_ptr ~builder:t1.builder
  let pow t1 t2 = W.Op.pow t1.ptr t2.ptr |> of_ptr ~builder:t1.builder
  let dot t1 t2 = W.Op.dot t1.ptr t2.ptr |> of_ptr ~builder:t1.builder
  let eq t1 t2 = W.Op.eq t1.ptr t2.ptr |> of_ptr ~builder:t1.builder
  let ne t1 t2 = W.Op.ne t1.ptr t2.ptr |> of_ptr ~builder:t1.builder
  let ge t1 t2 = W.Op.ge t1.ptr t2.ptr |> of_ptr ~builder:t1.builder
  let gt t1 t2 = W.Op.gt t1.ptr t2.ptr |> of_ptr ~builder:t1.builder
  let le t1 t2 = W.Op.le t1.ptr t2.ptr |> of_ptr ~builder:t1.builder
  let lt t1 t2 = W.Op.lt t1.ptr t2.ptr |> of_ptr ~builder:t1.builder
  let clamp t1 ~min ~max = W.Op.clamp t1.ptr min.ptr max.ptr |> of_ptr ~builder:t1.builder

  let select ~mask ~on_true ~on_false =
    W.Op.select mask.ptr on_true.ptr on_false.ptr |> of_ptr ~builder:mask.builder

  let einsum1 t s = W.Op.einsum1 t.ptr s |> of_ptr ~builder:t.builder
  let einsum2 t1 t2 s = W.Op.einsum2 t1.ptr t2.ptr s |> of_ptr ~builder:t1.builder
  let r0_i32 v ~builder = W.Op.r0_i32 builder (Int32.of_int_exn v) |> of_ptr ~builder
  let r0_i64 v ~builder = W.Op.r0_i64 builder (Int64.of_int_exn v) |> of_ptr ~builder

  let r0_u32 v ~builder =
    W.Op.r0_u32 builder (Unsigned.UInt32.of_int v) |> of_ptr ~builder

  let r0_u64 v ~builder =
    W.Op.r0_u64 builder (Unsigned.UInt64.of_int v) |> of_ptr ~builder

  let r0_f32 v ~builder = W.Op.r0_f32 builder v |> of_ptr ~builder
  let r0_f64 v ~builder = W.Op.r0_f64 builder v |> of_ptr ~builder

  let min_value ~element_type ~builder =
    W.Op.min_value builder (Element_type.to_c_int element_type) |> of_ptr ~builder

  let max_value ~element_type ~builder =
    W.Op.max_value builder (Element_type.to_c_int element_type) |> of_ptr ~builder

  let slice_in_dim ?(stride = 1) ?(start_index = 0) t ~stop_index ~dim =
    W.Op.slice_in_dim
      t.ptr
      (Int64.of_int_exn start_index)
      (Int64.of_int_exn stop_index)
      (Int64.of_int_exn stride)
      (Int64.of_int_exn dim)
    |> of_ptr ~builder:t.builder

  let dot_general t1 t2 ~lhs_c ~rhs_c ~lhs_b ~rhs_b =
    let lhs_c = carray_i64 lhs_c in
    let rhs_c = carray_i64 rhs_c in
    let lhs_b = carray_i64 lhs_b in
    let rhs_b = carray_i64 rhs_b in
    let t =
      W.Op.dot_general
        t1.ptr
        t2.ptr
        (CArray.start lhs_c)
        (CArray.length lhs_c |> Unsigned.Size_t.of_int)
        (CArray.start rhs_c)
        (CArray.length rhs_c |> Unsigned.Size_t.of_int)
        (CArray.start lhs_b)
        (CArray.length lhs_b |> Unsigned.Size_t.of_int)
        (CArray.start rhs_b)
        (CArray.length rhs_b |> Unsigned.Size_t.of_int)
      |> of_ptr ~builder:t1.builder
    in
    keep_alive lhs_c;
    keep_alive rhs_c;
    keep_alive lhs_b;
    keep_alive rhs_b;
    t

  let gather
    t
    ~start_indices
    ~offset_dims
    ~collapsed_slice_dims
    ~start_index_map
    ~set_index_vector_dim
    ~slice_sizes
    =
    let offset_dims = carray_i64 offset_dims in
    let collapsed_slice_dims = carray_i64 collapsed_slice_dims in
    let start_index_map = carray_i64 start_index_map in
    let set_index = Ctypes.(allocate_n int64_t ~count:1) in
    Option.iter set_index_vector_dim ~f:(fun i ->
      Ctypes.( <-@ ) set_index (Int64.of_int_exn i));
    let slice_sizes = carray_i64 slice_sizes in
    let t =
      W.Op.gather
        t.ptr
        start_indices.ptr
        (CArray.start offset_dims)
        (CArray.length offset_dims |> Unsigned.Size_t.of_int)
        (CArray.start collapsed_slice_dims)
        (CArray.length collapsed_slice_dims |> Unsigned.Size_t.of_int)
        (CArray.start start_index_map)
        (CArray.length start_index_map |> Unsigned.Size_t.of_int)
        (if Option.is_some set_index_vector_dim
         then set_index
         else Ctypes.null |> Ctypes.(from_voidp int64_t))
        (CArray.start slice_sizes)
        (CArray.length slice_sizes |> Unsigned.Size_t.of_int)
      |> of_ptr ~builder:t.builder
    in
    keep_alive offset_dims;
    keep_alive collapsed_slice_dims;
    keep_alive start_index_map;
    keep_alive set_index;
    keep_alive slice_sizes;
    t

  let reshape t ~dims =
    let dims = carray_i64 dims in
    let ptr =
      W.Op.reshape
        t.ptr
        (CArray.length dims |> Unsigned.Size_t.of_int)
        (CArray.start dims)
    in
    keep_alive dims;
    of_ptr ptr ~builder:t.builder

  let maybe_keep_dims t ~res ~reduce_dims ~keep_dims =
    if keep_dims && not (List.is_empty reduce_dims)
    then (
      let dims =
        dims t
        |> List.mapi ~f:(fun i d ->
             if List.mem reduce_dims i ~equal:Int.equal then 1 else d)
      in
      reshape res ~dims)
    else res

  let reduce t ~init ~f ~dims ~keep_dims =
    let rank = rank t in
    let dims = List.map dims ~f:(fun dim_index -> normalize_index ~rank ~dim_index) in
    let dims_ = carray_i64 dims in
    let res =
      W.Op.reduce
        t.ptr
        init.ptr
        f
        (CArray.start dims_)
        (CArray.length dims_ |> Unsigned.Size_t.of_int)
      |> of_ptr ~builder:t.builder
    in
    keep_alive dims_;
    maybe_keep_dims t ~res ~reduce_dims:dims ~keep_dims

  let broadcast t ~dims =
    let dims = carray_i64 dims in
    let ptr =
      W.Op.broadcast
        t.ptr
        (CArray.length dims |> Unsigned.Size_t.of_int)
        (CArray.start dims)
    in
    keep_alive dims;
    of_ptr ptr ~builder:t.builder

  let broadcast_in_dim t ~out_dims ~broadcast_dims =
    let out_dims = carray_i64 out_dims in
    let broadcast_dims = carray_i64 broadcast_dims in
    let ptr =
      W.Op.broadcast_in_dim
        t.ptr
        (CArray.length out_dims |> Unsigned.Size_t.of_int)
        (CArray.start out_dims)
        (CArray.length broadcast_dims |> Unsigned.Size_t.of_int)
        (CArray.start broadcast_dims)
    in
    keep_alive (out_dims, broadcast_dims);
    of_ptr ptr ~builder:t.builder

  let collapse t ~dim_indexes =
    let dims = normalize_indexes t ~dim_indexes |> CArray.of_list Ctypes.int64_t in
    let ptr =
      W.Op.collapse
        t.ptr
        (CArray.length dims |> Unsigned.Size_t.of_int)
        (CArray.start dims)
    in
    keep_alive dims;
    of_ptr ptr ~builder:t.builder

  let transpose t ~dim_indexes =
    let dims = normalize_indexes t ~dim_indexes |> CArray.of_list Ctypes.int64_t in
    let ptr =
      W.Op.transpose
        t.ptr
        (CArray.length dims |> Unsigned.Size_t.of_int)
        (CArray.start dims)
    in
    keep_alive dims;
    of_ptr ptr ~builder:t.builder

  let swap_dims t ~dim_index1 ~dim_index2 =
    let rank = rank t in
    let dim1 = normalize_index ~rank ~dim_index:dim_index1 in
    let dim2 = normalize_index ~rank ~dim_index:dim_index2 in
    let dims =
      List.init rank ~f:(fun idx ->
        (if idx = dim1 then dim2 else if idx = dim2 then dim1 else idx)
        |> Int64.of_int_exn)
      |> CArray.of_list Ctypes.int64_t
    in
    let ptr =
      W.Op.transpose
        t.ptr
        (CArray.length dims |> Unsigned.Size_t.of_int)
        (CArray.start dims)
    in
    keep_alive dims;
    of_ptr ptr ~builder:t.builder

  let convert t ~element_type =
    let ptr = W.Op.convert_element_types t.ptr (Element_type.to_c_int element_type) in
    of_ptr ptr ~builder:t.builder

  let builder t = t.builder
end

module Computation = struct
  type t = W.Computation.t

  let of_ptr ptr =
    if Ctypes.is_null ptr then failwith "null Computation pointer";
    Caml.Gc.finalise W.Computation.release ptr;
    ptr

  let name = W.Computation.name

  let build ~root =
    let ptr = Ctypes.(allocate_n (ptr W.Computation.struct_) ~count:1) in
    let status = W.Computation.build root.Op.builder root.ptr ptr in
    Status.ok_exn status;
    Ctypes.( !@ ) ptr |> of_ptr
end

module PjRtClient0 = struct
  type t = W.PjRtClient.t

  let of_ptr ptr =
    if Ctypes.is_null ptr then failwith "null PjRtlient pointer";
    Caml.Gc.finalise W.PjRtClient.release ptr;
    ptr
end

module PjRtDevice = struct
  (* The lifetime of a device pointer is tied to the client from which
     the device was extracted from. *)
  type t =
    { ptr : W.PjRtDevice.t
    ; client : PjRtClient0.t
    }

  let of_ptr ptr ~client = { ptr; client }
  let id t = W.PjRtDevice.id t.ptr
  let process_index t = W.PjRtDevice.process_index t.ptr
  let local_hardware_id t = W.PjRtDevice.local_hardware_id t.ptr
  let kind t = W.PjRtDevice.kind t.ptr
  let debug_string t = W.PjRtDevice.debug_string t.ptr
  let to_string t = W.PjRtDevice.to_string t.ptr
end

module PjRtClient = struct
  include PjRtClient0

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

  let tpu ~max_inflight_computations =
    let ptr = Ctypes.(allocate_n (ptr W.PjRtClient.struct_) ~count:1) in
    let status = W.PjRtClient.tpu ptr max_inflight_computations in
    Status.ok_exn status;
    Ctypes.( !@ ) ptr |> of_ptr

  let device_count = W.PjRtClient.device_count
  let addressable_device_count = W.PjRtClient.addressable_device_count

  let devices t =
    let count = device_count t in
    let device_ptr = Ctypes.(allocate_n (ptr W.PjRtDevice.struct_) ~count) in
    W.PjRtClient.devices t device_ptr;
    List.init count ~f:(fun index ->
      Ctypes.(!@(device_ptr +@ index)) |> PjRtDevice.of_ptr ~client:t)

  let addressable_devices t =
    let count = addressable_device_count t in
    let device_ptr = Ctypes.(allocate_n (ptr W.PjRtDevice.struct_) ~count) in
    W.PjRtClient.addressable_devices t device_ptr;
    List.init count ~f:(fun index ->
      Ctypes.(!@(device_ptr +@ index)) |> PjRtDevice.of_ptr ~client:t)

  let platform_name = W.PjRtClient.platform_name
  let platform_version = W.PjRtClient.platform_version
end

module PjRtBuffer = struct
  type t =
    { ptr : W.PjRtBuffer.t
    ; client : PjRtClient.t
    }

  let of_ptr ptr ~client =
    if Ctypes.is_null ptr then failwith "null PjRtBuffer pointer";
    Caml.Gc.finalise W.PjRtBuffer.release ptr;
    { ptr; client }

  let of_host_literal literal ~device =
    let ptr = Ctypes.(allocate_n (ptr W.PjRtBuffer.struct_) ~count:1) in
    let status =
      W.PjRtBuffer.from_host_literal device.PjRtDevice.client device.ptr literal ptr
    in
    Status.ok_exn status;
    Ctypes.( !@ ) ptr |> of_ptr ~client:device.client

  let copy_to_device t ~device =
    let ptr = Ctypes.(allocate_n (ptr W.PjRtBuffer.struct_) ~count:1) in
    let status = W.PjRtBuffer.copy_to_device t.ptr device.PjRtDevice.ptr ptr in
    Status.ok_exn status;
    Ctypes.( !@ ) ptr |> of_ptr ~client:device.client

  let on_device_shape t = W.PjRtBuffer.on_device_shape t.ptr |> Shape.of_ptr

  let to_literal_sync t =
    let ptr = Ctypes.(allocate_n (ptr W.Literal.struct_) ~count:1) in
    let status = W.PjRtBuffer.to_literal_sync t.ptr ptr in
    Status.ok_exn status;
    Ctypes.( !@ ) ptr |> Literal.of_ptr
end

module PjRtLoadedExecutable = struct
  type t =
    { ptr : W.PjRtLoadedExecutable.t
    ; client : PjRtClient.t
    }

  let of_ptr ptr ~client =
    if Ctypes.is_null ptr then failwith "null PjRtLoadedExecutable pointer";
    Caml.Gc.finalise W.PjRtLoadedExecutable.release ptr;
    { ptr; client }

  let compile client computation =
    let ptr = Ctypes.(allocate_n (ptr W.PjRtLoadedExecutable.struct_) ~count:1) in
    let status = W.PjRtLoadedExecutable.compile client computation ptr in
    Status.ok_exn status;
    Ctypes.( !@ ) ptr |> of_ptr ~client

  let execute_results_to_list ptr ~client =
    let ptr = Ctypes.( !@ ) ptr in
    let rec loop_inner acc ptr =
      let deref_ptr = Ctypes.( !@ ) ptr in
      if Ctypes.is_null deref_ptr
      then Array.of_list_rev acc
      else (
        let elem = PjRtBuffer.of_ptr deref_ptr ~client in
        loop_inner (elem :: acc) (Ctypes.( +@ ) ptr 1))
    in
    let rec loop acc ptr =
      let deref_ptr = Ctypes.( !@ ) ptr in
      if Ctypes.is_null deref_ptr
      then Array.of_list_rev acc
      else (
        let elem = loop_inner [] deref_ptr in
        loop (elem :: acc) (Ctypes.( +@ ) ptr 1))
    in
    loop [] ptr

  let execute t args =
    let args = CArray.of_list W.Literal.t args in
    let ptr = Ctypes.(allocate_n (ptr (ptr (ptr W.PjRtBuffer.struct_))) ~count:1) in
    let status =
      W.PjRtLoadedExecutable.execute t.ptr (CArray.start args) (CArray.length args) ptr
    in
    keep_alive args;
    Status.ok_exn status;
    execute_results_to_list ptr ~client:t.client
end
