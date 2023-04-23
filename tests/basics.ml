open Base

let set_log_level () = Unix.putenv "TF_CPP_MIN_LOG_LEVEL" "2"

let sexp_of_bigarray (type a b) (ba : (a, b, _) Bigarray.Genarray.t) =
  let sexp_of_elem : a -> Sexp.t =
    match (Bigarray.Genarray.kind ba : (a, b) Bigarray.kind) with
    | Float32 -> sexp_of_float
    | Float64 -> sexp_of_float
    | _ -> failwith "unsupported kind"
  in
  match Bigarray.Genarray.dims ba with
  | [||] -> Xla.Bigarray_helper.bigarray_to_array0_exn ba |> [%sexp_of: elem]
  | [| _ |] -> Xla.Bigarray_helper.bigarray_to_array1_exn ba |> [%sexp_of: elem array]
  | [| _; _ |] ->
    Xla.Bigarray_helper.bigarray_to_array2_exn ba |> [%sexp_of: elem array array]
  | [| _; _; _ |] ->
    Xla.Bigarray_helper.bigarray_to_array3_exn ba |> [%sexp_of: elem array array array]
  | dims ->
    [%message "unsupported shape" (dims : int array)] |> Sexp.to_string |> failwith

let eval ~args ~f =
  let open Xla in
  let cpu = Client.cpu () in
  let builder = Builder.create ~name:"mybuilder" in
  let root = f ~builder in
  let computation = Computation.build ~root in
  let exe = Executable.compile cpu computation in
  let buffers = Executable.execute exe args in
  let literal = Buffer.to_literal_sync buffers.(0).(0) in
  let ba = Literal.to_bigarray literal ~kind:Bigarray.float32 in
  let dims = Bigarray.Genarray.dims ba in
  Stdio.print_s [%message (dims : int array) (ba : bigarray)]

let%expect_test _ =
  set_log_level ();
  eval ~args:[||] ~f:(fun ~builder ->
    let r0_f32 = Xla.Op.r0_f32 ~builder in
    Xla.Op.add (r0_f32 39.) (r0_f32 3.));
  [%expect {|
        ((dims ()) (ba 42))
  |}];
  eval ~args:[||] ~f:(fun ~builder ->
    let r0_f32 = Xla.Op.r0_f32 ~builder in
    Xla.Op.sub (r0_f32 39.) (r0_f32 3.));
  [%expect {|
        ((dims ()) (ba 36))
  |}];
  eval ~args:[||] ~f:(fun ~builder ->
    let r0_f32 = Xla.Op.r0_f32 ~builder in
    Xla.Op.mul (r0_f32 39.) (r0_f32 3.));
  [%expect {|
        ((dims ()) (ba 117))
  |}];
  eval ~args:[||] ~f:(fun ~builder ->
    let r0_f32 = Xla.Op.r0_f32 ~builder in
    Xla.Op.div (r0_f32 39.) (r0_f32 3.));
  [%expect {|
        ((dims ()) (ba 13))
      |}]
