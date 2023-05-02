open! Base
(*
  This example is a conversion of examples/jax_cpp/main.cc from the jax repo.
  HLO files can be generated via the following command line in the jax repo.
  python \
      jax/tools/jax_to_ir.py \
      --fn examples.jax_cpp.prog.fn \
      --input_shapes '[("x", "f32[2,2]"), ("y", "f32[2,2]")]' \
      --constants '{"z": 2.0}' \
      --ir_format HLO \
      --ir_human_dest /tmp/fn_hlo.txt  \
      --ir_dest /tmp/fn_hlo.pb
*)

let literal_array1 vs =
  Bigarray.Array1.of_array Float32 C_layout vs
  |> Bigarray.genarray_of_array1
  |> Xla.Literal.of_bigarray

let () =
  let cpu = Xla.Client.cpu () in
  let mod_ = Xla.Hlo_module_proto.read_text_file "examples/fn_hlo.txt" in
  Stdio.printf "Platform name: %s\n" (Xla.Client.platform_name cpu);
  Stdio.printf "Platform version: %s\n" (Xla.Client.platform_version cpu);
  let computation = Xla.Hlo_module_proto.computation mod_ in
  Stdio.printf "Computation %s\n" (Xla.Computation.name computation);
  let exe = Xla.Executable.compile cpu computation in
  let x = literal_array1 [| 1.; 2.; 3.; 4. |] |> Xla.Literal.reshape ~dims:[| 2; 2 |] in
  let y = literal_array1 [| 1.; 1.; 1.; 1. |] |> Xla.Literal.reshape ~dims:[| 2; 2 |] in
  let buffers = Xla.Executable.execute exe [| x; y |] in
  let buffers = buffers.(0) in
  Stdio.printf "Got %d buffers\n%!" (Array.length buffers);
  let literal = Xla.Buffer.to_literal_sync buffers.(0) in
  Stdio.printf "Literal synced\n%!";
  let literal =
    match Xla.Literal.decompose_tuple literal with
    | [| l |] -> l
    | _tuple -> failwith "unexpected number of tuple elements"
  in
  Stdio.printf "Size in bytes %d\n%!" (Xla.Literal.size_bytes literal);
  let ba = Xla.Literal.to_bigarray literal ~kind:Bigarray.float32 in
  let a = Xla.Bigarray_helper.bigarray_to_array2_exn ba in
  Stdio.print_s [%message "result" (a : float array array)]
