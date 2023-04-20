open Base

let set_log_level () = Unix.putenv "TF_CPP_MIN_LOG_LEVEL" "2"

let%expect_test _ =
  set_log_level ();
  let cpu = Xla.Client.cpu () in
  let builder = Xla.Builder.create ~name:"mybuilder" in
  let r0_f32 = Xla.Op.r0_f32 ~builder in
  let sum = Xla.Op.add (r0_f32 39.) (r0_f32 3.) in
  let computation = Xla.Computation.build ~root:sum in
  let exe = Xla.Executable.compile cpu computation in
  let buffers = Xla.Executable.execute exe [] in
  let literal = Xla.Buffer.to_literal_sync buffers.(0).(0) in
  let ba = Xla.Literal.to_bigarray literal ~kind:Bigarray.float32 in
  Stdio.printf "%f\n" (Bigarray.Genarray.get ba [||]);
  [%expect {|
        42.000000
      |}]
