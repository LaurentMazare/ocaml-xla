let () =
  let cpu = Xla.Client.cpu () in
  Stdio.printf "Platform name: %s\n" (Xla.Client.platform_name cpu);
  Stdio.printf "Platform version: %s\n" (Xla.Client.platform_version cpu);
  let builder = Xla.Builder.create ~name:"mybuilder" in
  let r0_f32 = Xla.Op.r0_f32 ~builder in
  let sum = Xla.Op.add (r0_f32 39.) (r0_f32 3.) in
  let computation = Xla.Computation.build ~root:sum in
  Stdio.printf "Computation %s\n" (Xla.Computation.name computation);
  let exe = Xla.Executable.compile cpu computation in
  let buffers = Xla.Executable.execute exe [] in
  let buffers = buffers.(0) in
  Stdio.printf "Got %d buffers\n" (Array.length buffers);
  let literal = Xla.Buffer.to_literal_sync buffers.(0) in
  Stdio.printf "Size in bytes %d\n" (Xla.Literal.size_bytes literal);
  let ba = Xla.Literal.to_bigarray literal ~kind:Bigarray.float32 in
  Stdio.printf "Result %f\n" (Bigarray.Genarray.get ba [||])
