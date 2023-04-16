module W = Xla.Wrappers

let () =
  let cpu = W.PjRtClient.cpu () in
  Stdio.printf "Platform name: %s\n" (W.PjRtClient.platform_name cpu);
  Stdio.printf "Platform version: %s\n" (W.PjRtClient.platform_version cpu);
  let builder = W.Builder.create ~name:"mybuilder" in
  let r0_f32 = W.Op.r0_f32 ~builder in
  let sum = W.Op.add (r0_f32 39.) (r0_f32 3.) in
  let computation = W.Computation.build ~root:sum in
  Stdio.printf "Computation %s\n" (W.Computation.name computation);
  let exe = W.PjRtLoadedExecutable.compile cpu computation in
  let buffers = W.PjRtLoadedExecutable.execute exe [] in
  let buffers = buffers.(0) in
  Stdio.printf "Got %d buffers\n" (Array.length buffers);
  let literal = W.PjRtBuffer.to_literal_sync buffers.(0) in
  Stdio.printf "Size in bytes %d\n" (W.Literal.size_bytes literal);
  let ba = W.Literal.to_bigarray literal Bigarray.float32 in
  Stdio.printf "Result %f\n" (Bigarray.Genarray.get ba [||])
