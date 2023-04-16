module W = Xla.Wrappers

let () =
  let cpu = W.PjRtClient.cpu () in
  Stdio.printf "Platform name: %s\n" (W.PjRtClient.platform_name cpu);
  Stdio.printf "Platform version: %s\n" (W.PjRtClient.platform_version cpu)
