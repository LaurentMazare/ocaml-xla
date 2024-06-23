open Base
module C = Configurator.V1

let empty_flags = { C.Pkg_config.cflags = []; libs = [] }
let ( /^ ) = Stdlib.Filename.concat
let file_exists = Stdlib.Sys.file_exists

let xla_flags () =
  let config ~lib_dir =
    let cflags = [ "-isystem"; Printf.sprintf "%s/include" lib_dir ] in
    let libs =
      [ Printf.sprintf "-Wl,-rpath,%s/lib" lib_dir
      ; Printf.sprintf "-L%s/lib" lib_dir
      ; "-lxla_extension"
      ]
    in
    { C.Pkg_config.cflags; libs }
  in
  match Stdlib.Sys.getenv_opt "XLA_EXTENSION_DIR" with
  | Some lib_dir -> config ~lib_dir
  | None ->
    let lib_dir =
      Stdlib.Sys.getenv_opt "DUNE_SOURCEROOT"
      |> Option.bind ~f:(fun prefix ->
        let lib_dir = prefix /^ "xla_extension" in
        if file_exists lib_dir then Some lib_dir else None)
    in
    let lib_dir =
      match lib_dir with
      | Some _ -> lib_dir
      | None ->
        Stdlib.Sys.getenv_opt "OPAM_SWITCH_PREFIX"
        |> Option.bind ~f:(fun prefix ->
          let lib_dir = prefix /^ "lib" /^ "libxla" in
          if file_exists lib_dir then Some lib_dir else None)
    in
    (match lib_dir with
     | Some lib_dir -> config ~lib_dir
     | None -> empty_flags)

let () =
  C.main ~name:"xla-config" (fun _c ->
    (*let xla_flags =
      try xla_flags () with
      | _ -> empty_flags
      in*)
    let xla_flags = xla_flags () in
    C.Flags.write_sexp "c_flags.sexp" xla_flags.cflags;
    C.Flags.write_sexp "c_library_flags.sexp" xla_flags.libs)
