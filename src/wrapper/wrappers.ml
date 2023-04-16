open! Base
open! Import
module CArray = Ctypes.CArray

let _add_compact =
  match Sys.getenv "OCAML_XLA_ADD_COMPACT" with
  | None | Some "false" | Some "0" -> false
  | Some _ -> true

let _check_and_release_status status =
  if not (Ctypes.is_null status)
  then (
    let error_message = W.Status.error_message status in
    W.Status.release status;
    failwith error_message)

module Shape = struct end
