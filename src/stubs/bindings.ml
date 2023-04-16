open! Ctypes

module C (F : Cstubs.FOREIGN) = struct
  open! F

  module Status = struct
    type modl
    type struct_ = modl Ctypes.structure
    type t = struct_ ptr

    let struct_ : struct_ typ = structure "Status"
    let t : t typ = ptr struct_
    let error_message = foreign "status_error_message" (t @-> returning string)
    let release = foreign "status_free" (t @-> returning void)
  end
end
