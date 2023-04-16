open! Import
open! Base
module Shape = Wrappers.Shape
include Wrappers.Literal

let to_bigarray (type a b) t (kind : (a, b) Bigarray.kind) =
  let shape = shape t in
  Element_type.check_exn (Shape.element_type shape) kind;
  let dims = Shape.dimensions shape |> Array.of_list in
  let dst = Bigarray.Genarray.create kind C_layout dims in
  copy_to_bigarray t ~dst;
  dst
