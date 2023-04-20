open! Import
open! Base
module Shape = Wrappers.Shape
include Wrappers.Literal

let to_bigarray (type a b) t ~(kind : (a, b) Bigarray.kind) =
  let shape = shape t in
  Element_type.check_exn (Shape.ty shape) kind;
  let dst = Bigarray.Genarray.create kind C_layout (Shape.dimensions shape) in
  copy_to_bigarray t ~dst;
  dst

let dims t = shape t |> Shape.dimensions
