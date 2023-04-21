include Base

let bigarray_to_array0 bigarray =
  try
    let bigarray = Bigarray.array0_of_genarray bigarray in
    Bigarray.Array0.get bigarray |> Option.some
  with
  | Invalid_argument _ -> None

let bigarray_to_array1 bigarray =
  try
    let bigarray = Bigarray.array1_of_genarray bigarray in
    Array.init (Bigarray.Array1.dim bigarray) ~f:(fun i -> bigarray.{i}) |> Option.some
  with
  | Invalid_argument _ -> None

let bigarray_to_array2 bigarray =
  try
    let bigarray = Bigarray.array2_of_genarray bigarray in
    Array.init (Bigarray.Array2.dim1 bigarray) ~f:(fun i ->
      Array.init (Bigarray.Array2.dim2 bigarray) ~f:(fun j -> bigarray.{i, j}))
    |> Option.some
  with
  | Invalid_argument _ -> None

let bigarray_to_array3 bigarray =
  try
    let bigarray = Bigarray.array3_of_genarray bigarray in
    Array.init (Bigarray.Array3.dim1 bigarray) ~f:(fun i ->
      Array.init (Bigarray.Array3.dim2 bigarray) ~f:(fun j ->
        Array.init (Bigarray.Array3.dim3 bigarray) ~f:(fun k -> bigarray.{i, j, k})))
    |> Option.some
  with
  | Invalid_argument _ -> None

let bigarray_to_array0_exn ba = Option.value_exn (bigarray_to_array0 ba)
let bigarray_to_array1_exn ba = Option.value_exn (bigarray_to_array1 ba)
let bigarray_to_array2_exn ba = Option.value_exn (bigarray_to_array2 ba)
let bigarray_to_array3_exn ba = Option.value_exn (bigarray_to_array3 ba)
