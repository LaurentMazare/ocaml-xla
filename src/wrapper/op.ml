open! Import
open! Base
include Wrappers.Op
module Builder = Wrappers.Builder

let take t ~start_indices ~dim_index =
  let start_indices_dims = dims start_indices in
  let dims = dims t in
  let rank = List.length dims in
  let rank_start_indices = List.length start_indices_dims in
  let dim_index = normalize_index ~rank ~dim_index in
  let offset_dims =
    List.init (rank + rank_start_indices - 1) ~f:Fn.id
    |> List.filter ~f:(fun i -> i < dim_index || i >= dim_index + rank_start_indices)
  in
  let slice_sizes = List.mapi dims ~f:(fun i d -> if i = dim_index then 1 else d) in
  let start_indices = reshape start_indices ~dims:(start_indices_dims @ [ 1 ]) in
  (* Same as in Jax: always use the last dimension for index_vector_dim. *)
  gather
    t
    ~start_indices
    ~offset_dims
    ~collapsed_slice_dims:[ dim_index ]
    ~start_index_map:[ dim_index ]
    ~set_index_vector_dim:(Some rank_start_indices)
    ~slice_sizes

let reduce_sum t ~dims ~keep_dims =
  (* TODO: memoize the computation? *)
  let builder = Builder.create ~name:"sum" in
  let ty = ty t in
  let x = parameter "x" ~id:0 ~ty ~dims:[] ~builder in
  let y = parameter "y" ~id:1 ~ty ~dims:[] ~builder in
  reduce
    t
    ~init:(zero_like t)
    ~f:(Wrappers.Computation.build ~root:(add x y))
    ~dims
    ~keep_dims

let reduce_mean t ~dims ~keep_dims =
  let builder = builder t in
  let scale =
    List.fold dims ~init:(r0_i32 1 ~builder) ~f:(fun acc dim_index ->
      dimensions_size t ~dim_index |> mul acc)
  in
  div (reduce_sum t ~dims ~keep_dims) (convert scale ~ty:(ty t))

let reduce_max t ~dims ~keep_dims =
  (* TODO: memoize the computation? *)
  let max =
    let builder = Builder.create ~name:"max" in
    let ty = ty t in
    let x = parameter "x" ~id:0 ~ty ~dims:[] ~builder in
    let y = parameter "y" ~id:1 ~ty ~dims:[] ~builder in
    max x y
  in
  reduce
    t
    ~init:(min_value ~ty:(ty t) ~builder:(builder t))
    ~f:(Wrappers.Computation.build ~root:max)
    ~dims
    ~keep_dims

let reduce_min t ~dims ~keep_dims =
  (* TODO: memoize the computation? *)
  let min =
    let builder = Builder.create ~name:"max" in
    let ty = ty t in
    let x = parameter "x" ~id:0 ~ty ~dims:[] ~builder in
    let y = parameter "y" ~id:1 ~ty ~dims:[] ~builder in
    min x y
  in
  reduce
    t
    ~init:(max_value ~ty:(ty t) ~builder:(builder t))
    ~f:(Wrappers.Computation.build ~root:min)
    ~dims
    ~keep_dims

let softmax t ~dim_index =
  let max = reduce_max t ~dims:[ dim_index ] ~keep_dims:true in
  let unnormalized = sub t max |> exp in
  let sum = reduce_sum unnormalized ~dims:[ dim_index ] ~keep_dims:true in
  div unnormalized sum

let layer_norm t ~dim_index ~scale ~bias =
  let builder = builder t in
  let eps = r0_f32 1e-5 ~builder |> convert ~ty:(ty t) in
  let mean = reduce_mean t ~dims:[ dim_index ] ~keep_dims:true in
  let mean2 = reduce_mean (mul t t) ~dims:[ dim_index ] ~keep_dims:true in
  let var = sub mean2 (mul mean mean) in
  let s = add var eps |> rsqrt in
  add bias (mul s (sub t mean) |> mul scale)

(* Similar to the jax implementation but without the squeezing.
   https://github.com/google/jax/blob/849e47f79ac64ccba1a762804217c00a9905025b/jax/_src/numpy/lax_numpy.py#L3028
*)
let matmul lhs rhs =
  let lhs_dims = dims lhs |> Array.of_list in
  let rhs_dims = dims rhs |> Array.of_list in
  let lhs_ndims = Array.length lhs_dims in
  let rhs_ndims = Array.length rhs_dims in
  if lhs_ndims < 1 || rhs_ndims < 1
  then
    [%message "empty dimension in matmul" (lhs_dims : int array) (rhs_dims : int array)]
    |> failwith_s;
  let rhs_is_mat = rhs_ndims > 1 in
  let lhs_batch_ndims = Int.max 0 (lhs_ndims - 2) in
  let rhs_batch_ndims = Int.max 0 (rhs_ndims - 2) in
  let max_batch_ndims = Int.max lhs_batch_ndims rhs_batch_ndims in
  let lhs_batch_dims = Queue.create () in
  let rhs_batch_dims = Queue.create () in
  for idx = 0 to max_batch_ndims - 1 do
    let lhs_idx = idx + lhs_batch_ndims - max_batch_ndims in
    let rhs_idx = idx + rhs_batch_ndims - max_batch_ndims in
    if lhs_idx < 0 && rhs_idx < 0
    then
      [%message "matmul internal error" (lhs_dims : int array) (rhs_dims : int array)]
      |> failwith_s;
    if lhs_idx < 0 && rhs_idx >= 0
    then Queue.enqueue rhs_batch_dims rhs_idx
    else if lhs_idx >= 0 && rhs_idx < 0
    then Queue.enqueue lhs_batch_dims lhs_idx
    else if lhs_dims.(lhs_idx) = rhs_dims.(rhs_idx)
    then (
      Queue.enqueue lhs_batch_dims lhs_idx;
      Queue.enqueue rhs_batch_dims rhs_idx)
    else
      [%message
        "incompatible batch dims"
          (lhs_dims : int array)
          (rhs_dims : int array)
          (lhs_idx : int)
          (rhs_idx : int)]
      |> failwith_s
  done;
  dot_general
    lhs
    rhs
    ~lhs_c:[ lhs_ndims - 1 ]
    ~rhs_c:[ (rhs_ndims - if rhs_is_mat then 2 else 1) ]
    ~lhs_b:(Queue.to_list lhs_batch_dims)
    ~rhs_b:(Queue.to_list rhs_batch_dims)
