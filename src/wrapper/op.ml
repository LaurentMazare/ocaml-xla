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
  let element_type = element_type t in
  let x = parameter "x" ~id:0 ~element_type ~dims:[] ~builder in
  let y = parameter "y" ~id:1 ~element_type ~dims:[] ~builder in
  reduce
    t
    ~init:(zero_like t)
    ~f:(Wrappers.Computation.build ~root:(add x y))
    ~dims
    ~keep_dims

let reduce_mean t ~dims ~keep_dims =
  let builder = builder t in
  let scale =
    List.fold dims ~init:(r0_u64 0 ~builder) ~f:(fun acc dim_index ->
      dimensions_size t ~dim_index |> mul acc)
  in
  div (reduce_sum t ~dims ~keep_dims) (convert scale ~element_type:(element_type t))

let reduce_max t ~dims ~keep_dims =
  (* TODO: memoize the computation? *)
  let max =
    let builder = Builder.create ~name:"max" in
    let element_type = element_type t in
    let x = parameter "x" ~id:0 ~element_type ~dims:[] ~builder in
    let y = parameter "y" ~id:1 ~element_type ~dims:[] ~builder in
    max x y
  in
  reduce
    t
    ~init:(min_value ~element_type:(element_type t) ~builder:(builder t))
    ~f:(Wrappers.Computation.build ~root:max)
    ~dims
    ~keep_dims

let reduce_min t ~dims ~keep_dims =
  (* TODO: memoize the computation? *)
  let min =
    let builder = Builder.create ~name:"max" in
    let element_type = element_type t in
    let x = parameter "x" ~id:0 ~element_type ~dims:[] ~builder in
    let y = parameter "y" ~id:1 ~element_type ~dims:[] ~builder in
    min x y
  in
  reduce
    t
    ~init:(max_value ~element_type:(element_type t) ~builder:(builder t))
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
  let eps = r0_f32 1e-5 ~builder |> convert ~element_type:(element_type t) in
  let mean = reduce_mean t ~dims:[ dim_index ] ~keep_dims:true in
  let mean2 = reduce_mean (mul t t) ~dims:[ dim_index ] ~keep_dims:true in
  let var = sub mean2 (mul mean mean) in
  let s = add var eps |> rsqrt in
  add bias (mul s (sub t mean) |> mul scale)
