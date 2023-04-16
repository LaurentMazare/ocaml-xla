open! Import
open! Base
include Wrappers.Op

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
