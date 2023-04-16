open! Import
include module type of Wrappers.Op

val take : t -> start_indices:t -> dim_index:int -> t
