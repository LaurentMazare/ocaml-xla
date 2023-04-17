open! Import
include module type of Wrappers.Op

val take : t -> start_indices:t -> dim_index:int -> t
val reduce_sum : t -> dims:int list -> keep_dims:bool -> t
val reduce_mean : t -> dims:int list -> keep_dims:bool -> t
val reduce_max : t -> dims:int list -> keep_dims:bool -> t
val reduce_min : t -> dims:int list -> keep_dims:bool -> t
val softmax : t -> dim_index:int -> t
