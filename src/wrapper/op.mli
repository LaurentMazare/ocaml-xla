open! Import
include module type of Wrappers.Op

val take : t -> start_indices:t -> dim_index:int -> t
val reduce_sum : t -> dims:int array -> keep_dims:bool -> t
val reduce_mean : t -> dims:int array -> keep_dims:bool -> t
val reduce_max : t -> dims:int array -> keep_dims:bool -> t
val reduce_min : t -> dims:int array -> keep_dims:bool -> t
val softmax : t -> dim_index:int -> t
val layer_norm : t -> dim_index:int -> scale:t -> bias:t -> t
val matmul : t -> t -> t
