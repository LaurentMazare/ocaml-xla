(* A very simple GPT implementation based on https://github.com/karpathy/nanoGPT
   This only contains the inference part as the xla crate does not support backpropagation.
   No dropout as this is inference only.

   This example requires the following tokenizer config file:
   https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe
   And the gpt2.npz weight file that can be extracted by running the get_weights.py script.
*)

open! Base
module Literal = Xla.Literal
module Op = Xla.Op

let temperature = 0.8
let use_cpu = true
let num_samples = 10
let failwith_s s = Sexp.to_string s |> failwith

let time_it str ~f =
  let start_time = Unix.gettimeofday () in
  let res = f () in
  Stdio.printf "%s (%.2fs)\n%!" str (Unix.gettimeofday () -. start_time);
  res

module Config = struct
  type t =
    { block_size : int
    ; vocab_size : int
    ; n_layer : int
    ; n_head : int
    ; n_embd : int
    }

  let gpt2 =
    { block_size = 1024; vocab_size = 50257; n_layer = 12; n_head = 12; n_embd = 768 }
end

let new_gelu xs =
  let builder = Op.builder xs in
  let f = Op.r0_f32 ~builder in
  let sqrt_two_over_pi = Float.sqrt (2. /. Float.pi) |> f in
  (* 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))) *)
  let vs = Op.mul sqrt_two_over_pi (Op.add xs (Op.mul (Op.pow xs (f 3.)) (f 0.044715))) in
  Op.mul (f 0.5) (Op.mul xs (Op.add (Op.tanh vs) (f 1.)))

module Embedding = struct
  type t = { embeddings : Literal.t }

  let create ~vocab_size ~n_embd =
    let embeddings = Literal.create ~element_type:F32 ~dims:[ vocab_size; n_embd ] in
    { embeddings }

  let forward t xs =
    let builder = Op.builder xs in
    Op.take (Op.constant t.embeddings ~builder) ~start_indices:xs ~dim_index:0
end

module LayerNorm = struct
  type t =
    { scale : Literal.t
    ; bias : Literal.t
    ; dims : int list
    }

  let create ~size =
    let scale = Literal.create ~element_type:F32 ~dims:[ size ] in
    let bias = Literal.create ~element_type:F32 ~dims:[ size ] in
    { scale; bias; dims = [ 1; 1; size ] }

  let forward t xs =
    let builder = Op.builder xs in
    let scale = Op.constant t.scale ~builder |> Op.reshape ~dims:t.dims in
    let bias = Op.constant t.bias ~builder |> Op.reshape ~dims:t.dims in
    Op.layer_norm xs ~dim_index:(-1) ~scale ~bias
end

module Linear = struct
  type t =
    { ws : Literal.t
    ; bs : Literal.t option
    ; dims : int list
    }

  let create ~in_size ~out_size ~with_bias =
    let ws = Literal.create ~element_type:F32 ~dims:[ in_size; out_size ] in
    let bs =
      if with_bias
      then Literal.create ~element_type:F32 ~dims:[ out_size ] |> Option.some
      else None
    in
    { ws; bs; dims = [ 1; 1; out_size ] }

  let forward t xs =
    let builder = Op.builder xs in
    let rank = Op.rank xs in
    let ws = Op.constant t.ws ~builder in
    let xs = Op.dot_general xs ws ~lhs_c:[ rank - 1 ] ~rhs_c:[ 0 ] ~lhs_b:[] ~rhs_b:[] in
    match t.bs with
    | None -> xs
    | Some bs ->
      let bs = Op.constant bs ~builder |> Op.reshape ~dims:t.dims in
      Op.add xs bs
end

module CausalSelfAttention = struct
  type t =
    { c_attn : Linear.t
    ; c_proj : Linear.t
    ; n_head : int
    ; n_embd : int
    }

  let create ~n_head ~n_embd =
    let c_attn = Linear.create ~in_size:n_embd ~out_size:(3 * n_embd) ~with_bias:true in
    let c_proj = Linear.create ~in_size:n_embd ~out_size:n_embd ~with_bias:true in
    { c_attn; c_proj; n_head; n_embd }

  let masked_fill ~mask ~on_true ~on_false =
    let dims = Op.dims mask in
    let on_true = Op.r0_f32 on_true ~builder:(Op.builder mask) |> Op.broadcast ~dims in
    Op.select ~mask ~on_true ~on_false

  let forward t xs =
    let builder = Op.builder xs in
    let b_sz, t_sz, c_sz =
      match Op.dims xs with
      | [ b; t; c ] -> b, t, c
      | dims -> [%message "expected 3 dims" (dims : int list)] |> failwith_s
    in
    let qkv = Linear.forward t.c_attn xs in
    let slice_qkv ~start_index ~stop_index =
      Op.slice_in_dim qkv ~start_index ~stop_index ~dim:2
      |> Op.reshape ~dims:[ b_sz; t_sz; t.n_head; c_sz / t.n_head ]
      |> Op.swap_dims ~dim_index1:1 ~dim_index2:2
    in
    let q = slice_qkv ~start_index:(0 * t.n_embd) ~stop_index:(1 * t.n_embd) in
    let k = slice_qkv ~start_index:(1 * t.n_embd) ~stop_index:(2 * t.n_embd) in
    let v = slice_qkv ~start_index:(2 * t.n_embd) ~stop_index:(3 * t.n_embd) in
    let att =
      Op.matmul q (Op.swap_dims k ~dim_index1:(-2) ~dim_index2:(-1))
      |> Op.mul
           (Op.r0_f32
              (1. /. Float.sqrt (Op.dims k |> List.last_exn |> Float.of_int))
              ~builder)
    in
    let mask =
      Op.r0_i32 1 ~builder
      |> Op.broadcast ~dims:[ t_sz; t_sz ]
      |> Op.lower_triangle
      |> Op.reshape ~dims:[ 1; 1; t_sz; t_sz ]
    in
    let zero =
      Op.r0_i32 0 ~builder |> Op.broadcast ~dims:[ b_sz; t.n_head; t_sz; t_sz ]
    in
    let att =
      masked_fill ~mask:(Op.eq mask zero) ~on_false:att ~on_true:Float.neg_infinity
    in
    let y = Op.softmax att ~dim_index:(-1) in
    Op.matmul y v
    |> Op.swap_dims ~dim_index1:1 ~dim_index2:2
    |> Op.reshape ~dims:[ b_sz; t_sz; c_sz ]
    |> Linear.forward t.c_proj
end

module Mlp = struct
  type t =
    { c_fc : Linear.t
    ; c_proj : Linear.t
    }

  let create ~n_embd =
    let c_fc = Linear.create ~in_size:n_embd ~out_size:(4 * n_embd) ~with_bias:true in
    let c_proj = Linear.create ~in_size:(4 * n_embd) ~out_size:n_embd ~with_bias:true in
    { c_fc; c_proj }

  let forward t xs = Linear.forward t.c_fc xs |> new_gelu |> Linear.forward t.c_proj
end

module Block = struct
  type t =
    { ln1 : LayerNorm.t
    ; attn : CausalSelfAttention.t
    ; ln2 : LayerNorm.t
    ; mlp : Mlp.t
    }

  let create ~n_head ~n_embd =
    let ln1 = LayerNorm.create ~size:n_embd in
    let attn = CausalSelfAttention.create ~n_head ~n_embd in
    let ln2 = LayerNorm.create ~size:n_embd in
    let mlp = Mlp.create ~n_embd in
    { ln1; attn; ln2; mlp }

  let forward t xs =
    LayerNorm.forward t.ln1 xs
    |> CausalSelfAttention.forward t.attn
    |> Op.add xs
    |> LayerNorm.forward t.ln2
    |> Mlp.forward t.mlp
    |> Op.add xs
end

module Gpt = struct
  type t =
    { lm_head : Linear.t
    ; wte : Embedding.t
    ; wpe : Embedding.t
    ; blocks : Block.t list
    ; ln_f : LayerNorm.t
    }

  let create config =
    let { Config.n_embd; n_head; vocab_size; block_size; n_layer; _ } = config in
    let lm_head = Linear.create ~in_size:n_embd ~out_size:vocab_size ~with_bias:false in
    let wte = Embedding.create ~vocab_size ~n_embd in
    let wpe = Embedding.create ~vocab_size:block_size ~n_embd in
    let blocks = List.init n_layer ~f:(fun _i -> Block.create ~n_head ~n_embd) in
    let ln_f = LayerNorm.create ~size:n_embd in
    { lm_head; wte; wpe; blocks; ln_f }

  let forward t xs =
    let builder = Op.builder xs in
    let _b_sz, t_sz =
      match Op.dims xs with
      | [ b; t ] -> b, t
      | dims -> [%message "expected 2 dims" (dims : int list)] |> failwith_s
    in
    let pos =
      Array.init t_sz ~f:Int64.of_int_exn
      |> Bigarray.Array1.of_array Int64 C_layout
      |> Bigarray.genarray_of_array1
      |> Literal.of_bigarray
      |> Op.constant ~builder
      |> Op.reshape ~dims:[ 1; t_sz ]
    in
    let tok_emb = Embedding.forward t.wte xs in
    let pos_emb = Embedding.forward t.wpe pos in
    List.fold t.blocks ~init:(Op.add tok_emb pos_emb) ~f:(fun acc b ->
      Block.forward b acc)
    |> LayerNorm.forward t.ln_f
    |> Op.slice_in_dim ~start_index:(t_sz - 1) ~stop_index:t_sz ~dim:1
    |> Linear.forward t.lm_head
end

let gpt_computation (config : Config.t) ~b_sz =
  let gpt = Gpt.create config in
  let builder = Xla.Builder.create ~name:"gpt" in
  let input =
    Op.parameter
      "tokens"
      ~id:0
      ~element_type:S32
      ~dims:[ b_sz; config.block_size ]
      ~builder
  in
  let logits = Gpt.forward gpt input in
  let root =
    Op.div logits (Op.r0_f32 temperature ~builder) |> Op.softmax ~dim_index:(-1)
  in
  Xla.Computation.build ~root

let () =
  let b_sz = 2 in
  let client =
    if use_cpu
    then Xla.Client.cpu ()
    else Xla.Client.gpu ~memory_fraction:0.95 ~preallocate:false
  in
  Stdio.printf "Platform name: %s\n" (Xla.Client.platform_name client);
  Stdio.printf "Platform version: %s\n%!" (Xla.Client.platform_version client);
  let weights =
    time_it "Read weight file" ~f:(fun () -> Xla.Npy.Npz.read_all "gpt2.npz")
  in
  Hashtbl.iter_keys weights ~f:(fun name -> Stdio.printf "Literal %s\n%!" name);
  let gpt2 =
    time_it "Generated the op" ~f:(fun () -> gpt_computation Config.gpt2 ~b_sz)
  in
  let exe =
    time_it "Compiled the model" ~f:(fun () -> Xla.Executable.compile client gpt2)
  in
  for i = 1 to num_samples do
    let input = Literal.create ~element_type:F32 ~dims:[ b_sz; Config.gpt2.block_size ] in
    let buffers =
      time_it "Generated some samples" ~f:(fun () -> Xla.Executable.execute exe [ input ])
    in
    let buffers = buffers.(0) in
    Stdio.printf "%d Got %d buffers\n%!" i (Array.length buffers)
  done
