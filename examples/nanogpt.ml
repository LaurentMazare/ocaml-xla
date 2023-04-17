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
    let matmul _t1 _t2 = failwith "TODO" in
    let att =
      matmul q (Op.swap_dims k ~dim_index1:(-2) ~dim_index2:(-1))
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
    matmul y v
    |> Op.swap_dims ~dim_index1:1 ~dim_index2:2
    |> Op.reshape ~dims:[ b_sz; t_sz; c_sz ]
    |> Linear.forward t.c_proj
end

module Mlp = struct
  type t =
    { c_fc : Linear.t
    ; c_proj : Linear.t
    }

  let forward t xs = Linear.forward t.c_fc xs |> new_gelu |> Linear.forward t.c_proj
end

module Block = struct
  type t =
    { ln1 : LayerNorm.t
    ; attn : CausalSelfAttention.t
    ; ln2 : LayerNorm.t
    ; mlp : Mlp.t
    }

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

  let forward t xs =
    let _builder = Op.builder xs in
    let _b_sz, t_sz =
      match Op.dims xs with
      | [ b; t ] -> b, t
      | dims -> [%message "expected 2 dims" (dims : int list)] |> failwith_s
    in
    let pos = failwith "TODO" |> Op.reshape ~dims:[ 1; t_sz ] in
    let tok_emb = Embedding.forward t.wte xs in
    let pos_emb = Embedding.forward t.wpe pos in
    List.fold t.blocks ~init:(Op.add tok_emb pos_emb) ~f:(fun acc b ->
      Block.forward b acc)
    |> LayerNorm.forward t.ln_f
    |> Op.slice_in_dim ~start_index:(t_sz - 1) ~stop_index:t_sz ~dim:1
    |> Linear.forward t.lm_head
end

let _gpt_computation (config : Config.t) gpt ~b_sz =
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
  let client =
    if use_cpu
    then Xla.Client.cpu ()
    else Xla.Client.gpu ~memory_fraction:0.95 ~preallocate:false
  in
  Stdio.printf "Platform name: %s\n" (Xla.Client.platform_name client);
  Stdio.printf "Platform version: %s\n" (Xla.Client.platform_version client);
  let builder = Xla.Builder.create ~name:"mybuilder" in
  let r0_f32 = Xla.Op.r0_f32 ~builder in
  let sum = Xla.Op.add (r0_f32 39.) (r0_f32 3.) in
  let computation = Xla.Computation.build ~root:sum in
  Stdio.printf "Computation %s\n" (Xla.Computation.name computation);
  let exe = Xla.Executable.compile client computation in
  let buffers = Xla.Executable.execute exe [] in
  let buffers = buffers.(0) in
  Stdio.printf "Got %d buffers\n" (Array.length buffers);
  let literal = Xla.Buffer.to_literal_sync buffers.(0) in
  Stdio.printf "Size in bytes %d\n" (Xla.Literal.size_bytes literal);
  let ba = Xla.Literal.to_bigarray literal Bigarray.float32 in
  Stdio.printf "Result %f\n" (Bigarray.Genarray.get ba [||])
