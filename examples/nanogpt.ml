(* A very simple GPT implementation based on https://github.com/karpathy/nanoGPT
   This only contains the inference part as the xla crate does not support backpropagation.
   No dropout as this is inference only.

   This example requires the following tokenizer config file:
   https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe
   And the gpt2.npz weight file that can be extracted by running the get_nanogpt_weights.py script.
*)

open! Base
module Element_type = Xla.Element_type
module Literal = Xla.Literal
module Op = Xla.Op
module T = Gpt2_tokenizer

let temperature = 0.8
let use_gpu = true
let num_samples = 10
let sample_length = 100
let failwith_s s = Sexp.to_string s |> failwith

let start_prompt =
  {|
EDWARD:
I wonder how our princely father 'scaped,
Or whether he be 'scaped away or no
From Clifford's and Northumberland's pursuit:
Had he been ta'en, we should have heard the news;
Had he been slain, we should have heard the news;
Or had he 'scaped, methinks we should have heard
The happy tidings of his good escape.
How fares my brother? why is he so sad?

RICHARD:
I cannot joy, until I be resolved
Where our right valiant father is become.
I saw him in the battle range about;
And watch'd him how he singled Clifford forth.
Methought he bore him in the thickest troop
As doth a lion in a herd of neat;
Or as a bear, encompass'd round with dogs,
Who having pinch'd a few and made them cry,
The rest stand all aloof, and bark at him.
So fared our father with his enemies;
So fled his enemies my warlike father:
Methinks, 'tis prize enough to be his son.
See how the morning opes her golden gates,
And takes her farewell of the glorious sun!
How well resembles it the prime of youth,
Trimm'd like a younker prancing to his love!

EDWARD:
Dazzle mine eyes, or do I see three suns?

RICHARD:
Three glorious suns, each one a perfect sun;
Not separated with the racking clouds,
But sever'd in a pale clear-shining sky.
See, see! they join, embrace, and seem to kiss,
As if they vow'd some league inviolable:
Now are they but one lamp, one light, one sun.
In this the heaven figures some event.

EDWARD:
'Tis wondrous strange, the like yet never heard of.
I think it cites us, brother, to the field,
That we, the sons of brave Plantagenet,
Each one already blazing by our meeds,
Should notwithstanding join our lights together
And over-shine the earth as this the world.
Whate'er it bodes, henceforward will I bear
Upon my target three fair-shining suns.
|}

module VarStore : sig
  type t

  val create_npz : string -> t
  val get : t -> string -> ty:Element_type.t -> dims:int array -> Literal.t
  val sub : t -> string -> t
end = struct
  type t =
    { literals : (string, Literal.t) Hashtbl.t
    ; rev_path : string list
    }

  let create_npz filename =
    let literals = Xla.Npy.Npz.read_all_literal filename in
    { literals; rev_path = [] }

  let sub t dir_name = { literals = t.literals; rev_path = dir_name :: t.rev_path }

  let get t name ~ty ~dims =
    let name = List.rev (name :: t.rev_path) |> String.concat ~sep:"." in
    match Hashtbl.find t.literals name with
    | None ->
      [%message
        "tensor not found" (name : string) (Hashtbl.keys t.literals : string list)]
      |> failwith_s
    | Some literal ->
      let shape = Literal.shape literal in
      let read_ty = Xla.Shape.ty shape in
      let read_dims = Xla.Shape.dimensions shape in
      if not (Element_type.equal ty read_ty)
      then
        [%message
          "element type mismatch"
            (name : string)
            (ty : Element_type.t)
            (read_ty : Element_type.t)]
        |> failwith_s;
      if Caml.( <> ) dims read_dims
      then
        [%message
          "dims mismatch" (name : string) (dims : int array) (read_dims : int array)]
        |> failwith_s;
      literal
end

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

  let create vs ~vocab_size ~n_embd =
    let embeddings = VarStore.get vs "weight" ~ty:F32 ~dims:[| vocab_size; n_embd |] in
    { embeddings }

  let forward t xs =
    let builder = Op.builder xs in
    Op.take (Op.constant t.embeddings ~builder) ~start_indices:xs ~dim_index:0
end

module LayerNorm = struct
  type t =
    { scale : Literal.t
    ; bias : Literal.t
    ; dims : int array
    }

  let create vs ~size =
    let scale = VarStore.get vs "weight" ~ty:F32 ~dims:[| size |] in
    let bias = VarStore.get vs "bias" ~ty:F32 ~dims:[| size |] in
    { scale; bias; dims = [| 1; 1; size |] }

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
    ; dims : int array
    }

  let create vs ~in_size ~out_size ~with_bias =
    let ws = VarStore.get vs "weight" ~ty:F32 ~dims:[| in_size; out_size |] in
    let bs =
      if with_bias
      then VarStore.get vs "bias" ~ty:F32 ~dims:[| out_size |] |> Option.some
      else None
    in
    { ws; bs; dims = [| 1; 1; out_size |] }

  let forward t xs =
    let builder = Op.builder xs in
    let rank = Op.rank xs in
    let ws = Op.constant t.ws ~builder in
    let xs =
      Op.dot_general xs ws ~lhs_c:[| rank - 1 |] ~rhs_c:[| 0 |] ~lhs_b:[||] ~rhs_b:[||]
    in
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

  let create vs ~n_head ~n_embd =
    let sub = VarStore.sub vs in
    let c_attn =
      Linear.create (sub "c_attn") ~in_size:n_embd ~out_size:(3 * n_embd) ~with_bias:true
    in
    let c_proj =
      Linear.create (sub "c_proj") ~in_size:n_embd ~out_size:n_embd ~with_bias:true
    in
    { c_attn; c_proj; n_head; n_embd }

  let masked_fill ~mask ~on_true ~on_false =
    let dims = Op.dims mask in
    let on_true = Op.r0_f32 on_true ~builder:(Op.builder mask) |> Op.broadcast ~dims in
    Op.select ~mask ~on_true ~on_false

  let forward t xs =
    let builder = Op.builder xs in
    let b_sz, t_sz, c_sz =
      match Op.dims xs with
      | [| b; t; c |] -> b, t, c
      | dims -> [%message "expected 3 dims" (dims : int array)] |> failwith_s
    in
    let qkv = Linear.forward t.c_attn xs in
    let slice_qkv ~start_index ~stop_index =
      Op.slice_in_dim qkv ~start_index ~stop_index ~dim_index:2
      |> Op.reshape ~dims:[| b_sz; t_sz; t.n_head; c_sz / t.n_head |]
      |> Op.swap_dims ~dim_index1:1 ~dim_index2:2
    in
    let q = slice_qkv ~start_index:(0 * t.n_embd) ~stop_index:(1 * t.n_embd) in
    let k = slice_qkv ~start_index:(1 * t.n_embd) ~stop_index:(2 * t.n_embd) in
    let v = slice_qkv ~start_index:(2 * t.n_embd) ~stop_index:(3 * t.n_embd) in
    let att =
      Op.matmul q (Op.swap_dims k ~dim_index1:(-2) ~dim_index2:(-1))
      |> Op.mul
           (Op.r0_f32
              (1. /. Float.sqrt (Op.dims k |> Array.last |> Float.of_int))
              ~builder)
    in
    let mask =
      Op.r0_i32 1 ~builder
      |> Op.broadcast ~dims:[| t_sz; t_sz |]
      |> Op.lower_triangle
      |> Op.reshape ~dims:[| 1; 1; t_sz; t_sz |]
    in
    let zero =
      Op.r0_i32 0 ~builder |> Op.broadcast ~dims:[| b_sz; t.n_head; t_sz; t_sz |]
    in
    let att =
      masked_fill ~mask:(Op.eq mask zero) ~on_false:att ~on_true:Float.neg_infinity
    in
    let y = Op.softmax att ~dim_index:(-1) in
    Op.matmul y v
    |> Op.swap_dims ~dim_index1:1 ~dim_index2:2
    |> Op.reshape ~dims:[| b_sz; t_sz; c_sz |]
    |> Linear.forward t.c_proj
end

module Mlp = struct
  type t =
    { c_fc : Linear.t
    ; c_proj : Linear.t
    }

  let create vs ~n_embd =
    let sub = VarStore.sub vs in
    let c_fc =
      Linear.create (sub "c_fc") ~in_size:n_embd ~out_size:(4 * n_embd) ~with_bias:true
    in
    let c_proj =
      Linear.create (sub "c_proj") ~in_size:(4 * n_embd) ~out_size:n_embd ~with_bias:true
    in
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

  let create vs ~n_head ~n_embd =
    let sub = VarStore.sub vs in
    let ln1 = LayerNorm.create (sub "ln_1") ~size:n_embd in
    let attn = CausalSelfAttention.create (sub "attn") ~n_head ~n_embd in
    let ln2 = LayerNorm.create (sub "ln_2") ~size:n_embd in
    let mlp = Mlp.create (sub "mlp") ~n_embd in
    { ln1; attn; ln2; mlp }

  let forward t xs =
    let xs =
      LayerNorm.forward t.ln1 xs |> CausalSelfAttention.forward t.attn |> Op.add xs
    in
    LayerNorm.forward t.ln2 xs |> Mlp.forward t.mlp |> Op.add xs
end

module Gpt = struct
  type t =
    { lm_head : Linear.t
    ; wte : Embedding.t
    ; wpe : Embedding.t
    ; blocks : Block.t list
    ; ln_f : LayerNorm.t
    }

  let create vs config =
    let sub = VarStore.sub vs in
    let transformer_vs = sub "transformer" in
    let sub_t = VarStore.sub transformer_vs in
    let { Config.n_embd; n_head; vocab_size; block_size; n_layer; _ } = config in
    let lm_head =
      Linear.create (sub "lm_head") ~in_size:n_embd ~out_size:vocab_size ~with_bias:false
    in
    let wte = Embedding.create (sub_t "wte") ~vocab_size ~n_embd in
    let wpe = Embedding.create (sub_t "wpe") ~vocab_size:block_size ~n_embd in
    let blocks =
      let vs = sub_t "h" in
      let vs i = VarStore.sub vs (Int.to_string i) in
      List.init n_layer ~f:(fun i -> Block.create (vs i) ~n_head ~n_embd)
    in
    let ln_f = LayerNorm.create (sub_t "ln_f") ~size:n_embd in
    { lm_head; wte; wpe; blocks; ln_f }

  let forward t xs =
    let builder = Op.builder xs in
    let _b_sz, t_sz =
      match Op.dims xs with
      | [| b; t |] -> b, t
      | dims -> [%message "expected 2 dims" (dims : int array)] |> failwith_s
    in
    let pos =
      Array.init t_sz ~f:Int64.of_int_exn
      |> Bigarray.Array1.of_array Int64 C_layout
      |> Bigarray.genarray_of_array1
      |> Literal.of_bigarray
      |> Op.constant ~builder
      |> Op.reshape ~dims:[| 1; t_sz |]
    in
    let tok_emb = Embedding.forward t.wte xs in
    let pos_emb = Embedding.forward t.wpe pos in
    List.fold t.blocks ~init:(Op.add tok_emb pos_emb) ~f:(fun acc b ->
      Block.forward b acc)
    |> LayerNorm.forward t.ln_f
    |> Op.slice_in_dim ~start_index:(t_sz - 1) ~stop_index:t_sz ~dim_index:1
    |> Linear.forward t.lm_head
end

let gpt_computation vs config ~b_sz =
  let gpt = Gpt.create vs config in
  let builder = Xla.Builder.create ~name:"gpt" in
  let input =
    Op.parameter "tokens" ~id:0 ~ty:S32 ~dims:[| b_sz; config.block_size |] ~builder
  in
  let logits = Gpt.forward gpt input in
  let root =
    Op.div logits (Op.r0_f32 temperature ~builder) |> Op.softmax ~dim_index:(-1)
  in
  Xla.Computation.build ~root

let sample ~start_tokens ~tokenizer ~exe =
  let block_size = Config.gpt2.block_size in
  let vocab_size = Config.gpt2.vocab_size in
  let tokens = Queue.create () in
  let ba = Bigarray.Array2.create Int32 C_layout 1 block_size in
  for _i = 1 to sample_length do
    for idx = 0 to block_size - 1 do
      let tokens_index = Queue.length tokens - block_size + idx in
      let token =
        if tokens_index < 0
        then (
          let tokens_index = Array.length start_tokens + tokens_index in
          if tokens_index < 0 then 50256 else start_tokens.(tokens_index))
        else Queue.get tokens tokens_index
      in
      ba.{0, idx} <- Int32.of_int_exn token
    done;
    let ba = Bigarray.genarray_of_array2 ba in
    let buffers = Xla.Executable.execute exe [| Literal.of_bigarray ba |] in
    let probabilities =
      Xla.Buffer.to_literal_sync buffers.(0).(0)
      |> Literal.to_bigarray ~kind:Float32
      |> fun ba -> Bigarray.reshape_1 ba vocab_size
    in
    (* Naive linear time multinomial sampling. *)
    let sum_p = ref 0. in
    for i = 0 to vocab_size - 1 do
      sum_p := !sum_p +. probabilities.{i}
    done;
    let p = ref (Random.float !sum_p) in
    let token = ref None in
    for i = 0 to vocab_size - 1 do
      p := !p -. probabilities.{i};
      if Float.is_non_positive !p && Option.is_none !token then token := Some i
    done;
    let token = Option.value !token ~default:0 in
    Queue.enqueue tokens token
  done;
  Queue.to_list tokens |> T.decode tokenizer

let () =
  let tokenizer = T.create ~merge_filename:"vocab.bpe" in
  let start_tokens = T.encode tokenizer start_prompt |> Array.of_list in
  let client =
    if use_gpu
    then Xla.Client.gpu ~memory_fraction:0.95 ~preallocate:false
    else Xla.Client.cpu ()
  in
  Stdio.printf "Platform name: %s\n" (Xla.Client.platform_name client);
  Stdio.printf "Platform version: %s\n%!" (Xla.Client.platform_version client);
  let vs = time_it "Read weight file" ~f:(fun () -> VarStore.create_npz "gpt2.npz") in
  let gpt2 =
    time_it "Generated the op" ~f:(fun () -> gpt_computation vs Config.gpt2 ~b_sz:1)
  in
  let exe =
    time_it "Compiled the model" ~f:(fun () -> Xla.Executable.compile client gpt2)
  in
  for i = 1 to num_samples do
    time_it "Sampled" ~f:(fun () ->
      let sample = sample ~start_tokens ~tokenizer ~exe in
      Stdio.printf "%d ----\n%s\n----\n%!" i sample)
  done
