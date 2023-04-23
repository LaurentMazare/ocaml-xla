(* An implementation of LLaMA https://github.com/facebookresearch/llama
   This only contains the inference part as the xla crate does not support backpropagation.
  
   This is based on nanoGPT in a similar way to:
   https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py
  
   The tokenizer config can be retrieved from:
   https://huggingface.co/hf-internal-testing/llama-tokenizer/blob/main/tokenizer.json
  
   In order to convert the llama weights to a .npz file, run:
   python examples/convert_llama_checkpoint.py ..../LLaMA/7B/consolidated.00.pth
*)

open! Base
module Builder = Xla.Builder
module Ty = Xla.Element_type
module Literal = Xla.Literal
module Op = Xla.Op

let temperature = 1.0
let use_gpu = false
let num_samples = 10
let sample_length = 100
let context_size = 512
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

module VarBuilder : sig
  type t

  val create
    :  builder:Builder.t
    -> var_default_buffer_type:Ty.t
    -> var_default_op_type:Ty.t
    -> t

  val sub : t -> string -> t
  val var : t -> string -> dims:int array -> Op.t
  val arg : t -> string -> ty:Ty.t -> dims:int array -> Op.t
  val load_buffers : t -> filename:string -> device:Xla.Device.t -> Xla.Buffer.t array
end = struct
  module NamedVar = struct
    type t =
      { path : string
      ; ty : Ty.t
      ; dims : int array
      ; is_arg : bool
      }
  end

  type t =
    { vars : NamedVar.t Queue.t
    ; rev_path : string list
    ; builder : Builder.t
    ; var_default_buffer_type : Ty.t
    ; var_default_op_type : Ty.t
    }

  let create ~builder ~var_default_buffer_type ~var_default_op_type =
    { vars = Queue.create ()
    ; rev_path = []
    ; builder
    ; var_default_buffer_type
    ; var_default_op_type
    }

  let sub t dir_name = { t with rev_path = dir_name :: t.rev_path }

  let var_or_arg t name ~ty ~dims ~is_arg =
    let path = List.rev (name :: t.rev_path) |> String.concat ~sep:"." in
    let var = { NamedVar.path; ty; dims; is_arg } in
    let id = Queue.length t.vars in
    let parameter = Op.parameter path ~id ~ty ~dims ~builder:t.builder in
    Queue.enqueue t.vars var;
    parameter

  let var t name ~dims =
    let v = var_or_arg t name ~ty:t.var_default_buffer_type ~dims ~is_arg:false in
    Op.convert v ~ty:t.var_default_op_type

  let arg t name ~ty ~dims = var_or_arg t name ~ty ~dims ~is_arg:true

  let load_buffers t ~filename ~device =
    let npz = Xla.Npy.Npz.open_in filename in
    Exn.protect
      ~f:(fun () ->
        Queue.to_array t.vars
        |> Array.map ~f:(fun (named_var : NamedVar.t) ->
             if named_var.is_arg
             then
               Literal.create ~ty:named_var.ty ~dims:named_var.dims
               |> Xla.Buffer.of_host_literal ~device
             else Xla.Npy.Npz.read_buffer npz named_var.path ~device))
      ~finally:(fun () -> Xla.Npy.Npz.close_in npz)
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

  let config_7b =
    { block_size = 4096; vocab_size = 32000; n_layer = 32; n_head = 32; n_embd = 4096 }

  let _config_13b =
    { block_size = 4096; vocab_size = 32000; n_layer = 40; n_head = 40; n_embd = 5120 }

  let _config_30b =
    { block_size = 4096; vocab_size = 32000; n_layer = 60; n_head = 52; n_embd = 6656 }

  let _config_65b =
    { block_size = 4096; vocab_size = 32000; n_layer = 80; n_head = 64; n_embd = 8192 }
end

module Embedding = struct
  type t = { embeddings : Op.t }

  let create vb ~vocab_size ~n_embd =
    let embeddings = VarBuilder.var vb "weight" ~dims:[| vocab_size; n_embd |] in
    { embeddings }

  let forward t xs = Op.take t.embeddings ~start_indices:xs ~dim_index:0
end

module RmsNorm = struct
  type t =
    { scale : Op.t
    ; dims : int array
    }

  let create vb ~size =
    let scale = VarBuilder.var vb "scale" ~dims:[| size |] in
    { scale; dims = [| 1; 1; size |] }

  let forward t xs =
    let builder = Op.builder xs in
    let eps = Op.r0_f32 1e-5 ~builder |> Op.convert ~ty:(Op.ty xs) in
    let norm_x = Op.reduce_mean (Op.mul xs xs) ~dims:[| -1 |] ~keep_dims:true in
    let x_normed = Op.mul xs (Op.add norm_x eps |> Op.rsqrt) in
    let scale = Op.reshape t.scale ~dims:t.dims in
    Op.mul scale x_normed
end

module Linear = struct
  type t =
    { ws : Op.t
    ; bs : Op.t option
    ; dims : int array
    }

  let create vb ~in_size ~out_size ~with_bias =
    let ws = VarBuilder.var vb "weight" ~dims:[| in_size; out_size |] in
    let bs =
      if with_bias
      then VarBuilder.var vb "bias" ~dims:[| out_size |] |> Option.some
      else None
    in
    { ws; bs; dims = [| 1; 1; out_size |] }

  let forward t xs =
    let rank = Op.rank xs in
    let xs =
      Op.dot_general xs t.ws ~lhs_c:[| rank - 1 |] ~rhs_c:[| 0 |] ~lhs_b:[||] ~rhs_b:[||]
    in
    match t.bs with
    | None -> xs
    | Some bs ->
      let bs = Op.reshape bs ~dims:t.dims in
      Op.add xs bs
end

module CausalSelfAttention = struct
  type t =
    { c_attn : Linear.t
    ; c_proj : Linear.t
    ; n_head : int
    ; n_embd : int
    }

  let create vb ~n_head ~n_embd =
    let sub = VarBuilder.sub vb in
    let c_attn =
      Linear.create (sub "c_attn") ~in_size:n_embd ~out_size:(3 * n_embd) ~with_bias:false
    in
    let c_proj =
      Linear.create (sub "c_proj") ~in_size:n_embd ~out_size:n_embd ~with_bias:false
    in
    { c_attn; c_proj; n_head; n_embd }

  let masked_fill ~mask ~on_true ~on_false =
    let dims = Op.dims mask in
    let on_true = Op.r0_f32 on_true ~builder:(Op.builder mask) |> Op.broadcast ~dims in
    Op.select ~mask ~on_true ~on_false

  let apply_rotary_emb xs ~freqs_cis:fs =
    let dims = Op.dims xs in
    let init_dims = dims in
    let ndims = Array.length dims in
    let dims =
      Array.init (ndims + 1) ~f:(fun i ->
        if i = ndims then 2 else if i = ndims - 1 then dims.(i) / 2 else dims.(i))
    in
    let xs = Op.reshape xs ~dims in
    let re_x = Op.slice_in_dim xs ~start_index:0 ~stop_index:1 ~dim:(-1) in
    let im_x = Op.slice_in_dim xs ~start_index:1 ~stop_index:2 ~dim:(-1) in
    let re_f = Op.slice_in_dim fs ~start_index:0 ~stop_index:1 ~dim:(-1) in
    let im_f = Op.slice_in_dim fs ~start_index:1 ~stop_index:2 ~dim:(-1) in
    let re = Op.sub (Op.mul re_x re_f) (Op.mul im_x im_f) in
    let im = Op.add (Op.mul re_x im_f) (Op.mul im_x re_f) in
    Op.concat_in_dim re [ im ] ~dim_index:(-1) |> Op.reshape ~dims:init_dims

  let forward t xs ~freqs_cis =
    let builder = Op.builder xs in
    let ty = Op.ty xs in
    let freqs_cis = Op.convert freqs_cis ~ty in
    let b_sz, t_sz, c_sz =
      match Op.dims xs with
      | [| b; t; c |] -> b, t, c
      | dims -> [%message "expected 3 dims" (dims : int array)] |> failwith_s
    in
    let qkv = Linear.forward t.c_attn xs in
    let slice_qkv ~start_index ~stop_index =
      Op.slice_in_dim qkv ~start_index ~stop_index ~dim:2
      |> Op.reshape ~dims:[| b_sz; t_sz; t.n_head; c_sz / t.n_head |]
      |> Op.swap_dims ~dim_index1:1 ~dim_index2:2
    in
    let q = slice_qkv ~start_index:(0 * t.n_embd) ~stop_index:(1 * t.n_embd) in
    let k = slice_qkv ~start_index:(1 * t.n_embd) ~stop_index:(2 * t.n_embd) in
    let v = slice_qkv ~start_index:(2 * t.n_embd) ~stop_index:(3 * t.n_embd) in
    let q = apply_rotary_emb q ~freqs_cis in
    let k = apply_rotary_emb k ~freqs_cis in
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
    { c_fc1 : Linear.t
    ; c_fc2 : Linear.t
    ; c_proj : Linear.t
    }

  let create vb ~n_embd =
    let sub = VarBuilder.sub vb in
    let n_hidden = 8 * n_embd / 3 in
    let n_hidden = ((n_hidden - 1) / 256 * 256) + 256 in
    let c_fc1 =
      Linear.create (sub "c_fc1") ~in_size:n_embd ~out_size:n_hidden ~with_bias:false
    in
    let c_fc2 =
      Linear.create (sub "c_fc2") ~in_size:n_embd ~out_size:n_hidden ~with_bias:false
    in
    let c_proj =
      Linear.create (sub "c_proj") ~in_size:n_hidden ~out_size:n_embd ~with_bias:false
    in
    { c_fc1; c_fc2; c_proj }

  let forward t xs =
    Linear.forward t.c_fc1 xs
    |> Op.silu
    |> Op.mul (Linear.forward t.c_fc2 xs)
    |> Linear.forward t.c_proj
end

module Block = struct
  type t =
    { rms_1 : RmsNorm.t
    ; attn : CausalSelfAttention.t
    ; rms_2 : RmsNorm.t
    ; mlp : Mlp.t
    }

  let create vb ~n_head ~n_embd =
    let sub = VarBuilder.sub vb in
    let rms_1 = RmsNorm.create (sub "rms_1") ~size:n_embd in
    let attn = CausalSelfAttention.create (sub "attn") ~n_head ~n_embd in
    let rms_2 = RmsNorm.create (sub "rms_2") ~size:n_embd in
    let mlp = Mlp.create (sub "mlp") ~n_embd in
    { rms_1; attn; rms_2; mlp }

  let forward t xs ~freqs_cis =
    let xs =
      RmsNorm.forward t.rms_1 xs
      |> CausalSelfAttention.forward t.attn ~freqs_cis
      |> Op.add xs
    in
    RmsNorm.forward t.rms_2 xs |> Mlp.forward t.mlp |> Op.add xs
end

module Llama = struct
  type t =
    { lm_head : Linear.t
    ; wte : Embedding.t
    ; blocks : Block.t list
    ; ln_f : RmsNorm.t
    }

  let create vb config =
    let sub = VarBuilder.sub vb in
    let transformer_vs = sub "transformer" in
    let sub_t = VarBuilder.sub transformer_vs in
    let { Config.n_embd; n_head; vocab_size; n_layer; _ } = config in
    let lm_head =
      Linear.create (sub "lm_head") ~in_size:n_embd ~out_size:vocab_size ~with_bias:false
    in
    let wte = Embedding.create (sub_t "wte") ~vocab_size ~n_embd in
    let blocks =
      let vb = sub_t "h" in
      let vb i = VarBuilder.sub vb (Int.to_string i) in
      List.init n_layer ~f:(fun i -> Block.create (vb i) ~n_head ~n_embd)
    in
    let ln_f = RmsNorm.create (sub_t "ln_f") ~size:n_embd in
    { lm_head; wte; blocks; ln_f }

  let forward t xs ~freqs_cis =
    let _b_sz, t_sz =
      match Op.dims xs with
      | [| b; t |] -> b, t
      | dims -> [%message "expected 2 dims" (dims : int array)] |> failwith_s
    in
    List.fold t.blocks ~init:(Embedding.forward t.wte xs) ~f:(fun acc b ->
      Block.forward b acc ~freqs_cis)
    |> RmsNorm.forward t.ln_f
    |> Op.slice_in_dim ~start_index:(t_sz - 1) ~stop_index:t_sz ~dim:1
    |> Linear.forward t.lm_head
end

(* TODO: implement the sentencepiece tokenizer *)
module T : sig
  type t

  val create : config_filename:string -> t
  val encode : t -> string -> int list
  val decode : t -> int list -> string
end = struct
  type t

  let create ~config_filename:_ = failwith "TODO"
  let encode _ _ = failwith "TODO"
  let decode _ _ = failwith "TODO"
end

let precompute_freqs_cis ~config ~builder =
  let n_elem = config.Config.n_embd / config.n_head in
  let theta =
    Array.init (n_elem / 2) ~f:(fun i ->
      1. /. Float.(10000. ** (2. *. Float.of_int i /. Float.of_int n_elem)))
    |> Op.r1_f32 ~builder
  in
  let arange = Array.init context_size ~f:Float.of_int |> Op.r1_f32 ~builder in
  let idx_theta =
    Op.dot_general arange theta ~lhs_c:[||] ~rhs_c:[||] ~lhs_b:[||] ~rhs_b:[||]
  in
  let dims = [| 1; 1; context_size; n_elem / 2; 1 |] in
  let idx_theta_cos = Op.cos idx_theta |> Op.reshape ~dims in
  let idx_theta_sin = Op.sin idx_theta |> Op.reshape ~dims in
  Op.concat_in_dim idx_theta_cos [ idx_theta_sin ] ~dim_index:1

let llama_computation ~config ~b_sz =
  let builder = Builder.create ~name:"llama" in
  let var_default_op_type = if use_gpu then Ty.Bf16 else Ty.F32 in
  let vb = VarBuilder.create ~builder ~var_default_buffer_type:F16 ~var_default_op_type in
  let llama = Llama.create vb config in
  let input = VarBuilder.arg vb "tokens" ~ty:U32 ~dims:[| b_sz; context_size |] in
  let freqs_cis = precompute_freqs_cis ~config ~builder in
  let logits = Llama.forward llama input ~freqs_cis in
  let root =
    Op.div logits (Op.r0_f32 temperature ~builder) |> Op.softmax ~dim_index:(-1)
  in
  Xla.Computation.build ~root, vb

let sample config ~start_tokens ~tokenizer ~exe ~in_buffers =
  let vocab_size = config.Config.vocab_size in
  let tokens = Queue.create () in
  let ba = Bigarray.Array2.create Int32 C_layout 1 context_size in
  for _i = 1 to sample_length do
    for idx = 0 to context_size - 1 do
      let tokens_index = Queue.length tokens - context_size + idx in
      let token =
        if tokens_index < 0
        then (
          let tokens_index = Array.length start_tokens + tokens_index in
          if tokens_index < 0 then 50256 else start_tokens.(tokens_index))
        else Queue.get tokens tokens_index
      in
      ba.{0, idx} <- Int32.of_int_exn token
    done;
    let _ba = Bigarray.genarray_of_array2 ba in
    let buffers = Xla.Executable.execute_b exe in_buffers in
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
  let config = Config.config_7b in
  let client =
    if use_gpu
    then Xla.Client.gpu ~memory_fraction:0.95 ~preallocate:false
    else Xla.Client.cpu ()
  in
  let device = Xla.Client.addressable_devices client |> List.hd_exn in
  Stdio.printf "Platform name: %s\n" (Xla.Client.platform_name client);
  Stdio.printf "Platform version: %s\n%!" (Xla.Client.platform_version client);
  let llama, vb =
    time_it "Generated the op" ~f:(fun () -> llama_computation ~config ~b_sz:1)
  in
  let in_buffers =
    time_it "Load the npz data" ~f:(fun () ->
      VarBuilder.load_buffers vb ~filename:"llama.npz" ~device)
  in
  let exe =
    time_it "Compiled the model" ~f:(fun () -> Xla.Executable.compile client llama)
  in
  let tokenizer = T.create ~config_filename:"llama-tokenizer.json" in
  let start_tokens = T.encode tokenizer start_prompt |> Array.of_list in
  for i = 1 to num_samples do
    time_it "Sampled" ~f:(fun () ->
      let sample = sample config ~start_tokens ~tokenizer ~exe ~in_buffers in
      Stdio.printf "%d ----\n%s\n----\n%!" i sample)
  done
