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

let new_gelu xs =
  let builder = Op.builder xs in
  let f = Op.r0_f32 ~builder in
  let sqrt_two_over_pi = Float.sqrt (2. /. Float.pi) |> f in
  (* 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))) *)
  let vs = Op.mul sqrt_two_over_pi (Op.add xs (Op.mul (Op.pow xs (f 3.)) (f 0.044715))) in
  Op.mul (f 0.5) (Op.mul xs (Op.add (Op.tanh vs) (f 1.)))

module Embedding = struct
  type t = { embeddings : Literal.t }
end

module LayerNorm = struct
  type t =
    { scale : Literal.t
    ; bias : Literal.t
    ; size : int
    }
end

module Linear = struct
  type t =
    { ws : Literal.t
    ; bs : Literal.t option
    ; out_size : int
    }
end

module CausalSelfAttention = struct
  type t =
    { c_attn : Linear.t
    ; c_proj : Linear.t
    ; n_head : int
    ; n_embd : int
    }
end

module Mlp = struct
  type t =
    { c_fc : Linear.t
    ; c_proj : Linear.t
    }
end

module Block = struct
  type t =
    { ln1 : LayerNorm.t
    ; attn : CausalSelfAttention.t
    ; ln2 : LayerNorm.t
    ; mlp : Mlp.t
    }
end

module Gpt = struct
  type t =
    { lm_head : Linear.t
    ; wte : Embedding.t
    ; wpe : Embedding.t
    ; blocks : Block.t list
    ; ln_f : LayerNorm.t
    }
end

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
