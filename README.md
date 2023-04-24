# ocaml-xla
XLA (Accelerated Linear Algebra) bindings for OCaml. This is based on the
[xla-rs](https://github.com/LaurentMazare/xla-rs) Rust bindings, the semantics
for the various operands are documented on the [xla
  website](https://www.tensorflow.org/xla/operation_semantics).

Pre-compiled binaries for the xla library can be downloaded from the
[elixir-nx/xla repo](https://github.com/elixir-nx/xla/releases/tag/v0.4.4).
These should be extracted at the root of this repository, resulting
in a `xla_extension` subdirectory being created, the currently supported version
is 0.4.4.

For a linux platform, this can be done via:
```bash
wget https://github.com/elixir-nx/xla/releases/download/v0.4.4/xla_extension-x86_64-linux-gnu-cpu.tar.gz
tar -xzvf xla_extension-x86_64-linux-gnu-cpu.tar.gz
```

If the `xla_extension` directory is not in the main project directory, the path
can be specified via the `XLA_EXTENSION_DIR` environment variable.

## Generating some Text Samples with LLaMA

The [LLaMA large language model](https://github.com/facebookresearch/llama) can
be used to generate text. The model weights are only available after completing
[this form](https://forms.gle/jk851eBVbX1m5TAv5) and once downloaded can be
converted to a format this package can use. This requires a GPU with 16GB of
memory or 32GB of memory when running on CPU (tweak the `use_gpu` variable in
the example code to choose between CPU and GPU).

```bash
# Download the tokenizer config.
wget https://huggingface.co/hf-internal-testing/llama-tokenizer/raw/main/tokenizer.json -O llama-tokenizer.json

# Extract the pre-trained weights, this requires the transformers and
# safetensors python libraries to be installed.
python examples/convert_llama_checkpoint.py ..../LLaMA/7B/consolidated.00.pth

# Run the example.
dune exec examples/llama.exe
```

## Generating some Text Samples with GPT2 

One of the featured examples is GPT2. In order to run it, one should first
download the tokenization configuration file as well as the weights before
running the example. In order to do this, run the following commands:

```bash
# Download the tokenizer files.
wget https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe

# Extract the pre-trained weights, this requires the transformers python library to be installed.
# This creates a npz file storing all the weights.
python examples/get_gpt2_weights.py

# Run the example.
dune exec examples/nanogpt.exe
```
