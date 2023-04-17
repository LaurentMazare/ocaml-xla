# ocaml-xla
XLA (Accelerated Linear Algebra) bindings for OCaml. This is based on the
[xla-rs](https://github.com/LaurentMazare/xla-rs) Rust bindings.

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

## Generating some Text Samples with GPT2 

One of the featured examples is GPT2. In order to run it, one should first
download the tokenization configuration files as well as the weights before
running the example. In order to do this, run the following commands:

```bash
# Download the tokenizer files.
wget https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe
wget https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json

# Extract the pre-trained weights, this requires the transformers python library to be installed.
# This creates a npz file storing all the weights.
python examples/get_gpt2_weights.py

# Run the example.
dune exec examples/nanogpt.exe
```
