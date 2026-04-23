# ANE-ByteGrid-44M

## Goal

Design a **bidirectional byte-level encoder** from first principles for Apple Neural Engine execution, not for GPU convenience.

The design target is:

- native ANE kernels only for the main path
- fixed-shape tensors only
- no embedding lookup table
- no dynamic causal mask (bidirectional; all positions attend to all)
- high reuse of small working sets in ANE SRAM

**Model type:** Bidirectional encoder (comparable to BERT / ByT5 in function).
Training objective: masked byte prediction (BERT-style, 15% mask rate).
This is *not* a generative/autoregressive model.
A causal extension is possible (see Architecture section) but is left for future work.

This document uses the MIL / IOSurface conventions demonstrated in `maderix/ANE`:

- channel-first tensors packed as `[1, C, 1, S]`
- weights expressed as `conv` or `matmul` constants / packed inputs
- fixed-shape execution through `_ANEInMemoryModel`
- IOSurface-backed I/O with contiguous fp16 buffers

The result is a fixed-length byte-level model that replaces attention with reshape-driven grouped `conv1x1` token mixing.

## Implementation Status

Current repository status on this machine:

- The private ANE runtime path is implemented and verified end to end.
- The staged pipeline compiles `26` kernels:
  - `stem`
  - `24` core blocks
  - `head`
- The verified compile path uses `_ANEInMemoryModelDescriptor modelWithMILText:weights:optionsPlist:` with an in-memory dictionary of staged weight blobs.
- The generated MIL now mirrors the upstream `tmc/apple/x/ane/mil` structure closely:
  - `program(1.3)` plus the `buildInfo` header
  - `@model_path/weights/...` `BLOBFILE(...)` paths
  - `offset = uint64(64)` in MIL, with the actual single-blob payload beginning at byte `128`

Verified local runs:

- `./build/probe_runtime`
  - control sigmoid: compiles
  - exact upstream-style baked `conv`: compiles
  - exact upstream-style dynamic `conv`: compiles
- `./build/train`
  - pipeline ready: `kernels=26 compile=1744.08 ms load=64.39 ms`
  - staged evaluate: `avg=11.209 ms/pass throughput=1140.28 MB/s passes/s=89.21`

Root cause of the earlier false failure:

- The runtime’s `BLOBFILE(...)` parser only matched the no-space form `path=string(...)`.
- Upstream-style MIL uses `path = string(...)`.
- Because of that mismatch, baked conv weights were not being staged into the descriptor, which made valid weighted MIL look like `InvalidMILProgram`.
- After broadening the parser to accept the spaced form, upstream-style conv MIL compiled successfully.

## Executive Summary

Proposed model: `ANE-ByteGrid-44M`

- Sequence length: `S = 256` bytes fixed at compile time
- Vocabulary: raw byte prediction (`256` classes)
- Hidden width: `D = 512`
- Layers: `L = 24`
- Mixer grid: `16 x 16` over the sequence dimension
- Main token mixer: grouped `conv1x1` over reshaped sequence blocks
- Channel mixer: gated MLP using `conv1x1`
- Parameter count: about `44.3M`
- Expected ANE utilization on M5-class ANE: `30-40%` of measured fp16 peak
- Expected speedup vs naive causal transformer on ANE: `2.5-3.5x` at similar parameter count

Why this architecture:

- `conv1x1` is the ANE fast path
- sequence mixing is implemented by moving sequence positions into channels, then applying grouped `conv1x1`
- all shapes are compile-time constants
- no embedding lookup, no attention mask dispatch, no scan primitive

## Hardware-Derived Design Rules

Derived directly from the ANE constraints and the `maderix/ANE` reference implementation:

1. Treat the spatial axis as a fixed token axis.
2. Keep the primary representation in `[1, C, 1, S]`.
3. Express linear maps as `conv1x1` wherever possible.
4. Use `reshape` and `transpose` to move token positions into channels, because ANE is stronger at channel mixing than sequence-time control flow.
5. Avoid any operator sequence that forces CPU participation in the hot path.
6. Size per-layer working sets so activations plus current weights fit comfortably under the observed fast-memory regime rather than chasing max parameter density per layer.

## Research Questions

### 1. Sequence mixing mechanisms that compile natively to ANE conv ops

The best attention alternatives on ANE are:

1. Fixed-block grouped token mixing via `reshape -> grouped conv1x1 -> inverse reshape`
2. Hierarchical local/global token mixing using two different reshape packings
3. Static low-rank token mixers implemented as grouped `conv1x1` on packed token channels

The rejected options:

- Standard causal attention: works, but either needs static mask constants or CPU-side mask handling; it is not the clean ANE fast path.
- True recurrence / scan / Mamba-style selective state updates: poor match for ANE because scan-like data dependence and elementwise state updates are exactly where ANE-native execution becomes fragile.
- Dynamic attention masking: explicitly unsupported as a good native path.

Conclusion:

The primary sequence mixer should be a fixed-shape grouped `conv1x1` token mixer over packed token blocks, not attention.

### 2. Token representation without embedding lookup

Use a tokenizer-free byte representation:

- `256` one-hot byte channels
- `16` byte-class channels
- `32` fixed Fourier position channels
- `16` control channels

Total input channels: `320`

This eliminates:

- vocabulary embedding lookup
- dynamic token-id indexing
- tied embedding / de-embedding tables

The input stem is a learned `conv1x1` from `320 -> 512`.

### 3. Depth/width tradeoff for SRAM reuse

The right tradeoff is narrower and deeper than a GPU-first transformer.

Recommended:

- width `D = 512`
- depth `L = 24`
- MLP expansion `E = 1024`

Why:

- a width of `512` keeps mixer weights small enough for repeated SRAM reuse
- `24` layers preserve total capacity without forcing huge per-layer weight blobs
- global width above `768` starts shifting the design back toward memory-bandwidth-limited execution

### 4. Recurrence / SSM / depthwise conv vs attention

For fixed `S = 256`, a static grouped token mixer is preferable to recurrence.

Reason:

- an SSM can be written as a fixed Toeplitz convolution when `S` is fixed
- once written that way, it becomes another static linear token mixer
- ANE still prefers channel-heavy `conv1x1` over elementwise recurrent state updates

Depthwise temporal conv with kernel size `k > 1` may work, but it is not the most aligned primitive. The better ANE-native version is:

- pack `k` neighboring token slots into channels with fixed reshape/slice logic
- apply grouped `conv1x1`

So the proposed model is effectively a separable conv/state-space surrogate, but expressed entirely in the ANE-friendly `conv1x1` form.

### 5. Training objective

The model is a bidirectional encoder trained with **masked byte prediction** (BERT-style):

- 15% of byte positions per window are randomly selected as mask positions
- At masked positions: byte one-hot (channels 0–255) and class channels (256–271) are zeroed; control channel 10 is set to indicate MASK
- Cross-entropy loss is computed only over masked positions
- The model must use bidirectional context (all other 256 positions visible) to predict the original byte

**Why not next-byte prediction?**
Next-byte prediction is invalid for a bidirectional encoder: the model can trivially copy `b[t+1]` from position `t+1` of its fully visible input, collapsing loss without learning useful representations.

**Dataset:** FineWeb-Edu `sample-10BT` (10B bytes of filtered educational web text), streamed as non-overlapping 256-byte UTF-8 windows.

**Optimizer:** AdamW, β=(0.9, 0.95), lr=3e-4, cosine decay, 2000-step linear warmup, grad clip 1.0, weight decay 0.1.

## Architecture

### Input Encoding

Fixed tensor:

- `x_byte`: `[1, 256, 1, 256]`
- `x_class`: `[1, 16, 1, 256]`
- `x_pos`: `[1, 32, 1, 256]`
- `x_ctrl`: `[1, 16, 1, 256]`

Concatenate:

- `x0 = concat(...) -> [1, 320, 1, 256]`

Channel meaning:

- byte one-hot: exact byte identity
- byte-class: ASCII letter, digit, whitespace, punctuation, UTF-8 lead, UTF-8 continuation, newline, tab, quote, bracket, slash, underscore, symbol, control, BOS, PAD
- position: fixed sin/cos banks for period `2, 4, 8, ..., 65536`
- control: BOS, PAD-validity, document-boundary, domain tags

### Stem

- `StemConv`: `conv1x1 320 -> 512`
- output: `[1, 512, 1, 256]`

### Core Block

Each block has three residual sublayers:

1. Local token mixer
2. Global token mixer
3. Channel GLU mixer

Notation:

- `D = 512`
- `B = 16` local block size
- `G = S / B = 16` blocks
- `E = 1024` channel expansion

### Block Tensor Flow

Input to block `l`:

- `x_l: [1, 512, 1, 256]`

#### Sublayer A: Local Mixer

Purpose:

- mix tokens within each 16-byte chunk

Steps:

1. `u = RMSNorm(x_l)` -> `[1, 512, 1, 256]`
2. `u4 = reshape(u)` -> `[1, 512, 16, 16]`
   - interpreted as `[N, C, chunk, pos_in_chunk]`
3. `ul = transpose(u4, perm=[0,1,3,2])` -> `[1, 512, 16, 16]`
   - interpreted as `[N, C, pos_in_chunk, chunk]`
4. `ulp = reshape(ul)` -> `[1, 8192, 1, 16]`
   - `8192 = 512 * 16`
   - spatial axis is now `chunk`
5. `ulm = grouped conv1x1`
   - input: `[1, 8192, 1, 16]`
   - weight: `[8192, 16, 1, 1]`
   - groups: `512`
   - each group mixes the `16` positions of one feature channel
   - output: `[1, 8192, 1, 16]`
6. inverse reshape / transpose back to `[1, 512, 1, 256]`
7. residual:
   - `x'_l = x_l + alpha_local * local_out`

#### Sublayer B: Global Mixer

Purpose:

- mix across the 16 chunks for each intra-chunk position

Steps:

1. `v = RMSNorm(x'_l)` -> `[1, 512, 1, 256]`
2. `v4 = reshape(v)` -> `[1, 512, 16, 16]`
   - interpreted as `[N, C, chunk, pos_in_chunk]`
3. `vgp = reshape(v4)` -> `[1, 8192, 1, 16]`
   - here the spatial axis is `pos_in_chunk`
   - channels bundle the `16` chunks
4. `vgm = grouped conv1x1`
   - input: `[1, 8192, 1, 16]`
   - weight: `[8192, 16, 1, 1]`
   - groups: `512`
   - output: `[1, 8192, 1, 16]`
5. inverse reshape back to `[1, 512, 1, 256]`
6. residual:
   - `x''_l = x'_l + alpha_global * global_out`

After local plus global mixing, every token has a full-sequence receptive field:

- local mixer connects tokens inside each chunk
- global mixer connects same-position tokens across chunks
- stacking blocks propagates information densely across all `256` positions

#### Sublayer C: Channel GLU

Purpose:

- nonlinear channel mixing while staying on the `conv1x1` fast path

Steps:

1. `w = RMSNorm(x''_l)` -> `[1, 512, 1, 256]`
2. value projection:
   - `wv = conv1x1 512 -> 1024`
   - output: `[1, 1024, 1, 256]`
3. gate projection:
   - `wg = conv1x1 512 -> 1024`
   - output: `[1, 1024, 1, 256]`
4. `gate = sigmoid(wg)` -> `[1, 1024, 1, 256]`
5. `mix = gate * wv` -> `[1, 1024, 1, 256]`
6. output projection:
   - `wo = conv1x1 1024 -> 512`
   - output: `[1, 512, 1, 256]`
7. residual:
   - `x_{l+1} = x''_l + alpha_mlp * wo`

### Output Head

1. `xL = RMSNorm(x_24)` -> `[1, 512, 1, 256]`
2. `logits = conv1x1 512 -> 256` -> `[1, 256, 1, 256]`
3. CPU loss:
   - byte-level cross entropy over valid positions only

No output embedding matrix is required.

## Full Shape Table

### Model Input / Output

| Stage | Tensor | Shape |
|---|---|---|
| Byte one-hot | `x_byte` | `[1, 256, 1, 256]` |
| Byte-class features | `x_class` | `[1, 16, 1, 256]` |
| Position features | `x_pos` | `[1, 32, 1, 256]` |
| Control features | `x_ctrl` | `[1, 16, 1, 256]` |
| Concatenated input | `x0` | `[1, 320, 1, 256]` |
| Stem output | `x1` | `[1, 512, 1, 256]` |
| Block output | `x_{l+1}` | `[1, 512, 1, 256]` |
| Final normalized | `xL` | `[1, 512, 1, 256]` |
| Logits | `logits` | `[1, 256, 1, 256]` |

### Inside One Block

| Stage | Tensor | Shape |
|---|---|---|
| Block input | `x_l` | `[1, 512, 1, 256]` |
| Local norm | `u` | `[1, 512, 1, 256]` |
| Local pack | `ulp` | `[1, 8192, 1, 16]` |
| Local mixed | `ulm` | `[1, 8192, 1, 16]` |
| After local residual | `x'_l` | `[1, 512, 1, 256]` |
| Global norm | `v` | `[1, 512, 1, 256]` |
| Global pack | `vgp` | `[1, 8192, 1, 16]` |
| Global mixed | `vgm` | `[1, 8192, 1, 16]` |
| After global residual | `x''_l` | `[1, 512, 1, 256]` |
| Channel norm | `w` | `[1, 512, 1, 256]` |
| Value proj | `wv` | `[1, 1024, 1, 256]` |
| Gate proj | `wg` | `[1, 1024, 1, 256]` |
| GLU product | `mix` | `[1, 1024, 1, 256]` |
| Channel out | `wo` | `[1, 512, 1, 256]` |
| Block output | `x_{l+1}` | `[1, 512, 1, 256]` |

## Parameter Count

### Stem

- `320 x 512 = 163,840`

### Per Block

Local grouped mixer:

- `512 groups * 16 * 16 = 131,072`

Global grouped mixer:

- `512 groups * 16 * 16 = 131,072`

Channel GLU:

- value projection: `512 x 1024 = 524,288`
- gate projection: `512 x 1024 = 524,288`
- output projection: `1024 x 512 = 524,288`
- subtotal: `1,572,864`

Norm scales:

- `3 x 512 = 1,536`

Per block total:

- `131,072 + 131,072 + 1,572,864 + 1,536 = 1,836,544`

### 24 Blocks

- `24 x 1,836,544 = 44,077,056`

### Final Head

- final RMSNorm: `512`
- classifier: `512 x 256 = 131,072`

### Total

- stem: `163,840`
- blocks: `44,077,056`
- head: `131,584`

Total:

- `44,372,480` parameters

Rounded:

- `44.3M`

## Why This Fits ANE Better Than A Naive Transformer

### Naive Transformer Baseline

Reference baseline for comparison:

- `24` layers
- `D = 512`
- byte embeddings
- `8` heads
- causal self-attention
- FFN expansion `4x`

Main ANE problems:

- attention score path has poor SRAM locality at `S x S`
- causal masking is awkward on ANE
- embedding lookup is not naturally a `conv1x1`
- many small elementwise / reshape-heavy stages reduce realized throughput
- utilization in the `maderix/ANE` results is typically around `5-10%` with transformer-style training graphs

### Proposed Model Advantages

- no Q/K/V projections
- no `S x S` attention score matrix in the main path
- no dynamic mask logic
- no embedding lookup
- token mixing collapses to two highly regular grouped `conv1x1` kernels per block
- channel mixing is pure `conv1x1`
- the hot path is dominated by large, regular linear kernels instead of mixed control-heavy logic

## SRAM / Bandwidth Analysis

Approximate fp16 activation sizes:

- main hidden state `[1,512,1,256]`: `512 * 256 * 2 = 262,144 B = 256 KB`
- packed local/global tensor `[1,8192,1,16]`: same storage, also `256 KB`
- GLU expansion `[1,1024,1,256]`: `524,288 B = 512 KB`

Approximate fp16 weights per block:

- local mixer: `131,072 * 2 = 256 KB`
- global mixer: `256 KB`
- channel GLU: `1,572,864 * 2 = 3.0 MB`

Peak live working set for the active sublayer stays well below the rough `~8 MB` fast-memory budget:

- hidden activation: `256 KB`
- expansion activation: `512 KB`
- active weights: up to `~3.0 MB`
- scratch / norm / gating / output buffers: comfortably under the remainder

That is the main reason to choose `D = 512` and `E = 1024` instead of wider blocks.

## Estimated Compute And Utilization

These are model-side estimates, not measured numbers.

### Forward FLOPs Per Block

Local grouped mixer:

- `2 * 512 * 16 * 16 * 16 = 4.19M`

Global grouped mixer:

- `4.19M`

Channel GLU:

- value proj: `2 * 512 * 1024 * 256 = 268.44M`
- gate proj: `268.44M`
- output proj: `268.44M`
- subtotal: `805.31M`

Per block forward:

- about `813.7M` FLOPs

### Full Forward FLOPs

- `24 * 813.7M = 19.53G`
- stem + head add about `0.15G`

Total forward:

- about `19.7 GFLOPs`

### Training Estimate

Assume forward + backward + optimizer-equivalent compute at about `3.2x` forward:

- about `63 GFLOPs / step`

### Utilization Estimate

Using measured M4-class fp16 peak from the reference repo:

- measured peak fp16 throughput on M4 Pro: `12.57 TFLOPS`
- naive transformer-style training graphs in the repo: about `1.28 TFLOPS`, roughly `8-10%`

Expected for this model, by inference:

- sustained effective throughput: `3.8-5.0 TFLOPS`
- utilization: about `30-40%` of measured fp16 peak

Reason for the improvement:

- the model turns almost the entire hot path into regular `conv1x1` workloads with fixed tensor shapes and minimal CPU participation

## Comparison Table

| Item | Naive Byte Transformer | ANE-ByteGrid-44M |
|---|---|---|
| Tokenization | embedding lookup | one-hot byte channels |
| Sequence length | fixed or padded | fixed `256` |
| Main mixer | attention | local/global grouped `conv1x1` |
| Causal masking | awkward, often CPU-assisted | not needed |
| Main hot-path ops | matmul + softmax + masking + many elementwise ops | `conv1x1` + reshape + transpose + sigmoid |
| Activation layout | often transpose-heavy | native `[1,C,1,S]` |
| Param count target | similar | `44.3M` |
| Estimated utilization | `8-12%` | `30-40%` |
| Expected ANE speed | baseline | `2.5-3.5x` faster |

## MIL Generation Patterns

The snippets below follow the style used in `maderix/ANE`, especially:

- `program(1.3)`
- `func main<ios18>(tensor<fp16, ...> x)`
- `conv(...)`
- `reshape`, `transpose`, `concat`

### 1. Stem Projection

```objc
static NSString *gen_stem_mil(void) {
    return
    @"program(1.3)\n"
    "{\n"
    "  func main<ios18>(tensor<fp16, [1, 320, 1, 256]> x) {\n"
    "    string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
    "    tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
    "    tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
    "    tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
    "    int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
    "    tensor<fp16, [512, 320, 1, 1]> W = const()[name=string(\"W\"), "
    "      val=tensor<fp16, [512, 320, 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/stem.bin\"), offset=uint64(64)))];\n"
    "    tensor<fp16, [1, 512, 1, 256]> y = conv(dilations=dl, groups=gr, pad=pd, "
    "      pad_type=pt, strides=st, weight=W, x=x)[name=string(\"stem\")];\n"
    "  } -> (y);\n"
    "}\n";
}
```

### 2. Local Token Mixer

This is the core ANE-native attention replacement.

```objc
static NSString *gen_local_mixer_mil(void) {
    return
    @"program(1.3)\n"
    "{\n"
    "  func main<ios18>(tensor<fp16, [1, 512, 1, 256]> x) {\n"
    "    tensor<int32, [4]> sh0 = const()[name=string(\"sh0\"), val=tensor<int32, [4]>([1,512,16,16])];\n"
    "    tensor<fp16, [1,512,16,16]> x4 = reshape(shape=sh0, x=x)[name=string(\"x4\")];\n"
    "    tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"
    "    tensor<fp16, [1,512,16,16]> xt = transpose(perm=pm, x=x4)[name=string(\"xt\")];\n"
    "    tensor<int32, [4]> sh1 = const()[name=string(\"sh1\"), val=tensor<int32, [4]>([1,8192,1,16])];\n"
    "    tensor<fp16, [1,8192,1,16]> xp = reshape(shape=sh1, x=xt)[name=string(\"xp\")];\n"
    "    string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
    "    tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
    "    tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
    "    tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
    "    int32 gr = const()[name=string(\"gr\"), val=int32(512)];\n"
    "    tensor<fp16, [8192, 16, 1, 1]> W = const()[name=string(\"W\"), "
    "      val=tensor<fp16, [8192,16,1,1]>(BLOBFILE(path=string(\"@model_path/weights/local.bin\"), offset=uint64(64)))];\n"
    "    tensor<fp16, [1,8192,1,16]> ym = conv(dilations=dl, groups=gr, pad=pd, "
    "      pad_type=pt, strides=st, weight=W, x=xp)[name=string(\"mix\")];\n"
    "    tensor<fp16, [1,512,16,16]> y4 = reshape(shape=sh0, x=ym)[name=string(\"y4\")];\n"
    "    tensor<fp16, [1,512,16,16]> yi = transpose(perm=pm, x=y4)[name=string(\"yi\")];\n"
    "    tensor<int32, [4]> sho = const()[name=string(\"sho\"), val=tensor<int32, [4]>([1,512,1,256])];\n"
    "    tensor<fp16, [1,512,1,256]> y = reshape(shape=sho, x=yi)[name=string(\"out\")];\n"
    "  } -> (y);\n"
    "}\n";
}
```

### 3. Global Token Mixer

```objc
static NSString *gen_global_mixer_mil(void) {
    return
    @"program(1.3)\n"
    "{\n"
    "  func main<ios18>(tensor<fp16, [1, 512, 1, 256]> x) {\n"
    "    tensor<int32, [4]> sh0 = const()[name=string(\"sh0\"), val=tensor<int32, [4]>([1,512,16,16])];\n"
    "    tensor<fp16, [1,512,16,16]> x4 = reshape(shape=sh0, x=x)[name=string(\"x4\")];\n"
    "    tensor<int32, [4]> sh1 = const()[name=string(\"sh1\"), val=tensor<int32, [4]>([1,8192,1,16])];\n"
    "    tensor<fp16, [1,8192,1,16]> xp = reshape(shape=sh1, x=x4)[name=string(\"xp\")];\n"
    "    string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
    "    tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
    "    tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
    "    tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
    "    int32 gr = const()[name=string(\"gr\"), val=int32(512)];\n"
    "    tensor<fp16, [8192, 16, 1, 1]> W = const()[name=string(\"W\"), "
    "      val=tensor<fp16, [8192,16,1,1]>(BLOBFILE(path=string(\"@model_path/weights/global.bin\"), offset=uint64(64)))];\n"
    "    tensor<fp16, [1,8192,1,16]> ym = conv(dilations=dl, groups=gr, pad=pd, "
    "      pad_type=pt, strides=st, weight=W, x=xp)[name=string(\"mix\")];\n"
    "    tensor<int32, [4]> sho = const()[name=string(\"sho\"), val=tensor<int32, [4]>([1,512,1,256])];\n"
    "    tensor<fp16, [1,512,1,256]> y = reshape(shape=sho, x=ym)[name=string(\"out\")];\n"
    "  } -> (y);\n"
    "}\n";
}
```

### 4. Channel GLU

```objc
static NSString *gen_channel_glu_mil(void) {
    return
    @"program(1.3)\n"
    "{\n"
    "  func main<ios18>(tensor<fp16, [1, 512, 1, 256]> x) {\n"
    "    string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
    "    tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
    "    tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
    "    tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
    "    int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
    "    tensor<fp16, [1024, 512, 1, 1]> Wv = const()[name=string(\"Wv\"), "
    "      val=tensor<fp16, [1024,512,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wv.bin\"), offset=uint64(64)))];\n"
    "    tensor<fp16, [1024, 512, 1, 1]> Wg = const()[name=string(\"Wg\"), "
    "      val=tensor<fp16, [1024,512,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wg.bin\"), offset=uint64(64)))];\n"
    "    tensor<fp16, [512, 1024, 1, 1]> Wo = const()[name=string(\"Wo\"), "
    "      val=tensor<fp16, [512,1024,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wo.bin\"), offset=uint64(64)))];\n"
    "    tensor<fp16, [1,1024,1,256]> v = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=Wv, x=x)[name=string(\"v\")];\n"
    "    tensor<fp16, [1,1024,1,256]> g = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=Wg, x=x)[name=string(\"g\")];\n"
    "    tensor<fp16, [1,1024,1,256]> s = sigmoid(x=g)[name=string(\"sig\")];\n"
    "    tensor<fp16, [1,1024,1,256]> m = mul(x=v, y=s)[name=string(\"mul\")];\n"
    "    tensor<fp16, [1,512,1,256]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=Wo, x=m)[name=string(\"out\")];\n"
    "  } -> (y);\n"
    "}\n";
}
```

## IOSurface I/O Layout

The reference repo packs ANE tensors contiguously in channel-first order. The same rule should be used here.

For `[1, 512, 1, 256]` fp16:

- total bytes: `512 * 256 * 2 = 262,144`
- element order: `buf[c * 256 + t]`

Example:

```objc
for (int t = 0; t < 256; t++) {
    for (int c = 0; c < 512; c++) {
        dst[c * 256 + t] = src[t * 512 + c];
    }
}
```

This exactly matches the packing pattern used throughout `maderix/ANE`.

## Training Plan

### Objective

- **Masked byte prediction** (BERT-style, 15% mask rate) — see Section 5 above
- Cross-entropy loss computed only on masked positions
- No next-byte prediction; model is bidirectional and cannot use causal objectives

### Optimizer

- AdamW
- `beta1 = 0.9`
- `beta2 = 0.95`
- `weight_decay = 0.1`
- gradient clipping `1.0`

### Batch / Sequence

- compile-time sequence: `256`
- train with always-fixed `[1, C, 1, 256]`
- shorter samples are right-padded to 256 with `0x00`
- a valid-token control channel distinguishes pad from real data
- loss masking runs on CPU (outside the hot ANE path)

### Schedule

1. Warmup: `2k` steps linear ramp to peak lr `3e-4`
2. Main pretraining: cosine decay to `3e-5` over `500k` steps
3. (~20 hours on MPS at batch=16)

### Dataset

- FineWeb-Edu `sample-10BT` — 10B bytes of filtered educational web text
- Streamed as non-overlapping 256-byte UTF-8 windows

### Curriculum

Phase 1:

- windows mostly `32` and `64`, padded to `256`
- goal: local syntax, UTF-8, spelling, punctuation

Phase 2:

- windows mostly `128`
- goal: sentence and paragraph continuation

Phase 3:

- windows mostly `256`
- goal: full-context behavior at deployment length

## Deployment Variants

Because shapes are fixed, sequence length variants should be compiled as separate models:

- `S=128`: lower latency
- `S=256`: default
- `S=512`: larger context, same architecture with grid `16 x 32` or `32 x 16`

Do not try to make one compiled graph handle all lengths dynamically.

## Final Recommendation

If the target is ANE-first language modeling, the architecture should not be a transformer with ANE workarounds. It should be:

- byte-level
- fixed-shape
- `conv1x1` dominated
- sequence-mixed by static reshape/grouped-conv patterns

The recommended model is therefore:

- `ANE-ByteGrid-44M`
- `24` layers
- `512` hidden width
- `1024` GLU expansion
- fixed `256`-byte context
- local/global grouped token mixing in place of attention

This is the cleanest architecture I can justify from the observed ANE execution model and the MIL generation techniques shown in `maderix/ANE`.
