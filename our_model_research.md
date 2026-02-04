Designing a High-Capacity GOOM-Based LLM for a 32GB GPU
Overview of the GOOM-SSM RNN Architecture

The Generalized Orders of Magnitude (GOOM) approach enables a novel recurrent architecture that can capture very long-range dependencies in sequences without numerical instability. Instead of the typical Transformer attention, our model will use a deep RNN built on state-space models (SSMs) with GOOM-based parallel recurrence. In each recurrent layer, the time-step update is defined by a non-diagonal linear state-space system:

𝑥
𝑡
=
𝐴
 
𝑥
𝑡
−
1
+
𝐵
 
𝑢
𝑡
,
x
t
	​

=Ax
t−1
	​

+Bu
t
	​

,

𝑦
𝑡
=
𝐶
 
𝑥
𝑡
+
𝐷
 
𝑢
𝑡
,
y
t
	​

=Cx
t
	​

+Du
t
	​

,

where $u_t$ is the input at time $t$, $x_t$ is the hidden state, and $y_t$ the output. Importantly, all time steps are computed in parallel via a prefix-scan algorithm over the sequence, rather than sequentially. GOOM makes this possible by representing numbers in a complex logarithmic domain, vastly extending numeric dynamic range and preventing overflow/underflow during long recurrences. Essentially, each real value is stored as a complex number whose exponentiation yields the real value (log representation), including sign encoding via multiples of $i\pi$. This means the RNN’s internal state can grow or shrink exponentially over long sequences without numerical overflow or vanishing – any extreme magnitudes are handled by the log scale. The authors report that using complex-valued GOOM states renders stabilization mechanisms (like gradient clipping or forget-gates) unnecessary even for very long sequences. In other words, the model can learn long-term dependencies without exploding or vanishing gradients, unlike traditional RNNs.

Each recurrent layer computes its whole sequence update in two phases: (1) Apply the state update in GOOM-space – this involves taking the log of matrices and vectors and using a custom log_matmul_exp operation to combine them in the log domain. A parallel prefix-scan (via torch_parallel_scan) then efficiently composes these updates across the sequence dimension in $O(\log n)$ steps (conceptually) while leveraging GPU parallelism. (2) Map back to real space for the output: the hidden state $x_t$ computed in log space is exponentiated back to a real tensor (with appropriate scaling to fit in float range) and used to compute the layer’s output $y_t = Cx_t + D u_t$.

GOOM’s effect: By working in the log domain internally, this method can represent extraordinarily large or tiny values far beyond standard float32 range (e.g. complex64 GOOM covers magnitudes up to ~exp(10^38)!). This gives the RNN the ability to handle very long sequences and deep recurrences without numerical issues. In fact, the GOOM RNN was demonstrated to capture long-range dependencies with non-diagonal recurrences in parallel, with no need for explicit stabilization or gating. This is a significant improvement over prior state-space models like S4, which restricted $A$ to diagonal or required special initialization for stability. Here, $A$ can be a full (non-diagonal) matrix, enhancing expressiveness. The authors note their 124M-parameter GOOM RNN achieved a validation cross-entropy of ~2.7 on language modeling (1024-token sequences from The Pile) after 10B tokens of training. Although this is slightly behind comparable-size Transformers (around 2.4 loss when trained on more data), it shows the RNN can scale up in performance as data or model size increases.

Layer structure: Our model will stack multiple Residual Recurrent Layers (similar to Transformer blocks). Each such layer will:

Apply Layer Normalization to the input (pre-normalization) for stable training.

Pass the normalized sequence through the SSM-over-GOOM module (the parallel RNN) to produce an output of dimension 2×d_emb per token. This doubling is intentional to feed a gated linear unit.

Apply a GLU (Gated Linear Unit) activation: splitting the 2*d_emb output into two halves, using one half to gate the other, then a linear projection back to d_emb. This acts like the feed-forward network of a Transformer block, introducing non-linearity and mixing features. Gating (as in GLU or SwiGLU) is known to improve performance in large language models by allowing multiplicative interactions in the layer’s output.

Add the result back to the layer’s input (residual connection).

Every layer’s SSM uses its own learned parameters $A, B, C, D$ and an initial hidden state vector. Notably, in the reference implementation each layer’s $A$ is initialized as a near-orthogonal $d_{\text{hid}}\times d_{\text{hid}}$ matrix (with gain ~0.99 for stability), and shared across all “heads” of that layer. Here “heads” refers to splitting the hidden state into n_hid independent vectors of length d_hid each (so total hidden size per token = n_hid * d_hid = d_emb). Each head’s portion of the state evolves with the same $A$ matrix (shared dynamics), but the input projection $B$ and output projection $C$ are different for each head (they are effectively block matrices spanning all heads). This design is analogous to multi-head attention in that it allows multiple parallel subspaces for the model to encode information, though here the heads are within an RNN state rather than separate attention heads. We will retain this “multi-head RNN” design (with possibly even more heads/features to scale up capacity).

No positional embeddings needed: Unlike Transformers, this RNN does not require a separate positional encoding matrix. The sequential order is inherently captured by the state update (earlier tokens influence later ones through the state $x_t$). The model learns its own notion of position via the dynamics of $A$ and the recurrent state. This simplifies the architecture and lets us flexibly handle varying sequence lengths.

Model Configuration for Maximal Capacity on 32GB VRAM

Our goal is to maximize the model’s size and capability while remaining trainable on a single RTX 5090 (32 GB). We will therefore choose the largest dimensions that fit memory, and leverage optimizations to use that memory effectively.

Base dimensions: The reference model used d_emb=768, n_res=24 layers, n_hid=24 heads, d_hid=32, giving ~124M params. We aim to push much higher. Specifically, we can increase:

Embedding size (d_emb): A larger embedding (and hidden) size directly increases model expressiveness. Values like 1024, 1536, or 2048 are reasonable targets. For instance, d_emb=1536 (with corresponding hidden state 1536 per token) significantly increases the network’s capacity (embedding and output layers scale with this, as do the RNN layer parameters).

Number of residual layers (n_res): Stacking more layers increases model depth. We might expand to 32, 48, or even 64 layers if memory permits. Depth generally improves learning of complex functions, up to diminishing returns.

Hidden state decomposition (n_hid and d_hid): We should maintain n_hid * d_hid = d_emb. The reference chose many heads (24) with smaller each (32) – we can continue that pattern (e.g. 48 heads × 32 features = 1536, or even 64×32=2048). Using a moderate $d_{\text{hid}}$ (like 32 or 64) keeps the matrix $A$ size small (which is good for speed), while n_hid grows to fill the total dimension. Many heads means more parallel “sub-states” per token.

Vocabulary size: We’ll use a token vocabulary around 50k, similar to GPT-2’s BPE tokenizer. Unless we need multi-lingual support or specialized text, 50k subword tokens is sufficient for broad English text. We can reuse the GPT-2 tokenizer (via TikToken or Huggingface’s implementation) so that we leverage a well-tested segmentation. (This avoids the need to train a new tokenizer from scratch, and GPT-2's vocab covers The Pile and similar corpora well.) If we anticipate needing Unicode or multi-language, we might consider a SentencePiece unigram model or a larger vocab (~100k tokens), but that will also increase the embedding matrix size. For now, we’ll stick to ~50k and tie the input embedding and output projection weights (weight tying) to save parameters and improve consistency.

With these choices, we estimate the model size. For example, a configuration of d_emb=1536, n_res=48 layers, n_hid=48, d_hid=32 (48 heads of size 32) yields roughly ~1 billion parameters. This includes ~77M from embeddings (50k×1536) and on the order of 14M per layer (for $B, C, D$, and the GLU linear). 48 layers × 14M ≈ 672M, plus embeddings ~77M, totals ~749M, plus other small params (layer norms, etc.), ending up in the 0.8–0.9B range. If we instead use 64 layers, we’d exceed 1B. We should fine-tune these numbers experimentally to hit the sweet spot of maximum size that still leaves room for training memory. A model on the order of 0.5–1.0 billion parameters is likely achievable on 32GB with the right optimizations.

Memory considerations: Storing 1B parameters in 16-bit precision requires ~2 GB. Gradients and optimizer states add overhead: using AdamW, we have two moment tensors per parameter in FP32 by default, which would add ~8 GB. This sums to ~10 GB just for parameters+optimizer. The remainder (~22 GB) must cover activations, temporary tensors, and the complex-valued buffers used by GOOM computations. We anticipate roughly 2× memory overhead for GOOM operations (each number as complex64 uses 2×32-bit). To manage this:

We will use mixed precision training (fp16/bf16) for all real tensors. PyTorch’s autocast can convert most operations to FP16 on the fly, and we’ll use GradScaler to avoid underflow in gradients. (The reference already verified that autocasting to float16 for the non-complex parts works well.) The GOOM operations on complex tensors will remain at torch.complex64 (since PyTorch doesn’t support lower precision for complex yet), but those are only in the recurrent state updates. The rest (embeddings, linear layers, etc.) will benefit from half-precision, cutting their memory use in half.

We will enable torch.compile() on the model if using PyTorch 2.x, to optimize execution. Although PyTorch’s JIT/compiler can’t fully optimize across complex ops, it still significantly reduces overhead on the float32 parts, improving speed and memory efficiency. The GOOM authors report that compiling their model (except the complex ops) yielded faster training and lower memory usage despite some warnings.

We will employ gradient checkpointing on the residual layers. This means instead of storing every intermediate activation for backprop, we selectively drop them and recompute the forward pass of a few layers during backward pass. This can dramatically reduce activation memory at the cost of extra compute. Given “peak accuracy” is our priority, we accept a bit more computation to enable a larger model. PyTorch’s torch.utils.checkpoint can wrap each ResidualRecurrentLayer so that its internal activations (e.g., the large log_cum_A_atop_Bu tensor from the prefix scan) need not be kept in memory for all layers simultaneously.

To handle the large optimizer states, we can use an 8-bit optimizer (like bitsandbytes’ LION or 8-bit Adam). For example, 8-bit Adam quantizes the moment tensors to int8 with minimal loss in model quality, reducing optimizer memory by ~75%. This could save ~6 GB when using a 1B model, freeing more VRAM for the model and batch. Alternatively, we could keep Adam’s state on CPU memory (DeepSpeed ZeRO-Offload style), but that would slow training. An 8-bit in-GPU solution is preferable to maintain speed.

We will accumulate gradients over micro-batches to reach a high effective batch size without blowing memory. The reference handled 960 sequences per batch by splitting into the largest micro-batch that fits in GPU and accumulating. We will do similarly. For instance, if only 4 sequences of length 1024 fit in memory at once for our big model, we can do 4 at a time and accumulate gradients over 32 iterations to effectively train on 128 sequences before updating weights. A larger effective batch helps training stability for large models. We will experiment to find the largest micro-batch that 32GB can handle and set the gradient accumulation factor accordingly. (This might be on the order of 128–512 total tokens per batch if the model is ~1B, given memory constraints.)

Sequence length: We want the longest possible context the model can handle, since longer context = more capability to retain information and handle long conversations or documents. One advantage of our RNN approach is that computational cost grows linearly with sequence length (O(n)), unlike Transformers which are O(n²) with respect to context. This means we can potentially use very long sequences during training. However, memory does scale linearly with sequence length as well (because the prefix scan returns a full sequence of states). In the reference, they trained with sequence length 1024 tokens. On 32GB, with our larger model, we should be able to at least maintain 1024 or push to 2048 tokens context if the batch size is adjusted. For example, doubling sequence length to 2048 doubles the memory for storing activations of the recurrent state and outputs. We can likely compensate by halving the micro-batch size. If throughput remains acceptable, we prefer the 2048 sequence length for training. This would give the model a very large attention span (useful for long dialogues or documents). We will also incorporate variable length training (packing shorter sequences and some long sequences) to expose the model to a range of lengths up to the max, which helps it not degrade on short inputs and also learn to utilize long contexts. Because the RNN has no fixed positional embedding, it should generalize to any sequence length up to what it has seen. In theory, it could extrapolate to lengths beyond training (since the recurrence is the same for each additional step), but to be safe we will set the max train length equal to our target inference length.

Tokenization notes: As mentioned, we plan to use a byte-pair encoding (BPE) tokenizer such as the GPT-2 50k vocabulary. This is a good balance between granularity and vocabulary size for English text. It will yield subword tokens that handle common words efficiently while still being able to represent rare words or sequences of characters. We will include standard special tokens as needed (e.g., an end-of-sequence token <|endoftext|>). A careful tokenization choice is important because it affects the sequence length and the model’s understanding of text. Using an established tokenizer ensures we have reasonable token distribution (for example, GPT-2’s tokenizer has been used successfully for large corpora). If desired, we could fine-tune the tokenizer or use a SentencePiece Unigram model (like used in LLaMA or T5) for potentially slightly better compression, but the improvement may be minor. Sticking with a known tokenizer also allows leveraging pre-tokenized datasets like The Pile.

Training Strategy and Hyperparameters

Data and Training Length: We assume access to a large text dataset (the user mentions an arbitrarily large amount of data). To fully realize the model’s potential, we should follow scaling law guidelines. Recent studies (e.g., Hoffmann et al. Chinchilla) suggest training on roughly 20 tokens per parameter for a compute-optimal training. For a 1B-parameter model, that implies on the order of 20 billion tokens. If our model is ~800M params, ~16B tokens would be a target. In practice, this could be achieved by multiple epochs over a multi-billion-token corpus (The Pile, CommonCrawl, etc.). For example, The Pile has ~300B tokens (though not all high-quality); even a subset can provide tens of billions. We will likely train for at least several epochs until the loss stops appreciably decreasing. The reference trained 124M on 10B tokens; a larger model should have a higher data requirement for optimal performance. We will monitor validation metrics to decide when to stop (or use the scaling law estimate as a target).

Optimizer: We will use AdamW as in the reference, since it’s a robust choice for language models. We’ll apply a weight decay of 0.1 (which is relatively high, but the reference found it beneficial, likely as a form of regularization in lieu of dropout). Weight decay helps prevent the model weights from growing too large. We will disable weight decay on certain parameters like LayerNorm gains and biases, embedding weights, and the RNN’s internal state parameters, per standard practice (the reference provides a helper for splitting decay vs no-decay params which we’ll adopt).

Learning rate schedule: A One-Cycle LR schedule was used in the reference, and we can adopt a similar approach. This involves: a short warm-up from a low LR to the peak LR, then a gradual decay toward a final low LR. The reference model used a peak of 3e-4 and end at 1e-5. For a larger model, we might choose a slightly lower peak LR to maintain stability (large models often use ~1–2e-4). For instance, we can warm up over the first ~1% of training to lr_max ~2.5e-4, then cosine decay or OneCycle down to ~1e-5 by the end. The momentum (for AdamW, this relates to beta1) can follow the OneCycle policy as well (beta1 from 0.85 to 0.99 as they did). We will closely watch training; if we see signs of instability (loss spikes or NaNs), we may reduce LR or use gradient clipping. (Gradient clipping might not be needed thanks to GOOM’s stability, but it’s a safety net in case of any unexpected divergence).

Batch size and gradient accumulation: As discussed, we will use micro-batches with accumulation. For example, we might set an effective batch of, say, 512 sequences × 1024 tokens = 524k tokens per step. But we’ll achieve that via smaller chunks. If our GPU can only fit 8 sequences (1024 tokens each) at once, we’d accumulate gradients over 64 iterations (8×64=512) before updating. We will use PyTorch’s gradient accumulation (simply not zeroing the grad until after N mini-steps). We also ensure to shuffle data and break it into epochs in a way that each epoch covers the dataset once; within an epoch, we iterate with a DataLoader that yields micro-batches. We will print out the effective batch tokens/sec to monitor utilization. With 32GB and a highly optimized code, we hope to achieve a reasonable throughput (it will be slower than a smaller model, but we’ll aim to saturate the GPU compute).

Regularization: Besides weight decay, we can consider dropout in the model if needed. The reference implementation did not use dropout layers in the RNN or embedding – and still didn’t overfit on 10B tokens (if anything, it underfitted given the gap to theoretical 2.4 loss). With an even larger dataset and weight decay, we likely do not need dropout. However, if we observe overfitting on a validation set (validation loss starting to increase while training loss decreases), we could introduce a small dropout (e.g. 0.1) on the outputs of the SSM or on the feed-forward output. This is something we’ll monitor; initial training will probably be fine without it.

Evaluation and validation: We will hold out a portion of data as a validation set to periodically measure perplexity. Since this is a language model, the primary metric is cross-entropy loss / perplexity on validation data. We can compute this after each epoch (or every N training steps) by running the model in evaluation mode on a sample of validation tokens. A downward-trending validation loss will confirm the model is improving. We expect perplexity to decrease significantly over the course of training. (For reference, the 124M model got to ~perplexity 15 on The Pile – since cross-entropy ~2.7 means perplexity = exp(2.7) ≈ 14.9. A larger model should achieve lower perplexity given enough data, maybe approaching single-digit perplexity if trained extensively.) We’ll log this for insight.

Implementation Plan (Single-File PyTorch)

We will implement the entire model and training loop in a single Python file for simplicity. At the top of the file, we will have a configuration section where all settings are defined (architecture sizes, file paths, training hyperparams, etc.) as constants or a dataclass. This satisfies the requirement for “hardcoded” configs instead of CLI args.

Model definition: We will base this on the reference implementations. Key components to implement (or import) in code:

generalized_orders_of_magnitude.py – We might vendor in the GOOM library code directly (since we want one file, we can copy the necessary functions like goom.log, goom.exp, goom.log_matmul_exp, and the Config for GOOM). Alternatively, we can include pip install ... in instructions, but better to embed crucial parts to avoid dependency issues. We will configure GOOM as needed (e.g., goom.config.float_dtype=torch.float32 to use complex64, and goom.config.keep_logs_finite=True to handle log(0) gracefully).

The Log-space matrix operation for the recurrence: The reference uses a helper LogMultiDenseExp to combine two state updates in log space. We will include this module. It essentially creates a block matrix (by stacking A and identity matrices) to perform the scan step that multiplies and adds states in one go. We can largely reuse that code.

The SSMoverGOOMs module which implements one parallel recurrence as described. We’ll implement forward(u, continue_prev=False) such that:

If continue_prev is False, we start with the learned init_states as $x_0$; if True, we start from the last state of the previous invocation (this allows us to carry state between sequence chunks, useful for generation across long sequences or when doing truncated training, though for training we usually reset each sequence).

Prepare the per-step parameters: Expand A to shape [seq_len, d_hid, d_hid] (or broadcast it), compute $B u_t$ for each t and reshape to [seq_len, n_hid, d_hid].

Concatenate A and B*u to form a log-space combined matrix of shape [seq_len, d_hid + n_hid, d_hid] (this represents $A$ stacked over $B u$) and take goom.log of it.

Apply tps.prefix_scan with our LogMultiDenseExp as the combine function to this sequence. This produces a sequence of prefix-composed transformations log_cum_A_atop_Bu of the same shape.

Multiply the initial state (log-space) with this prefix product: we form log_x0_with_I = concat(log(x0), log(I)) as in the reference, then do goom.log_matmul_exp(log_x0_with_I, log_cum_A_atop_Bu) to get log_x (the log of each hidden state $x_t$ for t=1...T).

Save the last state log_x[-1] in self.log_prev_states (detached) if we might continue in future calls.

Scale and exponentiate: they recommend subtracting a constant from log_x to keep the values in a representable range before exp. In code, they find c = max(real(log_x)) per sequence and subtract it, then add a small constant (2). This ensures the exponentiated values don’t overflow float32 (because we ensured $log_x - c \le 2$, i.e., each exponent at most e^2). Then do goom.exp() to return to real domain. The result is the hidden states in real-space, of shape [seq_len, n_hid*d_hid].

Compute output $y_t = Cx_t + D u_t$ as two matrix multiplies and a sum. The output shape for this module is [seq_len, d_out].

The ResidualRecurrentLayer which contains an SSMoverGOOMs and the feed-forward: We’ll implement as in reference. It takes input of shape [seq_len, d_emb]:

Apply LayerNorm(d_emb).

ssm_over_gooms.forward(normed_input, continue_prev) produces shape [seq_len, d_emb*2] (since we set d_out = 2*d_emb for SSM to facilitate GLU).

Then a nn.GLU(dim=-1) applied to that yields [seq_len, d_emb]. We follow it with a linear layer (no bias) from d_emb to d_emb. This GLU+Linear is analogous to a feed-forward network of width 2× and contraction back to d_emb.

Add the skip connection: output = input + linear_out.

The top-level GenerativeRNN model: This wraps an embedding layer, a stack of residual recurrent layers, and an output projection:

nn.Embedding(vocab_size, d_emb) for input tokens.

A list of n_res ResidualRecurrentLayers (we can use nn.Sequential to stack them).

A final LayerNorm(d_emb) then nn.Linear(d_emb, vocab_size) as the language modeling head. We will tie the embedding and output weights (set linear.weight = embed.weight) to reduce parameters and slightly improve perplexity.

The model’s forward(token_ids, continue_prev=False) will embed the token IDs, run them through all the recurrent layers, apply final norm and linear to produce logits of shape [seq_len, vocab_size]. We’ll likely implement a helper body() that returns the final hidden states so that we can reuse it for generation.

We will include the convenience methods as in the reference: get_param_groups() to return parameter lists with/without decay (for the optimizer), compute_loss_and_metrics(preds, targets) to compute cross-entropy and optionally accuracy, and a generate() method for inference.

Training loop: We will write a train() function that does the following:

Set up the dataset and dataloader. Likely, we will stream data from disk or pre-tokenized files. Because the dataset is huge, an epoch could be defined as some fixed number of tokens or a full pass through a subset. We may not want to shuffle at the document level (to preserve some context); instead we can prepare sequences by concatenating text and chopping into blocks of seq_len. We will ensure an EOS token or reset at document boundaries as needed to avoid unrealistic cross-document blending.

Initialize the model and optimizer. We will set model.train() mode, move it to CUDA. We’ll initialize the AdamW optimizer with the two param groups from get_param_groups(weight_decay) (using weight_decay=0.1). If using 8-bit Adam, we will instantiate that accordingly (the API is similar).

If resuming from a checkpoint, load the state dict for the model and optimizer, and skip ahead in the dataset/iterations to the saved point.

For each epoch:

Loop over batches (each batch here meaning one micro-batch of, say, N sequences). Use a tqdm progress bar for the epoch to display live stats.

For each micro-batch:

Get the input token IDs tensor (shape [batch_size, seq_len]) and the target tensor (usually the same sequence shifted by 1 token for next-token prediction).

Move them to GPU, forward through the model to get logits. (We may wrap the forward pass in torch.cuda.amp.autocast() for mixed precision.)

Compute the loss with F.cross_entropy(logits.view(-1,vocab), targets.view(-1)) or use the model’s compute_loss_and_metrics helper which also can compute accuracy. Typically, we care about loss (and perplexity = exp(loss)). Accuracy for next-token prediction is usually very low (and not as informative as perplexity), but we can log it for completeness.

Divide the loss by the number of gradient accumulation steps (if using manual loss scaling for accum) or accumulate gradients without stepping optimizer.

Call loss.backward() to accumulate grads. We may need to call GradScaler.scale(loss) and later GradScaler.step(optimizer) if using AMP.

Every N micro-batches (where N * micro_batch_size = effective_batch), we perform an optimizer step:

If using GradScaler, unscale grads and optionally clip grad norm (if we decide to clip, e.g., at 1.0 – not strictly necessary unless we see instability).

optimizer.step() to update weights, then optimizer.zero_grad().

Update any LR scheduler if in use (OneCycle can be done stepwise or epoch-wise; we likely have an iteration-wise schedule for it).

The progress bar will be updated with current batch loss, a running average loss for the epoch, and other info (e.g. the current learning rate, token/sec throughput, etc.). This provides real-time feedback.

At epoch end:

Calculate the average training loss for the epoch (and maybe perplexity).

Evaluate on the validation set: switch model.eval(), and loop through a few validation batches (without grad) to compute val loss/perplexity. This is logged for comparison.

Sample generation: Using the model in eval mode, we will generate a few example continuations to monitor quality. For instance, we can select a prompt like "The quick brown fox" or a random prompt from validation data, then call model.generate(prompt_ids, n_new=100, ...) to produce 100 new tokens. We’ll use a moderate temperature (e.g. 1.0) and maybe top-k filtering (the generate method supports a topk parameter for truncation of the distribution). The model’s generate function carries forward the state between tokens using continue_prev=True after the first token, which is an efficient way to generate autoregressively without re-running the whole sequence each time. We ensure to wrap generation in torch.no_grad() and with torch.inference_mode() to prevent any gradient tracking and reduce memory usage. After generation, we can explicitly clear any large tensors or call torch.cuda.empty_cache() as a precaution so that we free VRAM before the next training epoch.

Print or save the sample text output. This “preview” helps qualitatively see the model’s progress (e.g., does it produce coherent sentences, does perplexity reduction translate to better text).

Save a checkpoint: we’ll use torch.save() to save a dictionary with model state_dict, optimizer state, current epoch and iteration, and any scheduler state. Checkpoints can be saved every epoch or even more frequently (e.g., every X updates) for safety. We will keep multiple checkpoints (e.g., the latest and best validation) in case we need to roll back.

Switch back to model.train() for the next epoch.

Continue to next epoch, unless the stopping criterion is met (e.g., reached target number of tokens or no improvement in val loss after some point).

Throughout training, logging will be comprehensive. The console will show per-batch or per-second metrics via the TQDM bar: e.g. Epoch 3: 40% [====......] loss=2.85 (avg 2.90) ppl=17.3 lr=2.0e-4 150k tok/s. We will also log epoch summaries to a file (or TensorBoard) including training loss, validation loss, and example outputs. This satisfies the requirement for extensive, informative logging in real-time.

In case of any interruptions or if we want to continue training further, the saved checkpoints allow resuming training seamlessly – the code will check if a checkpoint path is specified in the config and load it. The model and optimizer states are restored, and the LR scheduler will be recreated to the same state (we can save the scheduler last LR or step too, or simply recompute if using known schedule formulas). This way, resuming will pick up with the same weights and learning rate as if training never stopped.

Additional Considerations and Possible Enhancements

Fine-tuning and inference: Once the model is trained on the generic corpus, it can be further fine-tuned on conversation data to make it more "chat-like". For example, one could fine-tune on instruction-following dialogues or apply reinforcement learning from human feedback (RLHF). Those steps would use the same model architecture and codebase, just with different data and perhaps different sampling settings. Our implementation’s inference mode (via model.generate) will support generating responses given a prompt. We may implement a simple interactive loop when MODE='infer' in the script: it can load a trained checkpoint, then repeatedly prompt the user for input and generate a continuation.

Stability and testing: We will test the forward and backward pass on short sequences initially to ensure the integrated GOOM operations produce valid gradients. The GOOM library has taken care to make backprop stable even around singularities like log(0). If any part of the model doesn’t compile or autocast as expected (for instance, PyTorch 2.x might skip compiling some complex ops), we will adjust accordingly (e.g., ensure that the time-critical parts like matrix multiplications on floats are compiled).

Performance optimization: Because complex operations aren’t yet fused, one bottleneck might be the goom.log_matmul_exp in the prefix scan. As noted in the GOOM repo, an ideal kernel for this isn’t available in PyTorch as of 2025. We will use the implementation provided, but be aware it could be slower than equivalent float ops. Our use of torch.compile and ensuring the rest of the model (embedding, linear layers, etc.) are optimized should help. We will also set environment flags (like torch.set_float32_matmul_precision('high')) to prefer TF32 on tensor cores for matrix mult – this gives a speed boost on RTX GPUs with minimal loss of precision. Autocast does this by default for matmuls and convolutions. So effectively, the model will use tensor cores for big ops.

Verification of memory usage: We will track memory during training (e.g., torch.cuda.memory_allocated() after key steps) in debug mode to ensure we’re within limits. If we find we’re close to 32GB, we might reduce certain dimensions slightly or further tune gradient checkpointing strategy (e.g., checkpoint every layer vs every 2 layers, etc.). The target is to utilize most of the 32GB for the model (to maximize size) while leaving a bit of headroom to avoid OOM errors during spike usage.

In summary, the custom model will implement the cutting-edge GOOM-based parallel RNN from the research, enhanced with all feasible optimizations to push its size and performance on a single 32GB GPU. The architecture natively handles long sequences (we’ll train up to 2048 tokens context) and uses modern tricks like multi-head partitioned states, gated feed-forward, and pre-normalization – all aimed at high capability in text generation. With ~billion parameters trained on tens of billions of tokens, we expect this LLM to achieve strong results. The training process will be carefully managed with mixed precision and gradient accumulation to balance the GPU memory and utilization, and we’ll have rich logging and periodic sample generation to observe their “brainchild” learning to produce human-like text outputs.

By following this design and training strategy, we leverage the strengths of GOOM (numerical stability over long dependencies) to build an LLM-style model that is as large as possible for the hardware, and hopefully approaches the performance of transformer models of similar scale while being uniquely empowered to handle long-range patterns without attention mechanisms.

Sources:

Heinsen & Kozachkov (2025), “Generalized Orders of Magnitude for Scalable, Parallel, High-Dynamic-Range Computation” – introduced the GOOM concept and parallel RNN approach.

GOOM SSM RNN reference implementation – provided the base architecture and hyperparameter insights used here.

Training hyperparameters and results from GOOM RNN on language modeling.

Hoffmann et al. (2022), “Chinchilla Scaling Laws” – informed the choice of token count relative to model size.
