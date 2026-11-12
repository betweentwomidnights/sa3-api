# SA3 CFG dtype decode fix (gary4local handoff)

Audience: the agent maintaining the Stable Audio 3 localhost backend in
gary4local.

This documents a reproduced inference failure and the exact fixes deployed to
the Spark SA3 backend on 2026-06-06.

## Symptom

Generation succeeds through diffusion, then fails during autoencoder decode:

```text
RuntimeError: expected scalar type Half but found Float
```

The failing stack ends at:

```text
api.py: generation_worker
  pipe.model.pretransform.decode(latents)
stable_audio_3/models/pretransforms.py: decode
stable_audio_3/models/autoencoders.py: decode_audio -> decode -> decoder
torch/nn/modules/linear.py
  F.linear(input, self.weight, self.bias)
```

The failure was deterministic when classifier-free guidance was enabled with
`cfg_scale=1.1`. Equivalent `cfg_scale=1.0` requests decoded successfully.

## Root cause

Our serving loudness/diagnostic path calls:

```python
pipe.generate(..., return_latents=True)
```

It then manually calls the pretransform decoder. When CFG is active, the engine
can return `torch.float32` latents. The model was loaded with `model_half=True`,
so the pretransform/autoencoder decoder parameters are `torch.float16`.

The engine's normal decode path already handles this boundary in
`stable_audio_3/inference/sampling.py`:

```python
sampled = sampled.to(next(pretransform.parameters()).dtype)
sampled = pretransform.decode(sampled, chunked=chunked_decode)
```

The manual `return_latents=True` path bypassed that cast and passed FP32 input to
an FP16 decoder.

This is a dtype contract bug, not a CUDA driver fault.

## Required API fix

Immediately before every manual `pretransform.decode(latents)` call, normalize
the latent tensor to the decoder's device and dtype:

```python
# return_latents bypasses the engine's normal decode boundary, which casts
# sampler output to the autoencoder dtype. CFG can return float32 latents while
# the half-loaded decoder expects float16, so mirror that boundary here.
decoder_param = next(pipe.model.pretransform.parameters())
if (latents.dtype != decoder_param.dtype
        or latents.device != decoder_param.device):
    logger.info(
        "decode cast %s/%s -> %s/%s",
        latents.device,
        latents.dtype,
        decoder_param.device,
        decoder_param.dtype,
    )
    latents = latents.to(
        device=decoder_param.device,
        dtype=decoder_param.dtype,
    )

audio = pipe.model.pretransform.decode(latents)
```

The Spark implementation uses `print(...)` instead of `logger.info(...)`; use
gary4local's established logging API.

Do not hardcode `torch.float16` or `cuda:0`. Deriving both properties from a
pretransform parameter keeps this correct for FP32, FP16, BF16, CPU, CUDA, and
other device configurations.

Apply the cast after any latent rescale/shift operation and immediately before
decode. Any operation that replaces `latents` after the cast could promote or
move it again.

## Recommended loader peak-memory fix

During deployment, SA3 failed to reload even with enough memory for the steady
FP16 model. The engine loader moved the full FP32 model to CUDA before converting
it to FP16:

```python
model.to(device).eval().requires_grad_(False)
if model_half:
    model.to(torch.float16)
```

That creates a large transient FP32 CUDA allocation. On the GB10 unified-memory
system, CUDA reported about 39 GB free and the transfer still OOMed.

Cast on CPU before moving the model to the target device:

```python
model = create_diffusion_cond_from_config(model_config)
copy_state_dict(model, load_file(ckpt_path))
if model_half:
    # Avoid a transient full-precision device copy during load/reload.
    model.to(torch.float16)
model.to(device).eval().requires_grad_(False)
return model
```

Location in the current engine:

```text
stable_audio_3/loading_utils.py
load_diffusion_cond(...)
```

The final model is unchanged: FP16 on the requested device. This only lowers
peak transfer memory. It is particularly important for lifecycle unload/reload
support and machines sharing GPU memory with other local models.

## Exact reproduction and verification

The pre-fix request consistently reached diffusion completion and then failed at
decode:

```json
{
  "prompt": "84 bpm",
  "negative_prompt": "low quality",
  "bpm": 84,
  "bars": 8,
  "steps": 8,
  "cfg_scale": 1.1,
  "shift": "full",
  "sampler_type": "pingpong",
