# Parked DiT LoRAs

Kept OUTSIDE `/home/kev/sa3/loras/` on purpose: the registry is directory-driven
(the LoRA's name is the filename stem), so anything left in `loras/` becomes a
selectable entry in the app.

## koan_shipped_2026-05-24.ckpt

The `koan` that shipped in the 2026-05-20 beta — trained on latents from the
STOCK SAME-L encoder. md5 c576071e2fd2df577e8cabe867c852a3.

Displaced 2026-08-14 by the enclora Stage 1 build (see below). To roll back:

    cp /home/kev/sa3/lora_archive/koan_shipped_2026-05-24.ckpt \
       /home/kev/sa3/loras/koan.ckpt
    docker exec gary-backend-spark-sa3-1 curl -sX POST localhost:8016/reload

## What is serving as `koan` right now

`~/sa3_lora_out_s1_enclora/epoch=83-step=3500.ckpt`, md5
8460939848609581eee81252eb3aa879 — same training flags and seed as the shipped
one, differing ONLY in that its training latents were pre-encoded through
SAME-L + enclora_v2 step-3500 instead of the stock encoder.

Its A/B against the stock-latent twin is in `~/vae_probe/enclora_s1_ab/`; the
difference was judged subtle on frozen-seed text-to-audio, which is why it is
being tried on continuation / transform instead.
