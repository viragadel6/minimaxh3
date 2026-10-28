# MiniMax-H3 video + audio generation server

A Modal deployment that renders short videos with synchronised audio from MiniMax-H3.
Three pieces ship together:

| File | Role |
|---|---|
| `app.py` | Modal app: GPU worker class (`VideoAudioGenerator`), weight downloader, HTTP entrypoint (`web`) and a local CLI (`main`). |
| `api.py` | FastAPI application created by `create_api(generator_cls, job_store, settings, outputs_volume)`. |
| `runtime/` | Orchestration (`generator`), job records (`jobs`), weights (`model_manager`, `weights`), LoRA (`lora`, `lora_keys`), step caching (`step_cache`), pipeline glue (`pipeline_adapter`). |
| `va_kernels/` | FP8 matmul/attention kernels and the attention router. Named `va_kernels` because a top-level `kernels` package shadows the PyPI one diffusers imports. |

## HTTP API

All errors share one shape:

```json
{"error": {"code": "invalid_request", "message": "num_frames: ..."}}
```

| Method | Path | Notes |
|---|---|---|
| `GET` | `/health` | Liveness probe. |
| `GET` | `/v1/models` | Supported modes, profiles, resolutions, frame grid (17n+5) and native segment bounds. |
| `POST` | `/v1/generations` | `202` with a `JobStatus`; the new id is echoed in `X-Job-Id`. |
| `GET` | `/v1/generations/{id}` | Polls the worker exactly once per call and returns `queued`/`running`/`succeeded`/`failed`. |
| `GET` | `/v1/generations/{id}/video` | Streams the MP4. `409` while unfinished, `410` if the job failed, `404` if the file is gone. |
| `POST` | `/v1/generations/sync` | Blocks and returns the MP4 directly (`413` above `MAX_SYNC_BYTES`, `504` on timeout — the worker call is cancelled). |
| `DELETE` | `/v1/generations/{id}` | Cancels a running job, removes the file and commits the volume (`204`). |

Request body (`GenerationRequest`, strict — unknown fields are rejected):

```json
{
  "prompt": "a cat surfing",
  "mode": "t2v",                     // or i2v / fl2v / ref2v; inferred when omitted
  "resolution": "768p",              // 480p | 768p
  "aspect_ratio": "16:9",            // or "auto" (reads it from the first frame / reference)
  "duration_seconds": 5.0,           // xor num_frames
  "num_frames": 124,                 // must be 17n+5 and >= 124
  "profile": "fast",                 // quality | fast | turbo (turbo needs TURBO_TRANSFORMER_REPO)
  "num_inference_steps": 20,
  "seed": 42,
  "first_frame": {"source": "https://…/a.png"},
  "last_frame":  {"source": "https://…/b.png"},
  "references": [{"type": "audio", "source": "https://…/a.wav", "start_seconds": 0}],
  "loras": [{"source": "hf://owner/repo", "weight_name": "adapter.safetensors", "scale": 0.8}],
  "include_audio": true
}
```

`source` accepts `http(s):`, `data:` and local paths. Audio references are additive in
every mode and are still honoured when `include_audio` is `false` (which only stops the
generated audio from being returned); image/video references imply `ref2v`. Durations are rounded up to the
nearest native length (17n+5, 124–345 frames), and longer clips are rendered as
overlapping segments that share one boundary frame.

## Configuration

Every `Settings` field in `config.py` is overridable through an identically named
environment variable (`load_settings()` coerces and validates them). The ones that
change behaviour most:

| Variable | Default | Effect |
|---|---|---|
| `GPU_TYPE` | `B200` | `B200` / `B300` / `H200`. |
| `DEFAULT_PROFILE` | `fast` | Profile used when the request omits one. |
| `WEIGHT_PRECISION` | `fp8` | `fp8` converts eligible block linears; `bf16` keeps the checkpoint. |
| `FP8_EXCLUDE_PATTERNS` | `embed,norm,adaln,modulation,time,proj_out,final,head` | Layers kept in their original precision. |
| `FP8_KEEP_EDGE_BLOCKS` | `2` | Untouched blocks at both ends of every primary stack. |
| `ATTENTION_BACKEND` | `fp8` | `fp8` routes attention through `va_kernels`; anything else stays native. |
| `ATTN_FP8_MIN_SEQ` | `4096` | Sequences shorter than this always take the native path. |
| `FAST_CACHE_THRESHOLD` | `0.12` | Step-residual skip threshold for the `fast` profile (`0` disables). |
| `WARMUP_BUCKETS` | `768p:16:9:124,480p:16:9:124` | `RES:W:H:FRAMES` warm-up shapes; malformed entries are skipped with a warning. |
| `MAX_SYNC_BYTES` | `512 MiB` | Response cap for `/v1/generations/sync`. |
| `OUTPUT_RETENTION_HOURS` | `0` (off) | Renders older than this are pruned on commit. |
| `FUNCTION_TIMEOUT` | `3600` | Worker timeout; also bounds how long a job may stay non-terminal. |
| `SYNC_REQUEST_TIMEOUT` | `1800` | Deadline for the synchronous endpoint. |

## Development

```bash
python -m venv .testenv && .testenv/bin/pip install pytest pydantic fastapi httpx numpy pillow
.testenv/bin/python -m pytest       # full suite, no GPU required
```

The suite covers schemas, shapes, config, job records, the HTTP contract, media IO,
LoRA key parsing, the router/kernel decision logic and a full API → job store →
worker → generator round trip. `tests/stubs.py` provides shallow `torch`/`triton`
stand-ins so the GPU modules can be imported and their host-side logic exercised.

**Not covered by the suite:** kernel numerics, `torch.compile`, weight loading and real
inference — those need a GPU. A first deployment should warm up, run a short `t2v`
job, then one `i2v` and one `ref2v` job to exercise caching, LoRA and router paths.
