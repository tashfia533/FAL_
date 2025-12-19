import streamlit as st
import requests
import base64
import os
import json
import time
import hashlib
from datetime import datetime, timezone
import streamlit.components.v1 as components
from pathlib import Path
import uuid
import tempfile


# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Multi-Model Media Platform", layout="wide")

st.title("Multi-Model Media Platform")
st.caption(
    "Switch between different models in one UI, configure settings, call the FAL API, "
    "and browse generation history."
)

FAL_API_KEY = st.secrets.get("FAL_KEY") or os.getenv("FAL_KEY")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if not FAL_API_KEY and not OPENAI_API_KEY:
    st.error("Missing API keys. Add FAL_KEY and/or OPENAI_API_KEY in Streamlit Secrets (or env vars).")
    st.stop()
if not FAL_API_KEY:
    st.warning("FAL_KEY is missing: FAL models will fail until you add it.")
if not OPENAI_API_KEY:
    st.info("OPENAI_API_KEY is missing: OpenAI GPT Image 1.5 models will fail until you add it.")

# Use Queue API so models like FLUX/Ideogram work reliably (submit → poll → fetch output)
FAL_QUEUE_BASE = "https://queue.fal.run"
OPENAI_BASE_URL = "https://api.openai.com/v1"


HISTORY_FILE = "history.json"  # JSON "database"

# -----------------------------
# SMALL UI CSS (tiles)
# -----------------------------
st.markdown(
    """
    <style>
      /* Make sidebar buttons look like tabs */
      div[data-testid="column"] button[kind="secondary"] {
        width: 100%;
      }

      /* Tile container */
      .tile {
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 12px;
        padding: 10px;
        margin-bottom: 12px;
        background: rgba(255,255,255,0.02);
      }

      .tile-title {
        font-size: 12px;
        opacity: 0.85;
        margin-top: 6px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      /* Make Streamlit videos less tall in columns */
      div[data-testid="stVideo"] video {
        max-height: 160px !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# SESSION STATE (page + zoom)
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "Generator"

if "zoom_url" not in st.session_state:
    st.session_state.zoom_url = None
if "zoom_kind" not in st.session_state:
    st.session_state.zoom_kind = None
if "zoom_meta" not in st.session_state:
    st.session_state.zoom_meta = None

# -----------------------------
# HELPERS
# -----------------------------
def _hash(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]


def file_to_data_uri(uploaded_file) -> str | None:
    """
    Convert Streamlit UploadedFile to base64 data URI.
    NOTE: Base64 inflates size ~33%, and the FAL gateway enforces a ~10MB request-body limit.
    For large images, prefer uploading and passing a hosted URL instead.
    """
    if uploaded_file is None:
        return None
    file_bytes = uploaded_file.getvalue()
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    mime = getattr(uploaded_file, "type", None) or "application/octet-stream"
    return f"data:{mime};base64,{b64}"


def file_to_fal_url_or_data_uri(
    uploaded_file,
    *,
    upload_large_files: bool = True,
    max_request_body_bytes: int = 10_485_760,  # 10MB
    safety_margin_bytes: int = 900_000,        # room for JSON + prompt
) -> str | None:
    """
    Returns either:
      - a base64 data URI (small files), or
      - a hosted URL uploaded to fal.media via fal-client (large files).

    This avoids 413 "Maximum request body size 10485760 exceeded" when inlining large files.
    """
    if uploaded_file is None:
        return None

    file_bytes = uploaded_file.getvalue()
    mime = getattr(uploaded_file, "type", None) or "application/octet-stream"

    # Rough estimate of the request body if we inline the file as a data URI.
    # base64 expands by ~4/3, plus the "data:mime;base64," header and JSON quoting overhead.
    estimated_inline_bytes = int(len(file_bytes) * 4 / 3) + len(mime) + 64

    if upload_large_files and estimated_inline_bytes > (max_request_body_bytes - safety_margin_bytes):
        try:
            import fal_client  # pip install fal-client
        except Exception as e:
            raise RuntimeError(
                "Input image is too large to inline as base64 (would exceed the ~10MB request limit). "
                "Install 'fal-client' so the app can upload the file to fal.media and pass a URL instead:\n"
                "  pip install fal-client\n"
                f"Original import error: {e}"
            )

        # Ensure fal-client can see the key (it reads from env var FAL_KEY)
        if not os.environ.get("FAL_KEY") and "FAL_KEY" in st.secrets:
            os.environ["FAL_KEY"] = st.secrets["FAL_KEY"]

        # Upload via fal.media CDN, return URL
        suffix = Path(getattr(uploaded_file, "name", "")).suffix or ".bin"
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            return fal_client.upload_file(tmp_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    # Small enough to inline
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


# -----------------------------
# HISTORY HELPERS
# -----------------------------
def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def save_history(history):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"Could not save history: {e}")


def add_history_item(model_label, model_id, kind, urls, meta=None):
    """Store history. UI will NOT show time/type, but we keep timestamp for sorting."""
    if not urls:
        return
    history = load_history()
    history.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_label": model_label,
            "model": model_id,
            "kind": kind,  # image | video
            "urls": urls,
            "meta": meta or {},
        }
    )
    # Optional: keep file from growing forever
    history = history[-400:]  # last 400 entries
    save_history(history)


# -----------------------------
# FAL QUEUE CALL (submit -> poll -> fetch output)
# -----------------------------
def call_fal_model(model_id: str, payload: dict, timeout_s: int = 900, poll_s: float = 2.0) -> dict:
    """
    Queue flow:
      POST  https://queue.fal.run/<model_id> -> QueueStatus (request_id + urls). Typically returns 202.
      GET   status_url until COMPLETED (202 while IN_QUEUE/IN_PROGRESS, 200 when COMPLETED)
      GET   response_url to fetch final output (may be 400 if not ready yet)
    """
    submit_url = f"{FAL_QUEUE_BASE}/{model_id}"

    headers = {
        "Authorization": f"Key {FAL_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # --- submit ---
    r = requests.post(submit_url, json=payload, headers=headers, timeout=60)
    if r.status_code not in (200, 202):
        raise RuntimeError(f"FAL submit error {r.status_code}: {r.text[:500]}")

    status = r.json() if r.content else {}
    request_id = status.get("request_id")
    status_url = status.get("status_url")
    response_url = status.get("response_url")

    # Some endpoints may return output directly (rare); fallback
    if not request_id or not status_url or not response_url:
        if isinstance(status, dict):
            # Queue-style response wrapper: {"response": {...}}
            if isinstance(status.get("response"), dict):
                return status["response"]
            # Direct-style response: {"images": ...} / {"video": ...}
            if ("images" in status) or ("video" in status):
                return status
        raise RuntimeError(f"Unexpected queue response: {str(status)[:500]}")

    # --- poll status ---
    start = time.time()
    last_queue_position = None

    while True:
        if time.time() - start > timeout_s:
            raise TimeoutError("Timed out waiting for the model to finish.")

        sr = requests.get(status_url, headers=headers, timeout=60)

        # Queue status uses 202 while work is pending, and 200 when completed.
        if sr.status_code not in (200, 202):
            raise RuntimeError(f"FAL status error {sr.status_code}: {sr.text[:500]}")

        sjson = sr.json() if sr.content else {}
        st_status = sjson.get("status")

        if st_status == "COMPLETED":
            break

        if st_status in ("FAILED", "ERROR", "CANCELLED"):
            # Some endpoints return extra details under "error"
            raise RuntimeError(f"FAL request {request_id} failed: {json.dumps(sjson)[:800]}")

        # Optional: surface queue position when available (handy for debugging)
        qp = sjson.get("queue_position")
        if qp is not None and qp != last_queue_position:
            last_queue_position = qp

        time.sleep(poll_s)

    # --- fetch result ---
    # Occasionally the status flips to COMPLETED slightly before the response is readable,
    # so allow a few retries if we get 400 "not completed".
    for _ in range(12):
        rr = requests.get(response_url, headers=headers, timeout=120)
        if rr.status_code == 200:
            out = rr.json() if rr.content else {}
            # Most queue responses wrap model output inside {"response": {...}}
            if isinstance(out, dict) and isinstance(out.get("response"), dict):
                return out["response"]
            return out

        if rr.status_code == 400:
            time.sleep(1.0)
            continue

        raise RuntimeError(f"FAL response error {rr.status_code}: {rr.text[:500]}")

    raise TimeoutError("Model completed but response was not available in time.")

# -----------------------------
# OPENAI IMAGE CALLS (gpt-image-1.5)
# -----------------------------
def _openai_headers() -> dict:
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY.")
    return {"Authorization": f"Bearer {OPENAI_API_KEY}"}


def _save_openai_image_bytes(image_bytes: bytes, ext: str = "png") -> str:
    out_dir = Path.cwd() / "openai_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ext = (ext or "png").lower()
    fname = f"openai_{ts}_{uuid.uuid4().hex[:10]}.{ext}"
    path = out_dir / fname
    path.write_bytes(image_bytes)
    return str(path)


def openai_images_generate(
    prompt: str,
    n: int = 1,
    size: str = "auto",
    output_format: str = "png",
    quality: str = "auto",
    background: str = "auto",
    moderation: str = "auto",
) -> list[str]:
    url = f"{OPENAI_BASE_URL}/images/generations"
    payload = {
        "model": "gpt-image-1.5",
        "prompt": prompt,
        "n": int(n),
        "size": size,
        "output_format": output_format,
        "quality": quality,
        "background": background,
        "moderation": moderation,
    }
    r = requests.post(url, headers={**_openai_headers(), "Content-Type": "application/json"}, json=payload, timeout=300)
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI image generation error {r.status_code}: {r.text[:800]}")
    data = r.json()
    out = []
    for item in (data.get("data") or []):
        b64 = item.get("b64_json")
        if not b64:
            continue
        img_bytes = base64.b64decode(b64)
        out.append(_save_openai_image_bytes(img_bytes, ext=output_format))
    return out


def openai_images_edit(
    image_bytes: bytes,
    prompt: str,
    n: int = 1,
    size: str = "auto",
    output_format: str = "png",
    quality: str = "auto",
    background: str = "auto",
    moderation: str = "auto",
    mask_bytes: bytes | None = None,
) -> list[str]:
    url = f"{OPENAI_BASE_URL}/images/edits"

    files = [("image", ("input.png", image_bytes, "image/png"))]
    if mask_bytes:
        files.append(("mask", ("mask.png", mask_bytes, "image/png")))

    data = {
        "model": "gpt-image-1.5",
        "prompt": prompt,
        "n": str(int(n)),
        "size": size,
        "output_format": output_format,
        "quality": quality,
        "background": background,
        "moderation": moderation,
    }

    r = requests.post(url, headers=_openai_headers(), files=files, data=data, timeout=300)
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI image edit error {r.status_code}: {r.text[:800]}")
    payload = r.json()
    out = []
    for item in (payload.get("data") or []):
        b64 = item.get("b64_json")
        if not b64:
            continue
        img_bytes = base64.b64decode(b64)
        out.append(_save_openai_image_bytes(img_bytes, ext=output_format))
    return out




# -----------------------------
# PROMPT HELPER (OpenAI LLM)
# -----------------------------
def openai_make_pipeline_prompts(
    user_prompt: str,
    reference_image_url: str,
    storyboard_image_url: str,
    model: str = "gpt-4o-mini",
) -> dict:
    """
    Uses OpenAI Responses API (multimodal) to generate:
      - enhanced_prompt (for T2I if needed)
      - step1_prompt / step2_prompt / step3_prompt (for I2I steps)
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing (add it to secrets.toml or env vars).")

    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    system_text = (
        "You are a senior prompt engineer for image generation.\n"
        "Goal: Transform the REFERENCE image toward the STORYBOARD image while honoring the USER PROMPT.\n"
        "Return concise, production-ready prompts.\n"
        "Avoid copyrighted characters/logos and avoid mentioning any real person.\n"
        "Focus on: subject, composition, camera, lighting, materials, mood, background, and key details.\n"
    )

    schema = {
        "type": "object",
        "properties": {
            "enhanced_prompt": {"type": "string"},
            "step1_prompt": {"type": "string"},
            "step2_prompt": {"type": "string"},
            "step3_prompt": {"type": "string"},
            "negative_prompt": {"type": "string"},
            "notes": {"type": "string"},
        },
        "required": ["enhanced_prompt", "step1_prompt", "step2_prompt", "step3_prompt"],
        "additionalProperties": False,
    }

    payload = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_text}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": f"USER PROMPT:\n{user_prompt.strip()}"},
                    {"type": "input_text", "text": "REFERENCE IMAGE (starting point):"},
                    {"type": "input_image", "image_url": reference_image_url},
                    {"type": "input_text", "text": "STORYBOARD IMAGE (target direction):"},
                    {"type": "input_image", "image_url": storyboard_image_url},
                    {"type": "input_text", "text": "Generate prompts that apply storyboard direction onto the reference image."},
                ],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "pipeline_prompts",
                "strict": True,
                "schema": schema,
            }
        },
    }

    r = requests.post(url, headers=headers, json=payload, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI prompt helper error {r.status_code}: {r.text[:800]}")

    data = r.json()
    # Pull the JSON text from the response
    try:
        text_out = data.get("output", [])[0].get("content", [])[0].get("text")
        return json.loads(text_out)
    except Exception:
        # Fallback: try to find any text content
        for item in data.get("output", []):
            for c in item.get("content", []):
                if isinstance(c, dict) and c.get("type") == "output_text":
                    try:
                        return json.loads(c.get("text", "{}"))
                    except Exception:
                        pass
        raise RuntimeError("OpenAI prompt helper returned an unexpected format.")


# -----------------------------
# PIPELINE HELPERS
# -----------------------------
def _extract_image_urls(result: dict) -> list[str]:
    urls = []
    imgs = result.get("images") if isinstance(result, dict) else None
    if isinstance(imgs, list):
        for im in imgs:
            if isinstance(im, dict) and im.get("url"):
                urls.append(im["url"])
    # Some endpoints might use "image" or "output" fields; keep minimal.
    return urls


def _build_fal_edit_payload(model_id: str, prompt: str, image_urls: list[str], *, mask_url: str | None = None,
                           num_images: int = 1, output_format: str = "png",
                           image_size: str = "auto", quality: str = "high", background: str = "auto",
                           input_fidelity: str = "high") -> dict:
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("Empty prompt")

    # Model-specific knobs (use safe defaults)
    if model_id == "fal-ai/gpt-image-1.5/edit":
        payload = {
            "prompt": prompt,
            "image_urls": image_urls,
            "mask_image_url": mask_url,
            "num_images": int(num_images),
            "image_size": image_size,
            "quality": quality,
            "background": background,
            "input_fidelity": input_fidelity,
            "output_format": output_format,
            "sync_mode": False,
        }
        return {k: v for k, v in payload.items() if v is not None}

    if model_id == "fal-ai/nano-banana-pro/edit":
        payload = {
            "prompt": prompt,
            "image_urls": image_urls,
            "num_images": int(num_images),
            "output_format": output_format,
            "aspect_ratio": None,      # auto
            "resolution": "2K",        # good default
        }
        return {k: v for k, v in payload.items() if v is not None}

    if "/seedream/" in model_id and model_id.endswith("/edit"):
        # Seedream expects image_size object in the single-mode UI, but also works with defaults.
        payload = {
            "prompt": prompt,
            "image_urls": image_urls,
            "num_images": int(num_images),
            "output_format": output_format,
        }
        return {k: v for k, v in payload.items() if v is not None}

    if model_id == "fal-ai/flux-2-pro/edit":
        payload = {
            "prompt": prompt,
            "image_urls": image_urls,
            "output_format": output_format,
            "sync_mode": False,
            "enable_safety_checker": True,
        }
        return {k: v for k, v in payload.items() if v is not None}

    # Generic edit payload (many fal edit endpoints accept these)
    payload = {
        "prompt": prompt,
        "image_urls": image_urls,
        "num_images": int(num_images),
        "output_format": output_format,
    }
    return {k: v for k, v in payload.items() if v is not None}


def render_pipeline_ui():
    st.subheader("3-Step Pipeline")

    # Initialize pipeline session state
    for k, v in {
        "pipe_prompts": None,
        "pipe_step1_urls": None,
        "pipe_selected_idx": 0,
        "pipe_step2_url": None,
        "pipe_step3_url": None,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Pipeline-capable model lists (edit-first)
    edit_models = {label: mid for label, mid in MODEL_OPTIONS.items()
                   if mid.startswith("fal-ai") and ("/edit" in mid or "reframe" in mid or "kontext" in mid)}
    # Prefer the most useful ones first
    preferred_order = [
        "GPT Image 1.5 (FAL) Edit (Image + Text → Image)",
        "FLUX.2 Pro Edit (Image → Image)",
        "Seedream 4.5 Edit (Image + Text → Image)",
        "Seedream 4.0 Edit (Image + Text → Image)",
        "Nano Banana Pro Edit (Image + Text → Image)",
    ]
    ordered_labels = [x for x in preferred_order if x in edit_models] + [x for x in edit_models.keys() if x not in preferred_order]

    left, right = st.columns([1, 1])

    with right:
        st.markdown("### Results")
        if st.session_state.pipe_step1_urls:
            st.markdown("**Step 1 outputs** (pick one to continue):")
            cols = st.columns(4)
            for i, url in enumerate(st.session_state.pipe_step1_urls):
                with cols[i % 4]:
                    st.image(url, use_container_width=True)
                    if st.button(f"Use #{i+1}", key=f"pick_{i}"):
                        st.session_state.pipe_selected_idx = i
                        st.session_state.pipe_step2_url = None
                        st.session_state.pipe_step3_url = None
                        st.rerun()

        if st.session_state.pipe_step2_url:
            st.markdown("**Step 2 output**")
            st.image(st.session_state.pipe_step2_url, use_container_width=True)

        if st.session_state.pipe_step3_url:
            st.markdown("**Final (Step 3) output**")
            st.image(st.session_state.pipe_step3_url, use_container_width=True)

    with left:
        st.markdown("### Inputs")
        mode_help = "Pipeline mode: LLM writes prompts, then 3 models apply edits in sequence."
        st.caption(mode_help)

        user_prompt = st.text_area("What do you want to create?", key="pipe_user_prompt")

        ref_file = st.file_uploader("Reference image (starting point)", type=["png", "jpg", "jpeg", "webp"], key="pipe_ref")
        story_file = st.file_uploader("Storyboard image (target direction)", type=["png", "jpg", "jpeg", "webp"], key="pipe_story")
        mask_file = st.file_uploader("Optional mask (PNG)", type=["png"], key="pipe_mask")

        st.markdown("### Step models")
        step1_label = st.selectbox("Step 1 model (creates the first version)", ordered_labels, index=0, key="pipe_step1_model")
        step2_label = st.selectbox("Step 2 model (refine)", ordered_labels, index=min(1, len(ordered_labels)-1), key="pipe_step2_model")
        step3_label = st.selectbox("Step 3 model (final polish)", ordered_labels, index=min(2, len(ordered_labels)-1), key="pipe_step3_model")

        step1_id = edit_models[step1_label]
        step2_id = edit_models[step2_label]
        step3_id = edit_models[step3_label]

        st.markdown("### Output settings")
        output_format = st.selectbox("Format", ["png", "jpeg", "webp"], index=0, key="pipe_fmt")
        quality = st.selectbox("Quality", ["high", "medium", "low"], index=0, key="pipe_quality")
        background = st.selectbox("Background", ["auto", "opaque", "transparent"], index=0, key="pipe_bg")
        image_size = st.selectbox("Image size", ["auto", "1024x1024", "1536x1024", "1024x1536"], index=0, key="pipe_size")

        step1_n = st.slider("Step 1 variations", 1, 4, 2, key="pipe_n")  # 4 is safe for most fal image endpoints

        st.markdown("### Prompt helper")
        use_llm = st.checkbox("Use GPT‑4o mini to write prompts from (prompt + reference + storyboard)", value=True, key="pipe_use_llm")
        if use_llm and not OPENAI_API_KEY:
            st.warning("OPENAI_API_KEY missing. Add it to secrets.toml to enable the prompt helper.")
        gen_prompts_btn = st.button("✨ Generate prompts", use_container_width=True, disabled=(use_llm and not OPENAI_API_KEY))

        # Resolve reference/storyboard URLs for both OpenAI + FAL
        ref_url = None
        story_url = None
        mask_url = None
        if ref_file:
            ref_url = file_to_fal_url_or_data_uri(ref_file, upload_large_files=True)
        if story_file:
            story_url = file_to_fal_url_or_data_uri(story_file, upload_large_files=True)
        if mask_file:
            mask_url = file_to_fal_url_or_data_uri(mask_file, upload_large_files=True)

        if gen_prompts_btn:
            if not (user_prompt or "").strip():
                st.error("Please write a prompt first.")
                st.stop()
            if not ref_url or not story_url:
                st.error("Please upload both a reference image and a storyboard image.")
                st.stop()
            with st.spinner("Generating prompts with GPT‑4o mini…"):
                pack = openai_make_pipeline_prompts(user_prompt, ref_url, story_url, model="gpt-4o-mini")
                st.session_state.pipe_prompts = pack
                st.rerun()

        # Editable prompts (persist)
        if st.session_state.pipe_prompts is None:
            st.session_state.pipe_prompts = {
                "enhanced_prompt": "",
                "step1_prompt": "",
                "step2_prompt": "",
                "step3_prompt": "",
                "negative_prompt": "",
                "notes": "",
            }

        pp = st.session_state.pipe_prompts
        pp["enhanced_prompt"] = st.text_area("Enhanced prompt (optional)", value=pp.get("enhanced_prompt",""), key="pipe_enhanced")
        pp["step1_prompt"] = st.text_area("Step 1 prompt", value=pp.get("step1_prompt",""), key="pipe_p1")
        pp["step2_prompt"] = st.text_area("Step 2 prompt", value=pp.get("step2_prompt",""), key="pipe_p2")
        pp["step3_prompt"] = st.text_area("Step 3 prompt", value=pp.get("step3_prompt",""), key="pipe_p3")
        pp["negative_prompt"] = st.text_area("Negative prompt (optional)", value=pp.get("negative_prompt",""), key="pipe_neg")

        st.markdown("---")
        run_all_btn = st.button("🚀 Run pipeline", type="primary", use_container_width=True)

        if run_all_btn:
            if not FAL_API_KEY:
                st.error("FAL_KEY is missing.")
                st.stop()
            if not (user_prompt or "").strip():
                st.error("Please write a prompt.")
                st.stop()
            if not ref_url:
                st.error("Please upload a reference image.")
                st.stop()

            # If storyboard is missing, we still allow pipeline, but prompts should describe the target.
            if story_url is None:
                st.warning("No storyboard image uploaded. The pipeline will rely on your written prompt only.")

            # If LLM is enabled but prompts are blank, auto-generate once
            if use_llm and OPENAI_API_KEY and (not (pp.get("step1_prompt") or "").strip()):
                if not story_url:
                    st.error("To auto-generate prompts, please upload a storyboard image too.")
                    st.stop()
                with st.spinner("Generating prompts with GPT‑4o mini…"):
                    pack = openai_make_pipeline_prompts(user_prompt, ref_url, story_url, model="gpt-4o-mini")
                    st.session_state.pipe_prompts = pack
                    pp = pack

            # Step 1
            step1_images = [ref_url] + ([story_url] if story_url else [])
            payload1 = _build_fal_edit_payload(
                step1_id,
                pp.get("step1_prompt") or user_prompt,
                step1_images,
                mask_url=mask_url,
                num_images=int(step1_n),
                output_format=output_format,
                image_size=image_size,
                quality=quality,
                background=background,
            )
            with st.spinner("Running Step 1…"):
                res1 = call_fal_model(step1_id, payload1)
            urls1 = _extract_image_urls(res1)
            if not urls1:
                st.error("Step 1 returned no images.")
                st.stop()
            st.session_state.pipe_step1_urls = urls1
            st.session_state.pipe_selected_idx = 0

            # If multiple, let user choose before continuing
            if len(urls1) > 1:
                st.info("Step 1 produced multiple images. Pick one on the right, then click 'Run pipeline' again to continue.")
                st.rerun()

            chosen = urls1[0]

            # Step 2
            payload2 = _build_fal_edit_payload(
                step2_id,
                pp.get("step2_prompt") or user_prompt,
                [chosen],
                mask_url=None,
                num_images=1,
                output_format=output_format,
                image_size=image_size,
                quality=quality,
                background=background,
            )
            with st.spinner("Running Step 2…"):
                res2 = call_fal_model(step2_id, payload2)
            urls2 = _extract_image_urls(res2)
            if not urls2:
                st.error("Step 2 returned no images.")
                st.stop()
            st.session_state.pipe_step2_url = urls2[0]

            # Step 3
            payload3 = _build_fal_edit_payload(
                step3_id,
                pp.get("step3_prompt") or user_prompt,
                [urls2[0]],
                mask_url=None,
                num_images=1,
                output_format=output_format,
                image_size=image_size,
                quality=quality,
                background=background,
            )
            with st.spinner("Running Step 3…"):
                res3 = call_fal_model(step3_id, payload3)
            urls3 = _extract_image_urls(res3)
            if not urls3:
                st.error("Step 3 returned no images.")
                st.stop()
            st.session_state.pipe_step3_url = urls3[0]

            # Save to history as one "pipeline" entry
            add_history(
                model="pipeline",
                model_label=f"Pipeline: {step1_label} → {step2_label} → {step3_label}",
                prompt=user_prompt,
                result_urls=[st.session_state.pipe_step3_url],
                extra={
                    "step1": {"model": step1_id, "images": urls1},
                    "step2": {"model": step2_id, "image": st.session_state.pipe_step2_url},
                    "step3": {"model": step3_id, "image": st.session_state.pipe_step3_url},
                    "prompts": st.session_state.pipe_prompts,
                },
            )

            st.success("Pipeline complete.")
            st.rerun()

# -----------------------------
# MODEL OPTIONS
# -----------------------------
MODEL_OPTIONS = {
    "WAN Animate (Video + Image → Video)": "fal-ai/wan/v2.2-14b/animate/move",

    "Nano Banana Pro (Text → Image)": "fal-ai/nano-banana-pro",
    "Nano Banana Pro Edit (Image + Text → Image)": "fal-ai/nano-banana-pro/edit",

    "GPT Image 1.5 (FAL) (Text → Image)": "fal-ai/gpt-image-1.5",
    "GPT Image 1.5 (FAL) Edit (Image + Text → Image)": "fal-ai/gpt-image-1.5/edit",

    "OpenAI GPT Image 1.5 (Text → Image)": "openai/gpt-image-1.5",
    "OpenAI GPT Image 1.5 Edit (Image + Text → Image)": "openai/gpt-image-1.5/edit",


    "Seedream 4.0 (Text → Image)": "fal-ai/bytedance/seedream/v4/text-to-image",
    "Seedream 4.0 Edit (Image + Text → Image)": "fal-ai/bytedance/seedream/v4/edit",

    "Seedream 4.5 (Text → Image)": "fal-ai/bytedance/seedream/v4.5/text-to-image",
    "Seedream 4.5 Edit (Image + Text → Image)": "fal-ai/bytedance/seedream/v4.5/edit",

    "FLUX Kontext Max Multi (Multi-Image Edit)": "fal-ai/flux-pro/kontext/max/multi",
    "FLUX.2 Pro (Text → Image)": "fal-ai/flux-2-pro",
    "FLUX.2 Pro Edit (Image → Image)": "fal-ai/flux-2-pro/edit",

    "Ideogram v3 Reframe": "fal-ai/ideogram/v3/reframe",
}

# -----------------------------
# SIDEBAR "TABS" (side-by-side)
# -----------------------------
with st.sidebar:
    st.markdown("### Navigation")
    c1, c2 = st.columns(2)

    if c1.button("Generator", type="secondary", use_container_width=True):
        st.session_state.page = "Generator"
        st.session_state.zoom_url = None
        st.session_state.zoom_kind = None
        st.session_state.zoom_meta = None
        st.rerun()

    if c2.button("History", type="secondary", use_container_width=True):
        st.session_state.page = "History"
        st.session_state.zoom_url = None
        st.session_state.zoom_kind = None
        st.session_state.zoom_meta = None
        st.rerun()

# =========================================================
# HISTORY PAGE
# =========================================================
if st.session_state.page == "History":
    st.header("History")

    history = load_history()
    if not history:
        st.info("No history saved yet. Generate something first.")
        st.stop()

    # Newest first
    history_sorted = sorted(history, key=lambda x: x.get("timestamp", ""), reverse=True)

    # Group by model_label
    grouped = {}
    for entry in history_sorted:
        label = entry.get("model_label") or entry.get("model") or "Unknown"
        grouped.setdefault(label, []).append(entry)

    # Zoom viewer (top)
    if st.session_state.zoom_url:
        st.markdown("### Preview")
        if st.session_state.zoom_kind == "image":
            st.image(st.session_state.zoom_url, use_column_width=True)
        else:
            st.video(st.session_state.zoom_url)

        with st.expander("Details", expanded=False):
            st.json(st.session_state.zoom_meta or {})

        if st.button("Close preview"):
            st.session_state.zoom_url = None
            st.session_state.zoom_kind = None
            st.session_state.zoom_meta = None
            st.rerun()

        st.markdown("---")

    # Render per-model sections
    for model_label, entries in grouped.items():
        st.subheader(model_label)

        # Flatten urls into media tiles (so multi-image outputs become multiple tiles)
        tiles = []
        for e in entries:
            kind = e.get("kind", "image")
            meta = e.get("meta", {})
            for u in e.get("urls", []):
                tiles.append({"kind": kind, "url": u, "meta": meta, "model": e.get("model")})

        if not tiles:
            st.caption("No items.")
            continue

        # Separate image tiles and video tiles so layout looks clean
        image_tiles = [t for t in tiles if t["kind"] == "image"]
        video_tiles = [t for t in tiles if t["kind"] == "video"]

        # ---- Images grid ----
        if image_tiles:
            cols = st.columns(4)
            for i, t in enumerate(image_tiles):
                with cols[i % 4]:
                    st.markdown('<div class="tile">', unsafe_allow_html=True)
                    st.image(t["url"], width=180)
                    st.markdown(
                        f'<div class="tile-title">Image</div>',
                        unsafe_allow_html=True,
                    )
                    if st.button("Open", key=f"open_img_{model_label}_{i}_{_hash(t['url'])}"):
                        st.session_state.zoom_url = t["url"]
                        st.session_state.zoom_kind = "image"
                        st.session_state.zoom_meta = t["meta"]
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)

        # ---- Videos grid ----
        if video_tiles:
            cols = st.columns(3)
            for i, t in enumerate(video_tiles):
                with cols[i % 3]:
                    st.markdown('<div class="tile">', unsafe_allow_html=True)

                    # Smaller video using HTML so it behaves like a "tile"
                    components.html(
                        f"""
                        <video src="{t['url']}" controls preload="metadata"
                               style="width:100%; max-height:160px; border-radius:10px;"></video>
                        """,
                        height=190,
                    )

                    st.markdown(
                        f'<div class="tile-title">Video</div>',
                        unsafe_allow_html=True,
                    )
                    if st.button("Open", key=f"open_vid_{model_label}_{i}_{_hash(t['url'])}"):
                        st.session_state.zoom_url = t["url"]
                        st.session_state.zoom_kind = "video"
                        st.session_state.zoom_meta = t["meta"]
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")

    st.stop()

# =========================================================
# GENERATOR PAGE
# =========================================================
st.header("Generator")

# Mode switch: Single vs Pipeline
gen_mode = st.radio(
    "Mode",
    ["Single model", "3-step pipeline"],
    horizontal=True,
    index=0,
    key="gen_mode",
)

if gen_mode == "3-step pipeline":
    render_pipeline_ui()
    st.stop()


selected_model_label = st.selectbox("Choose model", list(MODEL_OPTIONS.keys()))
selected_model_id = MODEL_OPTIONS[selected_model_label]

st.write("---")

left, right = st.columns([1, 1])

with right:
    st.subheader("Result")
    output_area = st.empty()
    extra_info = st.empty()

with left:
    st.subheader("Inputs & Settings")

    run_btn = st.button("🚀 Run", type="primary", use_container_width=True)
    reset_btn = st.button("🔄 Reset", use_container_width=True)
    if reset_btn:
        st.rerun()

    st.markdown("---")

    # =====================================================
    # WAN ANIMATE
    # =====================================================
    if selected_model_id == "fal-ai/wan/v2.2-14b/animate/move":
        st.markdown("### WAN Animate – Video + Image → Video")

        video_file = st.file_uploader("Upload Source Video", type=["mp4", "mov", "webm", "m4v", "gif"])
        image_file = st.file_uploader("Upload Reference Image", type=["png", "jpg", "jpeg", "webp", "gif", "avif"])

        use_turbo = st.checkbox("Use Turbo", value=True)
        guidance_scale = st.slider("Guidance Scale", 0.0, 20.0, 1.0, 0.1)
        resolution = st.selectbox("Resolution", ["480p", "580p", "720p"], index=0)

        seed = st.number_input("Seed", min_value=0, max_value=2_147_483_647, value=1234)
        steps = st.number_input("Inference Steps", min_value=1, max_value=250, value=20)

        enable_safety = st.checkbox("Enable Safety Checker", value=True)
        enable_output_safety = st.checkbox("Enable Output Safety Checker", value=True)

        shift = st.slider("Shift", 1.0, 10.0, 5.0, 0.1)
        video_quality = st.selectbox("Video Quality", ["low", "medium", "high", "maximum"], index=2)
        video_mode = st.selectbox("Video Write Mode", ["fast", "balanced", "small"], index=1)
        return_zip = st.checkbox("Also return frames ZIP", value=False)

    # =====================================================
    # NANO BANANA PRO (T2I)
    # =====================================================
    elif selected_model_id == "fal-ai/nano-banana-pro":
        st.markdown("### Nano Banana Pro – Text → Image")

        prompt = st.text_area("Prompt")
        aspect_ratio = st.selectbox(
            "Aspect Ratio",
            ["21:9", "16:9", "3:2", "4:3", "5:4", "1:1", "4:5", "3:4", "2:3", "9:16"],
            index=5,
        )
        resolution = st.selectbox("Resolution", ["1K", "2K", "4K"], index=0)
        num_images = st.slider("Num Images", 1, 4, 1)
        output_format = st.selectbox("Output Format", ["png", "jpeg", "webp"], index=0)
        enable_web_search = st.checkbox("Enable Web Search", value=False)

    # =====================================================
    # NANO BANANA PRO EDIT (I2I)
    # =====================================================
    elif selected_model_id == "fal-ai/nano-banana-pro/edit":
        st.markdown("### Nano Banana Pro Edit – Image + Text → Image")

        edit_prompt = st.text_area("Edit Prompt")
        edit_images = st.file_uploader(
            "Upload Image(s) to Edit",
            type=["png", "jpg", "jpeg", "webp", "avif"],
            accept_multiple_files=True,
        )

        aspect_ratio = st.selectbox(
            "Aspect Ratio",
            ["auto", "21:9", "16:9", "3:2", "4:3", "5:4", "1:1", "4:5", "3:4", "2:3", "9:16"],
            index=0,
        )
        resolution = st.selectbox("Resolution", ["1K", "2K", "4K"], index=0)
        num_images = st.slider("Num Images", 1, 4, 1)
        output_format = st.selectbox("Output Format", ["png", "jpeg", "webp"], index=0)

    # =====================================================
    # SEEDREAM 4.0 T2I
    # =====================================================

    
    # =====================================================
    # GPT IMAGE 1.5 (FAL) (T2I)
    # =====================================================
    elif selected_model_id == "fal-ai/gpt-image-1.5":
        st.markdown("### GPT Image 1.5 (via FAL) – Text → Image")

        gi_prompt = st.text_area("Prompt", key="fal_gi15_prompt")
        gi_image_size = st.selectbox("Image Size", ["1024x1024", "1536x1024", "1024x1536"], index=0, key="fal_gi15_size")
        gi_quality = st.selectbox("Quality", ["high", "medium", "low"], index=0, key="fal_gi15_quality")
        gi_background = st.selectbox("Background", ["auto", "opaque", "transparent"], index=0, key="fal_gi15_bg")
        gi_output_format = st.selectbox("Output Format", ["png", "jpeg", "webp"], index=0, key="fal_gi15_fmt")
        gi_sync_mode = st.checkbox("Sync Mode (return data URI)", value=False, key="fal_gi15_sync")
        gi_num_images = st.slider("Num Images", 1, 4, 1, key="fal_gi15_n")

        st.caption("Note: fal currently caps this endpoint at 4 images per request.")

    # =====================================================
    # GPT IMAGE 1.5 (FAL) EDIT (I+T -> I)
    # =====================================================
    elif selected_model_id == "fal-ai/gpt-image-1.5/edit":
        st.markdown("### GPT Image 1.5 (via FAL) – Image + Text → Image")

        gi_edit_images = st.file_uploader(
            "Upload Image(s) to Edit (required)",
            type=["png", "jpg", "jpeg", "webp", "avif"],
            accept_multiple_files=True,
            key="fal_gi15_edit_images",
        )
        gi_mask = st.file_uploader("Optional Mask (png)", type=["png"], key="fal_gi15_mask")

        gi_edit_prompt = st.text_area("Edit Prompt", key="fal_gi15_edit_prompt")
        gi_edit_image_size = st.selectbox("Image Size", ["auto", "1024x1024", "1536x1024", "1024x1536"], index=0, key="fal_gi15_edit_size")
        gi_edit_quality = st.selectbox("Quality", ["high", "medium", "low"], index=0, key="fal_gi15_edit_quality")
        gi_edit_background = st.selectbox("Background", ["auto", "opaque", "transparent"], index=0, key="fal_gi15_edit_bg")
        gi_input_fidelity = st.selectbox("Input Fidelity", ["high", "low"], index=0, key="fal_gi15_fid")
        gi_edit_output_format = st.selectbox("Output Format", ["png", "jpeg", "webp"], index=0, key="fal_gi15_edit_fmt")
        gi_edit_sync_mode = st.checkbox("Sync Mode (return data URI)", value=False, key="fal_gi15_edit_sync")
        gi_edit_num_images = st.slider("Num Images", 1, 4, 1, key="fal_gi15_edit_n")

        st.caption("Note: fal currently caps this endpoint at 4 images per request.")


# =====================================================
    # OPENAI GPT IMAGE 1.5 (T2I)
    # =====================================================
    elif selected_model_id == "openai/gpt-image-1.5":
        st.markdown("### OpenAI GPT Image 1.5 – Text → Image")

        oai_prompt = st.text_area("Prompt", key="oai_t2i_prompt")
        oai_size = st.selectbox("Size", ["auto", "1024x1024", "1536x1024", "1024x1536"], index=0)
        oai_output_format = st.selectbox("Output Format", ["png", "jpeg", "webp"], index=0)
        oai_quality = st.selectbox("Quality", ["auto", "high", "medium", "low"], index=0)
        oai_background = st.selectbox("Background", ["auto", "opaque", "transparent"], index=0)
        oai_moderation = st.selectbox("Moderation", ["auto", "low"], index=0)
        oai_num_images = st.slider("Num Images", 1, 8, 1)

        if oai_background == "transparent" and oai_output_format not in ("png", "webp"):
            st.info("Transparent background works best with PNG or WEBP outputs.")

    # =====================================================
    # OPENAI GPT IMAGE 1.5 EDIT (I+T -> I)
    # =====================================================
    elif selected_model_id == "openai/gpt-image-1.5/edit":
        st.markdown("### OpenAI GPT Image 1.5 – Image + Text → Image")

        oai_edit_image = st.file_uploader("Input Image (png/jpg/webp)", type=["png", "jpg", "jpeg", "webp"], key="oai_edit_image")
        oai_edit_mask = st.file_uploader("Optional Mask (png with transparency)", type=["png"], key="oai_edit_mask")

        oai_edit_prompt = st.text_area("Prompt", key="oai_i2i_prompt")
        oai_edit_size = st.selectbox("Size", ["auto", "1024x1024", "1536x1024", "1024x1536"], index=0, key="oai_i2i_size")
        oai_edit_output_format = st.selectbox("Output Format", ["png", "jpeg", "webp"], index=0, key="oai_i2i_fmt")
        oai_edit_quality = st.selectbox("Quality", ["auto", "high", "medium", "low"], index=0, key="oai_i2i_quality")
        oai_edit_background = st.selectbox("Background", ["auto", "opaque", "transparent"], index=0, key="oai_i2i_bg")
        oai_edit_moderation = st.selectbox("Moderation", ["auto", "low"], index=0, key="oai_i2i_mod")
        oai_edit_num_images = st.slider("Num Images", 1, 8, 1, key="oai_i2i_n")

        if oai_edit_background == "transparent" and oai_edit_output_format not in ("png", "webp"):
            st.info("Transparent background works best with PNG or WEBP outputs.")
    elif selected_model_id == "fal-ai/bytedance/seedream/v4/text-to-image":
        st.markdown("### Seedream 4.0 – Text → Image")

        sd_prompt = st.text_area("Prompt")
        sd_width = st.number_input("Width (px)", min_value=512, max_value=4096, value=1280, step=64)
        sd_height = st.number_input("Height (px)", min_value=512, max_value=4096, value=1280, step=64)

        sd_num_images = st.slider("Num Images", 1, 4, 1)
        sd_max_images = st.number_input("Max Images", min_value=1, max_value=8, value=4)

        sd_seed = st.number_input("Seed (0 = random)", min_value=0, max_value=2_147_483_647, value=0)
        sd_sync_mode = st.checkbox("Sync Mode", value=False)
        sd_enable_safety = st.checkbox("Enable Safety Checker", value=True)
        sd_enhance_mode = st.selectbox("Enhance Prompt Mode", ["standard"], index=0)
        sd_output_format = st.selectbox("Output Format", ["png", "jpeg", "webp"], index=0)

    # =====================================================
    # SEEDREAM 4.0 EDIT
    # =====================================================
    elif selected_model_id == "fal-ai/bytedance/seedream/v4/edit":
        st.markdown("### Seedream 4.0 Edit – Image + Text → Image")

        sd_edit_prompt = st.text_area("Edit Prompt")
        sd_edit_images = st.file_uploader(
            "Upload Image(s) to Edit", type=["png", "jpg", "jpeg", "webp", "avif"], accept_multiple_files=True
        )

        sd_width = st.number_input("Width (px)", min_value=512, max_value=4096, value=1280, step=64)
        sd_height = st.number_input("Height (px)", min_value=512, max_value=4096, value=1280, step=64)

        sd_num_images = st.slider("Num Images", 1, 4, 1)
        sd_max_images = st.number_input("Max Images", min_value=1, max_value=8, value=4)

        sd_seed = st.number_input("Seed (0 = random)", min_value=0, max_value=2_147_483_647, value=0)
        sd_sync_mode = st.checkbox("Sync Mode", value=False)
        sd_enable_safety = st.checkbox("Enable Safety Checker", value=True)
        sd_enhance_mode = st.selectbox("Enhance Prompt Mode", ["standard"], index=0)
        sd_output_format = st.selectbox("Output Format", ["png", "jpeg", "webp"], index=0)

    # =====================================================
    # SEEDREAM 4.5 T2I
    # =====================================================
    elif selected_model_id == "fal-ai/bytedance/seedream/v4.5/text-to-image":
        st.markdown("### Seedream 4.5 – Text → Image")

        sd45_prompt = st.text_area("Prompt")
        sd45_image_size = st.selectbox(
            "Image Size",
            ["square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9", "auto_2K", "auto_4K"],
            index=6,
        )

        sd45_num_images = st.slider("Num Images", 1, 6, 1)
        sd45_max_images = st.number_input("Max Images", min_value=1, max_value=10, value=1)

        sd45_seed = st.number_input("Seed (0 = random)", min_value=0, max_value=2_147_483_647, value=0)
        sd45_sync_mode = st.checkbox("Sync Mode", value=False)
        sd45_enable_safety = st.checkbox("Enable Safety Checker", value=True)

    # =====================================================
    # SEEDREAM 4.5 EDIT
    # =====================================================
    elif selected_model_id == "fal-ai/bytedance/seedream/v4.5/edit":
        st.markdown("### Seedream 4.5 Edit – Image + Text → Image")

        sd45_edit_prompt = st.text_area("Edit Prompt")
        sd45_edit_images = st.file_uploader(
            "Upload Image(s) to Edit", type=["png", "jpg", "jpeg", "webp", "avif"], accept_multiple_files=True
        )

        sd45_image_size = st.selectbox(
            "Image Size",
            ["square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9", "auto_2K", "auto_4K"],
            index=7,
        )

        sd45_num_images = st.slider("Num Images", 1, 6, 1)
        sd45_max_images = st.number_input("Max Images", min_value=1, max_value=10, value=1)

        sd45_seed = st.number_input("Seed (0 = random)", min_value=0, max_value=2_147_483_647, value=0)
        sd45_sync_mode = st.checkbox("Sync Mode", value=False)
        sd45_enable_safety = st.checkbox("Enable Safety Checker", value=True)

    # =====================================================
    # FLUX PRO KONTEXT MAX MULTI
    # Schema (key fields): prompt (required), image_urls (required), guidance_scale, seed, sync_mode, num_images, output_format, safety_tolerance, enhance_prompt, aspect_ratio
    # =====================================================
    elif selected_model_id == "fal-ai/flux-pro/kontext/max/multi":
        st.markdown("### FLUX Kontext Max Multi – Multi-image edit")

        flux_prompt = st.text_area("Prompt", placeholder="Put the little duckling on top of the woman's t-shirt.")

        flux_images_up = st.file_uploader(
            "Upload Image(s) (required)",
            type=["png", "jpg", "jpeg", "webp", "avif"],
            accept_multiple_files=True,
            help="Upload 1+ images (the model requires image_urls).",
        )

        flux_image_urls_text = st.text_area(
            "Or paste image URL(s) (one per line)",
            help="Optional. If you upload images above, you can leave this empty.",
        )

        flux_guidance = st.slider("Guidance scale (CFG)", 1.0, 20.0, 3.5, 0.1)
        flux_seed = st.number_input("Seed (0 = random)", 0, 2_147_483_647, 0)
        flux_sync = st.checkbox("Sync Mode", value=False)
        flux_num = st.slider("Num Images", 1, 4, 1)
        flux_format = st.selectbox("Output Format", ["jpeg", "png"], index=0)
        flux_safety = st.selectbox("Safety Tolerance", ["1", "2", "3", "4", "5", "6"], index=1)
        flux_enhance = st.checkbox("Enhance Prompt", value=False)
        flux_aspect = st.selectbox(
            "Aspect Ratio (optional)",
            ["(none)", "21:9", "16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16", "9:21"],
            index=0,
        )

    # =====================================================
    # FLUX.2 PRO (TEXT TO IMAGE)
    # Schema: prompt(required), image_size (preset or {w,h}), seed, safety_tolerance, enable_safety_checker, output_format, sync_mode
    # =====================================================
    elif selected_model_id == "fal-ai/flux-2-pro":
        st.markdown("### FLUX.2 Pro – Text → Image")

        f2_prompt = st.text_area("Prompt")

        f2_size_mode = st.selectbox("Image Size Mode", ["Preset", "Custom"], index=0)
        if f2_size_mode == "Preset":
            f2_image_size = st.selectbox(
                "Preset",
                ["square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9"],
                index=4,  # landscape_4_3 default
            )
            f2_custom_w = None
            f2_custom_h = None
        else:
            f2_custom_w = st.number_input("Width", min_value=64, max_value=14142, value=1024, step=64)
            f2_custom_h = st.number_input("Height", min_value=64, max_value=14142, value=768, step=64)
            f2_image_size = None

        f2_seed = st.number_input("Seed (0 = random)", 0, 2_147_483_647, 0)
        f2_safety = st.selectbox("Safety Tolerance", ["1", "2", "3", "4", "5"], index=1)
        f2_enable_checker = st.checkbox("Enable Safety Checker", value=True)
        f2_format = st.selectbox("Output Format", ["jpeg", "png"], index=0)
        f2_sync = st.checkbox("Sync Mode", value=False)

    # =====================================================
    # FLUX.2 PRO EDIT
    # (based on playground): prompt(required), image_urls(required), image_size(auto or preset or custom), seed, safety_tolerance, enable_safety_checker, output_format, sync_mode
    # =====================================================
    elif selected_model_id == "fal-ai/flux-2-pro/edit":
        st.markdown("### FLUX.2 Pro Edit – Image + Text → Image")

        f2e_prompt = st.text_area("Prompt (edit instruction)")

        f2e_images = st.file_uploader(
            "Upload Image(s) (required)",
            type=["png", "jpg", "jpeg", "webp", "avif"],
            accept_multiple_files=True,
        )

        f2e_size_mode = st.selectbox("Image Size Mode", ["Auto", "Preset", "Custom"], index=0)
        if f2e_size_mode == "Auto":
            f2e_image_size = "auto"
            f2e_w = None
            f2e_h = None
        elif f2e_size_mode == "Preset":
            f2e_image_size = st.selectbox(
                "Preset",
                ["square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9"],
                index=0,
            )
            f2e_w = None
            f2e_h = None
        else:
            f2e_w = st.number_input("Width", min_value=64, max_value=14142, value=1024, step=64)
            f2e_h = st.number_input("Height", min_value=64, max_value=14142, value=1024, step=64)
            f2e_image_size = None

        f2e_seed = st.number_input("Seed (0 = random)", 0, 2_147_483_647, 0)
        f2e_safety = st.selectbox("Safety Tolerance", ["1", "2", "3", "4", "5"], index=1)
        f2e_enable_checker = st.checkbox("Enable Safety Checker", value=True)
        f2e_format = st.selectbox("Output Format", ["jpeg", "png"], index=0)
        f2e_sync = st.checkbox("Sync Mode", value=False)

    # =====================================================
    # IDEOGRAM V3 REFRAME
    # Schema (key fields): image_url(required), image_size(required), rendering_speed, color_palette, style, style_codes, style_preset,
    #                      image_urls(style reference), num_images, seed, sync_mode
    # =====================================================
    elif selected_model_id == "fal-ai/ideogram/v3/reframe":
        st.markdown("### Ideogram v3 Reframe")

        # Required: image_url
        ideo_source_mode = st.selectbox("Source Image Input", ["Upload", "URL"], index=0)
        if ideo_source_mode == "Upload":
            ideo_upload = st.file_uploader("Upload Image (required)", type=["jpg", "jpeg", "png", "webp", "gif", "avif"])
            ideo_image_url = None
            ideo_image_url_text = None
        else:
            ideo_upload = None
            ideo_image_url_text = st.text_input("Image URL (required)")
            ideo_image_url = None

        ideo_image_size_mode = st.selectbox("Image Size", ["square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9"], index=0)

        ideo_render_speed = st.selectbox("Rendering Speed", ["BALANCED", "QUALITY", "TURBO"], index=0)
        ideo_num_images = st.slider("Num Images", 1, 8, 1)
        ideo_seed = st.number_input("Seed (0 = random)", 0, 2_147_483_647, 0)
        ideo_sync = st.checkbox("Sync Mode", value=False)

        st.markdown("#### Style (optional)")
        ideo_style_codes = st.text_input("Style Codes (comma-separated)", help="If set, you cannot use Style or Style Reference Images.")
        ideo_style = st.selectbox("Style", ["(none)", "AUTO", "GENERAL", "REALISTIC", "DESIGN"], index=0)

        # Huge list exists; keep it optional and freeform-friendly:
        ideo_style_preset = st.text_input("Style Preset (optional)", placeholder="e.g. PHOTO / CINEMATIC / ... (paste preset name)")

        st.markdown("#### Style Reference Images (optional)")
        ideo_ref_images = st.file_uploader(
            "Upload reference image(s)",
            type=["jpg", "jpeg", "png", "webp", "gif", "avif"],
            accept_multiple_files=True,
            help="Used for style reference. Not allowed when Style Codes are provided.",
        )

        st.markdown("#### Color Palette (optional)")
        palette_mode = st.selectbox("Color Palette Mode", ["None", "Preset name", "Custom RGB"], index=0)
        if palette_mode == "None":
            ideo_palette = None
        elif palette_mode == "Preset name":
            palette_name = st.text_input("Palette Name", placeholder="e.g. 'ARCADIA' or another Ideogram palette name")
            palette_weight = st.slider("Color Weight", 0.0, 1.0, 0.5, 0.05)
            ideo_palette = {"name": palette_name.strip() if palette_name else "", "members": [], "color_weight": palette_weight}
        else:
            # Custom palette with up to 5 colors via hex
            palette_weight = st.slider("Color Weight", 0.0, 1.0, 0.5, 0.05)
            ncols = st.slider("How many colors?", 1, 5, 2)
            members = []
            for i in range(ncols):
                hx = st.text_input(f"Color {i+1} hex", value="#FF0000" if i == 0 else "#00FF00")
                hx = hx.strip().lstrip("#")
                if len(hx) == 6:
                    r = int(hx[0:2], 16)
                    g = int(hx[2:4], 16)
                    b = int(hx[4:6], 16)
                    members.append({"rgb": {"r": r, "g": g, "b": b}})
            ideo_palette = {"name": "", "members": members, "color_weight": palette_weight}

# -----------------------------
# RUN LOGIC
# -----------------------------
if run_btn:
    try:
        with st.spinner("Calling API…"):

            # -------- WAN ANIMATE --------
            if selected_model_id == "fal-ai/wan/v2.2-14b/animate/move":
                if not video_file or not image_file:
                    st.error("Please upload both a video and an image.")
                    st.stop()

                payload = {
                    "video_url": file_to_data_uri(video_file),
                    "image_url": file_to_data_uri(image_file),
                    "guidance_scale": float(guidance_scale),
                    "resolution": resolution,
                    "seed": int(seed),
                    "num_inference_steps": int(steps),
                    "enable_safety_checker": bool(enable_safety),
                    "enable_output_safety_checker": bool(enable_output_safety),
                    "shift": float(shift),
                    "video_quality": video_quality,
                    "video_write_mode": video_mode,
                    "return_frames_zip": bool(return_zip),
                    "use_turbo": bool(use_turbo),
                }

                result = call_fal_model(selected_model_id, payload)

                video_data = result.get("video") or {}
                video_url = video_data.get("url")

                if not video_url:
                    st.error("No video URL returned.")
                    st.stop()

                output_area.video(video_url)

                add_history_item(
                    model_label=selected_model_label,
                    model_id=selected_model_id,
                    kind="video",
                    urls=[video_url],
                    meta={"seed": result.get("seed")},
                )

            # -------- NANO BANANA PRO --------
            elif selected_model_id == "fal-ai/nano-banana-pro":
                if not (prompt or "").strip():
                    st.error("Please enter a prompt.")
                    st.stop()

                payload = {
                    "prompt": prompt.strip(),
                    "num_images": int(num_images),
                    "aspect_ratio": aspect_ratio,
                    "resolution": resolution,
                    "output_format": output_format,
                    "enable_web_search": bool(enable_web_search),
                }

                result = call_fal_model(selected_model_id, payload)
                images = result.get("images") or []
                if not images:
                    st.error("No images returned.")
                    st.stop()

                urls = []
                cols = st.columns(min(len(images), 2))
                for i, img in enumerate(images):
                    url = img.get("url")
                    if not url:
                        continue
                    urls.append(url)
                    cols[i % len(cols)].image(url, use_column_width=True)

                add_history_item(
                    model_label=selected_model_label,
                    model_id=selected_model_id,
                    kind="image",
                    urls=urls,
                    meta={"prompt": prompt.strip()},
                )

            # -------- NANO BANANA PRO EDIT --------
            elif selected_model_id == "fal-ai/nano-banana-pro/edit":
                if not (edit_prompt or "").strip():
                    st.error("Please enter an edit prompt.")
                    st.stop()
                if not edit_images:
                    st.error("Please upload at least one image to edit.")
                    st.stop()

                image_urls = [file_to_data_uri(f) for f in edit_images]
                payload = {
                    "prompt": edit_prompt.strip(),
                    "image_urls": image_urls,
                    "num_images": int(num_images),
                    "aspect_ratio": None if aspect_ratio == "auto" else aspect_ratio,
                    "resolution": resolution,
                    "output_format": output_format,
                }
                payload = {k: v for k, v in payload.items() if v is not None}

                result = call_fal_model(selected_model_id, payload)
                images = result.get("images") or []
                if not images:
                    st.error("No images returned.")
                    st.stop()

                urls = [img.get("url") for img in images if img.get("url")]
                cols = st.columns(min(len(urls), 2))
                for i, url in enumerate(urls):
                    cols[i % len(cols)].image(url, use_column_width=True)

                add_history_item(
                    model_label=selected_model_label,
                    model_id=selected_model_id,
                    kind="image",
                    urls=urls,
                    meta={"prompt": edit_prompt.strip()},
                )


            # -------- GPT IMAGE 1.5 (FAL) T2I --------
            elif selected_model_id == "fal-ai/gpt-image-1.5":
                if not FAL_API_KEY:
                    st.error("FAL_KEY is missing. Add it in Streamlit Secrets or env vars.")
                    st.stop()
                if not (gi_prompt or "").strip():
                    st.error("Please enter a prompt.")
                    st.stop()

                payload = {
                    "prompt": gi_prompt.strip(),
                    "num_images": int(gi_num_images),
                    "image_size": gi_image_size,
                    "background": gi_background,
                    "quality": gi_quality,
                    "output_format": gi_output_format,
                    "sync_mode": bool(gi_sync_mode),
                }

                result = call_fal_model(selected_model_id, payload)
                images = result.get("images") or []
                urls = [img.get("url") for img in images if img.get("url")]
                if not urls:
                    st.error("No images returned.")
                    st.stop()

                cols = st.columns(min(len(urls), 2))
                for i, url in enumerate(urls):
                    cols[i % len(cols)].image(url, use_column_width=True)

                add_history_item(
                    model_label=selected_model_label,
                    model_id=selected_model_id,
                    kind="image",
                    urls=urls,
                    meta={"prompt": gi_prompt.strip(), "image_size": gi_image_size, "quality": gi_quality},
                )

            # -------- GPT IMAGE 1.5 (FAL) EDIT --------
            elif selected_model_id == "fal-ai/gpt-image-1.5/edit":
                if not FAL_API_KEY:
                    st.error("FAL_KEY is missing. Add it in Streamlit Secrets or env vars.")
                    st.stop()
                if not (gi_edit_prompt or "").strip():
                    st.error("Please enter an edit prompt.")
                    st.stop()
                if not gi_edit_images:
                    st.error("Please upload at least 1 image to edit.")
                    st.stop()

                payload = {
                    "prompt": gi_edit_prompt.strip(),
                    "image_urls": [file_to_fal_url_or_data_uri(f) for f in gi_edit_images],
                    "mask_image_url": file_to_fal_url_or_data_uri(gi_mask) if gi_mask else None,
                    "num_images": int(gi_edit_num_images),
                    "image_size": gi_edit_image_size,
                    "background": gi_edit_background,
                    "quality": gi_edit_quality,
                    "input_fidelity": gi_input_fidelity,
                    "output_format": gi_edit_output_format,
                    "sync_mode": bool(gi_edit_sync_mode),
                }
                payload = {k: v for k, v in payload.items() if v is not None}

                result = call_fal_model(selected_model_id, payload)
                images = result.get("images") or []
                urls = [img.get("url") for img in images if img.get("url")]
                if not urls:
                    st.error("No images returned.")
                    st.stop()

                cols = st.columns(min(len(urls), 2))
                for i, url in enumerate(urls):
                    cols[i % len(cols)].image(url, use_column_width=True)

                add_history_item(
                    model_label=selected_model_label,
                    model_id=selected_model_id,
                    kind="image",
                    urls=urls,
                    meta={"prompt": gi_edit_prompt.strip(), "image_size": gi_edit_image_size, "quality": gi_edit_quality},
                )

            # -------- OPENAI DIRECT GPT IMAGE 1.5 (T2I) --------
            elif selected_model_id == "openai/gpt-image-1.5":
                if not OPENAI_API_KEY:
                    st.error("OPENAI_API_KEY is missing. Add it in Streamlit Secrets or env vars.")
                    st.stop()
                if not (oai_prompt or "").strip():
                    st.error("Please enter a prompt.")
                    st.stop()

                paths = openai_images_generate(
                    prompt=oai_prompt.strip(),
                    n=int(oai_num_images),
                    size=oai_size,
                    output_format=oai_output_format,
                    quality=oai_quality,
                    background=oai_background,
                    moderation=oai_moderation,
                )
                if not paths:
                    st.error("No images returned.")
                    st.stop()

                cols = st.columns(min(len(paths), 2))
                for i, p in enumerate(paths):
                    cols[i % len(cols)].image(p, use_column_width=True)

                add_history_item(
                    model_label=selected_model_label,
                    model_id=selected_model_id,
                    kind="image",
                    urls=paths,
                    meta={"prompt": oai_prompt.strip(), "size": oai_size, "quality": oai_quality},
                )

            # -------- OPENAI DIRECT GPT IMAGE 1.5 (EDIT) --------
            elif selected_model_id == "openai/gpt-image-1.5/edit":
                if not OPENAI_API_KEY:
                    st.error("OPENAI_API_KEY is missing. Add it in Streamlit Secrets or env vars.")
                    st.stop()
                if not oai_edit_image:
                    st.error("Please upload an input image.")
                    st.stop()
                if not (oai_edit_prompt or "").strip():
                    st.error("Please enter a prompt.")
                    st.stop()

                image_bytes = oai_edit_image.read()
                mask_bytes = oai_edit_mask.read() if oai_edit_mask else None

                paths = openai_images_edit(
                    image_bytes=image_bytes,
                    prompt=oai_edit_prompt.strip(),
                    n=int(oai_edit_num_images),
                    size=oai_edit_size,
                    output_format=oai_edit_output_format,
                    quality=oai_edit_quality,
                    background=oai_edit_background,
                    moderation=oai_edit_moderation,
                    mask_bytes=mask_bytes,
                )
                if not paths:
                    st.error("No images returned.")
                    st.stop()

                cols = st.columns(min(len(paths), 2))
                for i, p in enumerate(paths):
                    cols[i % len(cols)].image(p, use_column_width=True)

                add_history_item(
                    model_label=selected_model_label,
                    model_id=selected_model_id,
                    kind="image",
                    urls=paths,
                    meta={"prompt": oai_edit_prompt.strip(), "size": oai_edit_size, "quality": oai_edit_quality},
                )

            # -------- SEEDREAM 4.0 T2I --------
            elif selected_model_id == "fal-ai/bytedance/seedream/v4/text-to-image":
                if not (sd_prompt or "").strip():
                    st.error("Please enter a prompt.")
                    st.stop()

                payload = {
                    "prompt": sd_prompt.strip(),
                    "image_size": {"width": int(sd_width), "height": int(sd_height)},
                    "num_images": int(sd_num_images),
                    "max_images": int(sd_max_images),
                    "seed": None if int(sd_seed) == 0 else int(sd_seed),
                    "sync_mode": bool(sd_sync_mode),
                    "enable_safety_checker": bool(sd_enable_safety),
                    "enhance_prompt_mode": sd_enhance_mode,
                    "output_format": sd_output_format,
                }
                payload = {k: v for k, v in payload.items() if v is not None}

                result = call_fal_model(selected_model_id, payload)
                images = result.get("images") or []
                urls = [img.get("url") for img in images if img.get("url")]
                if not urls:
                    st.error("No images returned.")
                    st.stop()

                cols = st.columns(min(len(urls), 2))
                for i, url in enumerate(urls):
                    cols[i % len(cols)].image(url, use_column_width=True)

                add_history_item(selected_model_label, selected_model_id, "image", urls, {"prompt": sd_prompt.strip()})

            # -------- SEEDREAM 4.0 EDIT --------
            elif selected_model_id == "fal-ai/bytedance/seedream/v4/edit":
                if not (sd_edit_prompt or "").strip():
                    st.error("Please enter an edit prompt.")
                    st.stop()
                if not sd_edit_images:
                    st.error("Please upload at least one image.")
                    st.stop()

                payload = {
                    "prompt": sd_edit_prompt.strip(),
                    "image_urls": [file_to_data_uri(f) for f in sd_edit_images],
                    "image_size": {"width": int(sd_width), "height": int(sd_height)},
                    "num_images": int(sd_num_images),
                    "max_images": int(sd_max_images),
                    "seed": None if int(sd_seed) == 0 else int(sd_seed),
                    "sync_mode": bool(sd_sync_mode),
                    "enable_safety_checker": bool(sd_enable_safety),
                    "enhance_prompt_mode": sd_enhance_mode,
                    "output_format": sd_output_format,
                }
                payload = {k: v for k, v in payload.items() if v is not None}

                result = call_fal_model(selected_model_id, payload)
                images = result.get("images") or []
                urls = [img.get("url") for img in images if img.get("url")]
                if not urls:
                    st.error("No images returned.")
                    st.stop()

                cols = st.columns(min(len(urls), 2))
                for i, url in enumerate(urls):
                    cols[i % len(cols)].image(url, use_column_width=True)

                add_history_item(selected_model_label, selected_model_id, "image", urls, {"prompt": sd_edit_prompt.strip()})

            # -------- SEEDREAM 4.5 T2I --------
            elif selected_model_id == "fal-ai/bytedance/seedream/v4.5/text-to-image":
                if not (sd45_prompt or "").strip():
                    st.error("Please enter a prompt.")
                    st.stop()

                payload = {
                    "prompt": sd45_prompt.strip(),
                    "image_size": sd45_image_size,
                    "num_images": int(sd45_num_images),
                    "max_images": int(sd45_max_images),
                    "seed": None if int(sd45_seed) == 0 else int(sd45_seed),
                    "sync_mode": bool(sd45_sync_mode),
                    "enable_safety_checker": bool(sd45_enable_safety),
                }
                payload = {k: v for k, v in payload.items() if v is not None}

                result = call_fal_model(selected_model_id, payload)
                images = result.get("images") or []
                urls = [img.get("url") for img in images if img.get("url")]
                if not urls:
                    st.error("No images returned.")
                    st.stop()

                cols = st.columns(min(len(urls), 2))
                for i, url in enumerate(urls):
                    cols[i % len(cols)].image(url, use_column_width=True)

                add_history_item(selected_model_label, selected_model_id, "image", urls, {"prompt": sd45_prompt.strip()})

            # -------- SEEDREAM 4.5 EDIT --------
            elif selected_model_id == "fal-ai/bytedance/seedream/v4.5/edit":
                if not (sd45_edit_prompt or "").strip():
                    st.error("Please enter an edit prompt.")
                    st.stop()
                if not sd45_edit_images:
                    st.error("Please upload at least one image.")
                    st.stop()

                payload = {
                    "prompt": sd45_edit_prompt.strip(),
                    "image_urls": [file_to_data_uri(f) for f in sd45_edit_images],
                    "image_size": sd45_image_size,
                    "num_images": int(sd45_num_images),
                    "max_images": int(sd45_max_images),
                    "seed": None if int(sd45_seed) == 0 else int(sd45_seed),
                    "sync_mode": bool(sd45_sync_mode),
                    "enable_safety_checker": bool(sd45_enable_safety),
                }
                payload = {k: v for k, v in payload.items() if v is not None}

                result = call_fal_model(selected_model_id, payload)
                images = result.get("images") or []
                urls = [img.get("url") for img in images if img.get("url")]
                if not urls:
                    st.error("No images returned.")
                    st.stop()

                cols = st.columns(min(len(urls), 2))
                for i, url in enumerate(urls):
                    cols[i % len(cols)].image(url, use_column_width=True)

                add_history_item(selected_model_label, selected_model_id, "image", urls, {"prompt": sd45_edit_prompt.strip()})

            # -------- FLUX PRO KONTEXT MAX MULTI --------
            elif selected_model_id == "fal-ai/flux-pro/kontext/max/multi":
                if not (flux_prompt or "").strip():
                    st.error("Please enter a prompt.")
                    st.stop()

                # Combine uploaded images + pasted URLs
                all_imgs = []
                if flux_images_up:
                    all_imgs.extend([file_to_data_uri(f) for f in flux_images_up])

                if (flux_image_urls_text or "").strip():
                    for line in flux_image_urls_text.splitlines():
                        line = line.strip()
                        if line:
                            all_imgs.append(line)

                if not all_imgs:
                    st.error("This model requires image_urls. Upload or paste at least 1 image.")
                    st.stop()

                payload = {
                    "prompt": flux_prompt.strip(),
                    "image_urls": all_imgs,
                    "guidance_scale": float(flux_guidance),
                    "sync_mode": bool(flux_sync),
                    "num_images": int(flux_num),
                    "output_format": flux_format,
                    "safety_tolerance": flux_safety,
                    "enhance_prompt": bool(flux_enhance),
                }
                if int(flux_seed) != 0:
                    payload["seed"] = int(flux_seed)
                if flux_aspect != "(none)":
                    payload["aspect_ratio"] = flux_aspect

                result = call_fal_model(selected_model_id, payload)
                images = result.get("images") or []
                urls = [img.get("url") for img in images if img.get("url")]
                if not urls:
                    st.error("No images returned.")
                    st.stop()

                cols = st.columns(min(len(urls), 2))
                for i, url in enumerate(urls):
                    cols[i % len(cols)].image(url, use_column_width=True)

                add_history_item(selected_model_label, selected_model_id, "image", urls, {"prompt": flux_prompt.strip()})

            # -------- FLUX.2 PRO T2I --------
            elif selected_model_id == "fal-ai/flux-2-pro":
                if not (f2_prompt or "").strip():
                    st.error("Please enter a prompt.")
                    st.stop()

                payload = {
                    "prompt": f2_prompt.strip(),
                    "output_format": f2_format,
                    "sync_mode": bool(f2_sync),
                    "safety_tolerance": f2_safety,
                    "enable_safety_checker": bool(f2_enable_checker),
                }
                if f2_size_mode == "Preset":
                    payload["image_size"] = f2_image_size
                else:
                    payload["image_size"] = {"width": int(f2_custom_w), "height": int(f2_custom_h)}

                if int(f2_seed) != 0:
                    payload["seed"] = int(f2_seed)

                result = call_fal_model(selected_model_id, payload)
                images = result.get("images") or []
                urls = [img.get("url") for img in images if img.get("url")]
                if not urls:
                    st.error("No images returned.")
                    st.stop()

                cols = st.columns(min(len(urls), 2))
                for i, url in enumerate(urls):
                    cols[i % len(cols)].image(url, use_column_width=True)

                add_history_item(selected_model_label, selected_model_id, "image", urls, {"prompt": f2_prompt.strip()})

            # -------- FLUX.2 PRO EDIT --------
            elif selected_model_id == "fal-ai/flux-2-pro/edit":
                if not (f2e_prompt or "").strip():
                    st.error("Please enter a prompt.")
                    st.stop()
                if not f2e_images:
                    st.error("Please upload at least 1 image (image_urls required).")
                    st.stop()

                payload = {
                    "prompt": f2e_prompt.strip(),
                    "image_urls": [file_to_data_uri(f) for f in f2e_images],
                    "output_format": f2e_format,
                    "sync_mode": bool(f2e_sync),
                    "safety_tolerance": f2e_safety,
                    "enable_safety_checker": bool(f2e_enable_checker),
                }
                if f2e_size_mode == "Auto":
                    payload["image_size"] = "auto"
                elif f2e_size_mode == "Preset":
                    payload["image_size"] = f2e_image_size
                else:
                    payload["image_size"] = {"width": int(f2e_w), "height": int(f2e_h)}

                if int(f2e_seed) != 0:
                    payload["seed"] = int(f2e_seed)

                result = call_fal_model(selected_model_id, payload)
                images = result.get("images") or []
                urls = [img.get("url") for img in images if img.get("url")]
                if not urls:
                    st.error("No images returned.")
                    st.stop()

                cols = st.columns(min(len(urls), 2))
                for i, url in enumerate(urls):
                    cols[i % len(cols)].image(url, use_column_width=True)

                add_history_item(selected_model_label, selected_model_id, "image", urls, {"prompt": f2e_prompt.strip()})

            # -------- IDEOGRAM V3 REFRAME --------
            elif selected_model_id == "fal-ai/ideogram/v3/reframe":
                # Resolve required image_url
                if ideo_source_mode == "Upload":
                    if not ideo_upload:
                        st.error("Please upload an image.")
                        st.stop()
                    resolved_image_url = file_to_data_uri(ideo_upload)
                else:
                    if not (ideo_image_url_text or "").strip():
                        st.error("Please enter an image URL.")
                        st.stop()
                    resolved_image_url = ideo_image_url_text.strip()

                # Style rules:
                style_codes_list = []
                if (ideo_style_codes or "").strip():
                    style_codes_list = [s.strip() for s in ideo_style_codes.split(",") if s.strip()]

                # If style codes present, cannot use style or reference images
                if style_codes_list and ideo_ref_images:
                    st.warning("Ideogram: When Style Codes are provided, Style Reference Images are not allowed. Ignoring reference images.")
                    ideo_ref_images = None

                payload = {
                    "image_url": resolved_image_url,
                    "image_size": ideo_image_size_mode,
                    "rendering_speed": ideo_render_speed,
                    "num_images": int(ideo_num_images),
                    "sync_mode": bool(ideo_sync),
                }
                if int(ideo_seed) != 0:
                    payload["seed"] = int(ideo_seed)

                if ideo_palette and (ideo_palette.get("name") or ideo_palette.get("members")):
                    payload["color_palette"] = ideo_palette

                if style_codes_list:
                    payload["style_codes"] = style_codes_list
                else:
                    if ideo_style != "(none)":
                        payload["style"] = ideo_style
                    if (ideo_style_preset or "").strip():
                        payload["style_preset"] = ideo_style_preset.strip()
                    if ideo_ref_images:
                        payload["image_urls"] = [file_to_data_uri(f) for f in ideo_ref_images]

                result = call_fal_model(selected_model_id, payload)

                images = result.get("images") or []
                urls = [img.get("url") for img in images if img.get("url")]
                if not urls:
                    st.error("No images returned.")
                    st.stop()

                cols = st.columns(min(len(urls), 2))
                for i, url in enumerate(urls):
                    cols[i % len(cols)].image(url, use_column_width=True)

                add_history_item(
                    selected_model_label,
                    selected_model_id,
                    "image",
                    urls,
                    {
                        "image_size": ideo_image_size_mode,
                        "rendering_speed": ideo_render_speed,
                        "style": None if ideo_style == "(none)" else ideo_style,
                        "style_preset": (ideo_style_preset or "").strip() or None,
                    },
                )

            else:
                st.error("Model not wired yet.")
                st.stop()

    except Exception as e:
        st.error("Something went wrong while calling the FAL API.")
        st.code(str(e))
