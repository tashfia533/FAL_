import streamlit as st
import requests
import base64
import os
import json
import time
from datetime import datetime, timezone

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Multi-Model Media Platform",
    layout="wide",
)

st.title("Multi-Model Media Platform")
st.caption(
    "Switch between different models in one UI, configure settings, call the FAL API directly, "
    "and browse a model-grouped history."
)

# Secrets
FAL_API_KEY = st.secrets.get("FAL_KEY")
if not FAL_API_KEY:
    st.error("Missing FAL_KEY. Add it in Streamlit Cloud → Settings → Secrets.")
    st.stop()

FAL_RUN_BASE_URL = "https://fal.run"
FAL_QUEUE_BASE_URL = "https://queue.fal.run"

# NOTE: On Streamlit Cloud, local files can reset on redeploy/restart.
HISTORY_FILE = "history.json"

# -----------------------------
# SESSION STATE
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "Generator"

if "zoom_media_url" not in st.session_state:
    st.session_state.zoom_media_url = None
if "zoom_media_kind" not in st.session_state:
    st.session_state.zoom_media_kind = None
if "zoom_media_meta" not in st.session_state:
    st.session_state.zoom_media_meta = None
if "zoom_media_model" not in st.session_state:
    st.session_state.zoom_media_model = None

# -----------------------------
# MODEL OPTIONS
# -----------------------------
MODEL_OPTIONS = {
    "WAN Animate (Video + Image → Video)": "fal-ai/wan/v2.2-14b/animate/move",
    "Nano Banana Pro (Text → Image)": "fal-ai/nano-banana-pro",
    "Nano Banana Pro Edit (Image + Text → Image)": "fal-ai/nano-banana-pro/edit",
    "Seedream 4.0 (Text → Image)": "fal-ai/bytedance/seedream/v4/text-to-image",
    "Seedream 4.0 Edit (Image + Text → Image)": "fal-ai/bytedance/seedream/v4/edit",
    "Seedream 4.5 (Text → Image)": "fal-ai/bytedance/seedream/v4.5/text-to-image",
    "Seedream 4.5 Edit (Image + Text → Image)": "fal-ai/bytedance/seedream/v4.5/edit",

    # ✅ New additions
    "FLUX Kontext Max Multi (Edit / Multi-Image)": "fal-ai/flux-pro/kontext/max/multi",
    "FLUX.2 Pro (Text → Image)": "fal-ai/flux-2-pro",
    "FLUX.2 Pro Edit (Image + Text → Image)": "fal-ai/flux-2-pro/edit",
}

MODEL_LABEL_BY_ID = {v: k for k, v in MODEL_OPTIONS.items()}

# -----------------------------
# HISTORY HELPERS
# -----------------------------
def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_history(history):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"Could not save history: {e}")

def add_history_item(kind, model_id, urls, meta=None):
    if not urls:
        return
    history = load_history()
    history.append(
        {
            # kept internally (not shown in UI), useful later if you want sorting
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "kind": kind,              # "image" | "video"
            "model": model_id,
            "urls": urls,              # list[str]
            "meta": meta or {},        # dict
        }
    )
    save_history(history)

# -----------------------------
# FILE → DATA URI
# -----------------------------
def file_to_data_uri(uploaded_file) -> str:
    if uploaded_file is None:
        return None
    file_bytes = uploaded_file.read()
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    mime = uploaded_file.type or "application/octet-stream"
    return f"data:{mime};base64,{b64}"

# -----------------------------
# FAL CALLER (supports both fal.run + queue.fal.run)
# -----------------------------
def _headers():
    return {
        "Authorization": f"Key {FAL_API_KEY}",
        "Content-Type": "application/json",
    }

def poll_queue(model_id: str, request_id: str, timeout_s: int = 900, poll_every_s: float = 1.5):
    """
    Polls queue.fal.run until completed, then fetches final output.
    """
    status_url = f"{FAL_QUEUE_BASE_URL}/{model_id}/requests/{request_id}/status"
    result_url = f"{FAL_QUEUE_BASE_URL}/{model_id}/requests/{request_id}"

    start = time.time()
    while True:
        if time.time() - start > timeout_s:
            raise TimeoutError("Timed out waiting for the FAL queue job to complete.")

        s = requests.get(status_url, headers=_headers(), timeout=60).json()
        status = s.get("status")

        if status == "COMPLETED":
            return requests.get(result_url, headers=_headers(), timeout=60).json()

        if status in ("IN_QUEUE", "IN_PROGRESS"):
            time.sleep(poll_every_s)
            continue

        # Unknown / error case
        raise RuntimeError(f"Unexpected queue status: {s}")

def call_fal_model(model_id: str, payload: dict) -> dict:
    """
    Tries fal.run (sync). If endpoint is queue-based (returns request_id/status)
    or fal.run fails, falls back to queue.fal.run and polls.
    """
    # 1) Try fal.run first (works for many models)
    try:
        r = requests.post(f"{FAL_RUN_BASE_URL}/{model_id}", json=payload, headers=_headers(), timeout=120)
        if r.status_code == 200:
            data = r.json()
            # Queue-style response?
            if isinstance(data, dict) and data.get("request_id") and data.get("status"):
                return poll_queue(model_id, data["request_id"])
            return data
    except Exception:
        pass

    # 2) Fall back to queue.fal.run
    r = requests.post(f"{FAL_QUEUE_BASE_URL}/{model_id}", json=payload, headers=_headers(), timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"FAL API error {r.status_code}: {r.text[:800]}")

    data = r.json()
    if not data.get("request_id"):
        # Some queue endpoints might still return final output, but usually not.
        return data
    return poll_queue(model_id, data["request_id"])

# -----------------------------
# SIDEBAR NAV (side-by-side buttons)
# -----------------------------
st.sidebar.markdown("### Navigation")
c1, c2 = st.sidebar.columns(2)
if c1.button("🧪 Generator", use_container_width=True):
    st.session_state.page = "Generator"
if c2.button("📚 History", use_container_width=True):
    st.session_state.page = "History"

page = st.session_state.page

# -----------------------------
# SMALL TILE RENDERERS (History)
# -----------------------------
def render_image_tile(url: str, key: str):
    st.image(url, width=180)
    if st.button("Open", key=key, use_container_width=True):
        st.session_state.zoom_media_url = url
        st.session_state.zoom_media_kind = "image"

def render_video_tile(url: str, key: str):
    # Small video tile (HTML) so it doesn't blow up the layout
    st.markdown(
        f"""
        <video width="180" controls muted playsinline
               style="border-radius:10px; border:1px solid rgba(49,51,63,0.2);">
          <source src="{url}">
        </video>
        """,
        unsafe_allow_html=True
    )
    if st.button("Open", key=key, use_container_width=True):
        st.session_state.zoom_media_url = url
        st.session_state.zoom_media_kind = "video"

# -----------------------------
# HISTORY PAGE
# -----------------------------
if page == "History":
    st.subheader("History")

    history = load_history()
    if not history:
        st.info("No history saved yet. Generate something on the Generator page first.")
        st.stop()

    # Group by model (latest first)
    history = list(reversed(history))

    grouped = {}
    for item in history:
        grouped.setdefault(item["model"], []).append(item)

    # Zoom view (top)
    if st.session_state.zoom_media_url:
        with st.container():
            st.markdown("#### Zoom view")
            if st.session_state.zoom_media_kind == "image":
                st.image(st.session_state.zoom_media_url, use_column_width=True)
            else:
                st.video(st.session_state.zoom_media_url)

            if st.button("Close", use_container_width=True):
                st.session_state.zoom_media_url = None
                st.session_state.zoom_media_kind = None
                st.rerun()

        st.markdown("---")

    # Filters
    model_ids = list(grouped.keys())
    model_labels = [MODEL_LABEL_BY_ID.get(mid, mid) for mid in model_ids]
    picked = st.selectbox("Filter by model", ["All models"] + sorted(model_labels))

    # Render sections
    for model_id, items in grouped.items():
        label = MODEL_LABEL_BY_ID.get(model_id, model_id)

        if picked != "All models" and picked != label:
            continue

        st.markdown(f"### {label}")

        # Flatten media from items (keep order newest→older)
        media = []
        for it in items:
            kind = it.get("kind")
            meta = it.get("meta", {})
            for u in it.get("urls", []):
                media.append((u, kind, meta))

        # Tiles
        cols = st.columns(4)
        for idx, (url, kind, meta) in enumerate(media):
            with cols[idx % 4]:
                if kind == "image":
                    render_image_tile(url, key=f"hist_open_img_{model_id}_{idx}")
                else:
                    render_video_tile(url, key=f"hist_open_vid_{model_id}_{idx}")

        st.markdown("---")

    st.stop()

# -----------------------------
# GENERATOR PAGE
# -----------------------------
selected_model_label = st.selectbox("Choose model", list(MODEL_OPTIONS.keys()))
selected_model_id = MODEL_OPTIONS[selected_model_label]

st.write("---")

left, right = st.columns([1, 1], gap="large")

with right:
    st.header("Result")
    output_area = st.empty()
    extra_info = st.empty()

with left:
    st.header("Input & Settings")

    run_btn = st.button("🚀 Run", type="primary", use_container_width=True)
    reset_btn = st.button("🔄 Reset", use_container_width=True)
    if reset_btn:
        st.rerun()

    st.markdown("---")

    # ========== WAN ANIMATE ==========
    if selected_model_id == "fal-ai/wan/v2.2-14b/animate/move":
        st.subheader("WAN Animate – Video + Image → Video")

        video_file = st.file_uploader("Upload Source Video", type=["mp4", "mov", "webm", "m4v", "gif"])
        image_file = st.file_uploader("Upload Reference Image", type=["png", "jpg", "jpeg", "webp", "gif", "avif"])

        use_turbo = st.checkbox("Use Turbo", value=True)
        guidance_scale = st.slider("Guidance Scale", 0.0, 20.0, 1.0, 0.1)
        resolution = st.selectbox("Resolution", ["480p", "580p", "720p"], index=0)
        seed = st.number_input("Seed", 0, 2_147_483_647, 1234)
        steps = st.number_input("Inference Steps", 1, 250, 20)

        enable_safety = st.checkbox("Enable Safety Checker", value=True)
        enable_output_safety = st.checkbox("Enable Output Safety Checker", value=True)
        shift = st.slider("Shift", 1.0, 10.0, 5.0, 0.1)
        video_quality = st.selectbox("Video Quality", ["low", "medium", "high", "maximum"], index=2)
        video_mode = st.selectbox("Video Write Mode", ["fast", "balanced", "small"], index=1)
        return_zip = st.checkbox("Return Frames ZIP", value=False)

    # ========== NANO BANANA T2I ==========
    elif selected_model_id == "fal-ai/nano-banana-pro":
        st.subheader("Nano Banana Pro – Text → Image")

        prompt = st.text_area("Prompt")
        aspect_ratio = st.selectbox("Aspect Ratio", ["21:9","16:9","3:2","4:3","5:4","1:1","4:5","3:4","2:3","9:16"], index=5)
        resolution = st.selectbox("Resolution", ["1K", "2K", "4K"], index=0)
        num_images = st.slider("Num Images", 1, 4, 1)
        output_format = st.selectbox("Output Format", ["png", "jpeg", "webp"], index=0)
        enable_web_search = st.checkbox("Enable Web Search", value=False)

    # ========== NANO BANANA EDIT ==========
    elif selected_model_id == "fal-ai/nano-banana-pro/edit":
        st.subheader("Nano Banana Pro Edit – Image + Text → Image")

        edit_prompt = st.text_area("Edit Prompt")
        edit_images = st.file_uploader("Upload Image(s)", type=["png", "jpg", "jpeg", "webp", "avif"], accept_multiple_files=True)

        aspect_ratio = st.selectbox("Aspect Ratio", ["auto","21:9","16:9","3:2","4:3","5:4","1:1","4:5","3:4","2:3","9:16"], index=0)
        resolution = st.selectbox("Resolution", ["1K", "2K", "4K"], index=0)
        num_images = st.slider("Num Images", 1, 4, 1)
        output_format = st.selectbox("Output Format", ["png", "jpeg", "webp"], index=0)

    # ========== SEEDREAM 4.0 T2I ==========
    elif selected_model_id == "fal-ai/bytedance/seedream/v4/text-to-image":
        st.subheader("Seedream 4.0 – Text → Image")

        sd_prompt = st.text_area("Prompt")

        sd_width = st.number_input("Width", 512, 4096, 1280, 64)
        sd_height = st.number_input("Height", 512, 4096, 1280, 64)

        sd_num_images = st.slider("Num Images", 1, 4, 1)
        sd_max_images = st.number_input("Max Images", 1, 8, 4)
        sd_seed = st.number_input("Seed (0=random)", 0, 2_147_483_647, 0)
        sd_sync_mode = st.checkbox("Sync Mode", value=False)
        sd_enable_safety = st.checkbox("Enable Safety Checker", value=True)
        sd_enhance_mode = st.selectbox("Enhance Prompt Mode", ["standard"], index=0)
        sd_output_format = st.selectbox("Output Format", ["png", "jpeg", "webp"], index=0)

    # ========== SEEDREAM 4.0 EDIT ==========
    elif selected_model_id == "fal-ai/bytedance/seedream/v4/edit":
        st.subheader("Seedream 4.0 Edit – Image + Text → Image")

        sd_edit_prompt = st.text_area("Edit Prompt")
        sd_edit_images = st.file_uploader("Upload Image(s)", type=["png", "jpg", "jpeg", "webp", "avif"], accept_multiple_files=True)

        sd_width = st.number_input("Width", 512, 4096, 1280, 64)
        sd_height = st.number_input("Height", 512, 4096, 1280, 64)

        sd_num_images = st.slider("Num Images", 1, 4, 1)
        sd_max_images = st.number_input("Max Images", 1, 8, 4)
        sd_seed = st.number_input("Seed (0=random)", 0, 2_147_483_647, 0)
        sd_sync_mode = st.checkbox("Sync Mode", value=False)
        sd_enable_safety = st.checkbox("Enable Safety Checker", value=True)
        sd_enhance_mode = st.selectbox("Enhance Prompt Mode", ["standard"], index=0)
        sd_output_format = st.selectbox("Output Format", ["png", "jpeg", "webp"], index=0)

    # ========== SEEDREAM 4.5 T2I ==========
    elif selected_model_id == "fal-ai/bytedance/seedream/v4.5/text-to-image":
        st.subheader("Seedream 4.5 – Text → Image")

        sd45_prompt = st.text_area("Prompt")
        sd45_image_size = st.selectbox(
            "Image Size",
            ["square_hd","square","portrait_4_3","portrait_16_9","landscape_4_3","landscape_16_9","auto_2K","auto_4K"],
            index=6
        )
        sd45_num_images = st.slider("Num Images", 1, 6, 1)
        sd45_max_images = st.number_input("Max Images", 1, 10, 1)
        sd45_seed = st.number_input("Seed (0=random)", 0, 2_147_483_647, 0)
        sd45_sync_mode = st.checkbox("Sync Mode", value=False)
        sd45_enable_safety = st.checkbox("Enable Safety Checker", value=True)

    # ========== SEEDREAM 4.5 EDIT ==========
    elif selected_model_id == "fal-ai/bytedance/seedream/v4.5/edit":
        st.subheader("Seedream 4.5 Edit – Image + Text → Image")

        sd45_edit_prompt = st.text_area("Edit Prompt")
        sd45_edit_images = st.file_uploader("Upload Image(s)", type=["png", "jpg", "jpeg", "webp", "avif"], accept_multiple_files=True)

        sd45_image_size = st.selectbox(
            "Image Size",
            ["square_hd","square","portrait_4_3","portrait_16_9","landscape_4_3","landscape_16_9","auto_2K","auto_4K"],
            index=7
        )
        sd45_num_images = st.slider("Num Images", 1, 6, 1)
        sd45_max_images = st.number_input("Max Images", 1, 10, 1)
        sd45_seed = st.number_input("Seed (0=random)", 0, 2_147_483_647, 0)
        sd45_sync_mode = st.checkbox("Sync Mode", value=False)
        sd45_enable_safety = st.checkbox("Enable Safety Checker", value=True)

    # ========== FLUX KONTEXT MAX MULTI (EDIT) ==========
    elif selected_model_id == "fal-ai/flux-pro/kontext/max/multi":
        st.subheader("FLUX Kontext Max Multi – Multi-Image Edit")

        flux_k_prompt = st.text_area("Prompt", placeholder="Put the little duckling on top of the woman's t-shirt.")
        flux_k_images = st.file_uploader("Upload Image(s)", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)

        flux_k_guidance = st.slider("Guidance scale (CFG)", 0.0, 10.0, 3.5, 0.1)
        flux_k_seed = st.number_input("Seed (0=random)", 0, 2_147_483_647, 0)
        flux_k_num_images = st.slider("Num Images", 1, 4, 1)
        flux_k_output_format = st.selectbox("Output Format", ["jpeg", "png"], index=0)
        flux_k_sync_mode = st.checkbox("Sync Mode", value=False)
        flux_k_safety_tolerance = st.selectbox("Safety Tolerance", ["1","2","3","4","5"], index=1)
        flux_k_enhance_prompt = st.checkbox("Enhance Prompt", value=False)
        flux_k_aspect_ratio = st.selectbox("Aspect Ratio (optional)", ["(none)","1:1","4:3","3:4","16:9","9:16"], index=0)

    # ========== FLUX.2 PRO (T2I) ==========
    elif selected_model_id == "fal-ai/flux-2-pro":
        st.subheader("FLUX.2 Pro – Text → Image")

        flux2_prompt = st.text_area("Prompt")

        flux2_image_size_mode = st.selectbox(
            "Image Size",
            ["landscape_4_3","landscape_16_9","portrait_4_3","portrait_16_9","square","square_hd","auto","custom"],
            index=0
        )
        if flux2_image_size_mode == "custom":
            flux2_w = st.number_input("Width", 256, 14142, 1024, 64)
            flux2_h = st.number_input("Height", 256, 14142, 768, 64)

        flux2_seed = st.number_input("Seed (0=random)", 0, 2_147_483_647, 0)
        flux2_safety_tolerance = st.selectbox("Safety Tolerance", ["1","2","3","4","5"], index=1)
        flux2_enable_safety = st.checkbox("Enable Safety Checker", value=True)
        flux2_output_format = st.selectbox("Output Format", ["jpeg", "png"], index=0)
        flux2_sync_mode = st.checkbox("Sync Mode", value=False)

    # ========== FLUX.2 PRO EDIT (I2I) ==========
    elif selected_model_id == "fal-ai/flux-2-pro/edit":
        st.subheader("FLUX.2 Pro Edit – Image + Text → Image")

        flux2e_prompt = st.text_area("Prompt", placeholder="Place realistic flames emerging from the top of the coffee cup...")
        flux2e_images = st.file_uploader("Upload Image(s)", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)

        flux2e_image_size_mode = st.selectbox(
            "Image Size",
            ["auto","landscape_4_3","landscape_16_9","portrait_4_3","portrait_16_9","square","square_hd","custom"],
            index=0
        )
        if flux2e_image_size_mode == "custom":
            flux2e_w = st.number_input("Width", 256, 14142, 1024, 64)
            flux2e_h = st.number_input("Height", 256, 14142, 1024, 64)

        flux2e_seed = st.number_input("Seed (0=random)", 0, 2_147_483_647, 0)
        flux2e_safety_tolerance = st.selectbox("Safety Tolerance", ["1","2","3","4","5"], index=1)
        flux2e_enable_safety = st.checkbox("Enable Safety Checker", value=True)
        flux2e_output_format = st.selectbox("Output Format", ["jpeg", "png"], index=0)
        flux2e_sync_mode = st.checkbox("Sync Mode", value=False)

# -----------------------------
# RUN LOGIC
# -----------------------------
if run_btn:
    try:
        with st.spinner("Calling FAL API…"):

            # ------- WAN ANIMATE -------
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
                add_history_item("video", selected_model_id, [video_url], meta={"seed": result.get("seed")})

            # ------- NANO BANANA PRO -------
            elif selected_model_id == "fal-ai/nano-banana-pro":
                if not prompt.strip():
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
                urls = [img.get("url") for img in images if img.get("url")]

                if not urls:
                    st.error("No images returned.")
                    st.stop()

                cols = st.columns(min(len(urls), 2))
                for i, url in enumerate(urls):
                    cols[i % len(cols)].image(url, use_column_width=True)

                add_history_item("image", selected_model_id, urls, meta={"prompt": prompt.strip()})

            # ------- NANO BANANA EDIT -------
            elif selected_model_id == "fal-ai/nano-banana-pro/edit":
                if not edit_prompt.strip():
                    st.error("Please enter an edit prompt.")
                    st.stop()
                if not edit_images:
                    st.error("Please upload at least one image.")
                    st.stop()

                payload = {
                    "prompt": edit_prompt.strip(),
                    "image_urls": [file_to_data_uri(f) for f in edit_images],
                    "num_images": int(num_images),
                    "resolution": resolution,
                    "output_format": output_format,
                }
                if aspect_ratio != "auto":
                    payload["aspect_ratio"] = aspect_ratio

                result = call_fal_model(selected_model_id, payload)
                images = result.get("images") or []
                urls = [img.get("url") for img in images if img.get("url")]

                if not urls:
                    st.error("No images returned.")
                    st.stop()

                cols = st.columns(min(len(urls), 2))
                for i, url in enumerate(urls):
                    cols[i % len(cols)].image(url, use_column_width=True)

                add_history_item("image", selected_model_id, urls, meta={"prompt": edit_prompt.strip()})

            # ------- SEEDREAM 4.0 T2I -------
            elif selected_model_id == "fal-ai/bytedance/seedream/v4/text-to-image":
                if not sd_prompt.strip():
                    st.error("Please enter a prompt.")
                    st.stop()

                payload = {
                    "prompt": sd_prompt.strip(),
                    "image_size": {"width": int(sd_width), "height": int(sd_height)},
                    "num_images": int(sd_num_images),
                    "max_images": int(sd_max_images),
                    "sync_mode": bool(sd_sync_mode),
                    "enable_safety_checker": bool(sd_enable_safety),
                    "enhance_prompt_mode": sd_enhance_mode,
                    "output_format": sd_output_format,
                }
                if int(sd_seed) != 0:
                    payload["seed"] = int(sd_seed)

                result = call_fal_model(selected_model_id, payload)
                images = result.get("images") or []
                urls = [img.get("url") for img in images if img.get("url")]

                if not urls:
                    st.error("No images returned.")
                    st.stop()

                cols = st.columns(min(len(urls), 2))
                for i, url in enumerate(urls):
                    cols[i % len(cols)].image(url, use_column_width=True)

                add_history_item("image", selected_model_id, urls, meta={"prompt": sd_prompt.strip()})

            # ------- SEEDREAM 4.0 EDIT -------
            elif selected_model_id == "fal-ai/bytedance/seedream/v4/edit":
                if not sd_edit_prompt.strip():
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
                    "sync_mode": bool(sd_sync_mode),
                    "enable_safety_checker": bool(sd_enable_safety),
                    "enhance_prompt_mode": sd_enhance_mode,
                    "output_format": sd_output_format,
                }
                if int(sd_seed) != 0:
                    payload["seed"] = int(sd_seed)

                result = call_fal_model(selected_model_id, payload)
                images = result.get("images") or []
                urls = [img.get("url") for img in images if img.get("url")]

                if not urls:
                    st.error("No images returned.")
                    st.stop()

                cols = st.columns(min(len(urls), 2))
                for i, url in enumerate(urls):
                    cols[i % len(cols)].image(url, use_column_width=True)

                add_history_item("image", selected_model_id, urls, meta={"prompt": sd_edit_prompt.strip()})

            # ------- SEEDREAM 4.5 T2I -------
            elif selected_model_id == "fal-ai/bytedance/seedream/v4.5/text-to-image":
                if not sd45_prompt.strip():
                    st.error("Please enter a prompt.")
                    st.stop()

                payload = {
                    "prompt": sd45_prompt.strip(),
                    "image_size": sd45_image_size,
                    "num_images": int(sd45_num_images),
                    "max_images": int(sd45_max_images),
                    "sync_mode": bool(sd45_sync_mode),
                    "enable_safety_checker": bool(sd45_enable_safety),
                }
                if int(sd45_seed) != 0:
                    payload["seed"] = int(sd45_seed)

                result = call_fal_model(selected_model_id, payload)
                images = result.get("images") or []
                urls = [img.get("url") for img in images if img.get("url")]

                if not urls:
                    st.error("No images returned.")
                    st.stop()

                cols = st.columns(min(len(urls), 2))
                for i, url in enumerate(urls):
                    cols[i % len(cols)].image(url, use_column_width=True)

                add_history_item("image", selected_model_id, urls, meta={"prompt": sd45_prompt.strip()})

            # ------- SEEDREAM 4.5 EDIT -------
            elif selected_model_id == "fal-ai/bytedance/seedream/v4.5/edit":
                if not sd45_edit_prompt.strip():
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
                    "sync_mode": bool(sd45_sync_mode),
                    "enable_safety_checker": bool(sd45_enable_safety),
                }
                if int(sd45_seed) != 0:
                    payload["seed"] = int(sd45_seed)

                result = call_fal_model(selected_model_id, payload)
                images = result.get("images") or []
                urls = [img.get("url") for img in images if img.get("url")]

                if not urls:
                    st.error("No images returned.")
                    st.stop()

                cols = st.columns(min(len(urls), 2))
                for i, url in enumerate(urls):
                    cols[i % len(cols)].image(url, use_column_width=True)

                add_history_item("image", selected_model_id, urls, meta={"prompt": sd45_edit_prompt.strip()})

            # ------- FLUX KONTEXT MAX MULTI -------
            elif selected_model_id == "fal-ai/flux-pro/kontext/max/multi":
                if not flux_k_prompt.strip():
                    st.error("Please enter a prompt.")
                    st.stop()
                if not flux_k_images:
                    st.error("Please upload at least one image.")
                    st.stop()

                payload = {
                    "prompt": flux_k_prompt.strip(),
                    "image_urls": [file_to_data_uri(f) for f in flux_k_images],
                    "guidance_scale": float(flux_k_guidance),
                    "num_images": int(flux_k_num_images),
                    "output_format": flux_k_output_format,
                    "sync_mode": bool(flux_k_sync_mode),
                    "safety_tolerance": flux_k_safety_tolerance,
                    "enhance_prompt": bool(flux_k_enhance_prompt),
                }
                if int(flux_k_seed) != 0:
                    payload["seed"] = int(flux_k_seed)
                if flux_k_aspect_ratio != "(none)":
                    payload["aspect_ratio"] = flux_k_aspect_ratio

                result = call_fal_model(selected_model_id, payload)
                images = result.get("images") or []
                urls = [img.get("url") for img in images if img.get("url")]

                if not urls:
                    st.error("No images returned.")
                    st.stop()

                cols = st.columns(min(len(urls), 2))
                for i, url in enumerate(urls):
                    cols[i % len(cols)].image(url, use_column_width=True)

                add_history_item("image", selected_model_id, urls, meta={"prompt": flux_k_prompt.strip()})

            # ------- FLUX.2 PRO (T2I) -------
            elif selected_model_id == "fal-ai/flux-2-pro":
                if not flux2_prompt.strip():
                    st.error("Please enter a prompt.")
                    st.stop()

                if flux2_image_size_mode == "custom":
                    image_size = {"width": int(flux2_w), "height": int(flux2_h)}
                else:
                    image_size = flux2_image_size_mode

                payload = {
                    "prompt": flux2_prompt.strip(),
                    "image_size": image_size,
                    "output_format": flux2_output_format,
                    "sync_mode": bool(flux2_sync_mode),
                    "safety_tolerance": flux2_safety_tolerance,
                    "enable_safety_checker": bool(flux2_enable_safety),
                }
                if int(flux2_seed) != 0:
                    payload["seed"] = int(flux2_seed)

                result = call_fal_model(selected_model_id, payload)
                images = result.get("images") or []
                urls = [img.get("url") for img in images if img.get("url")]

                if not urls:
                    st.error("No images returned.")
                    st.stop()

                cols = st.columns(min(len(urls), 2))
                for i, url in enumerate(urls):
                    cols[i % len(cols)].image(url, use_column_width=True)

                add_history_item("image", selected_model_id, urls, meta={"prompt": flux2_prompt.strip()})

            # ------- FLUX.2 PRO EDIT -------
            elif selected_model_id == "fal-ai/flux-2-pro/edit":
                if not flux2e_prompt.strip():
                    st.error("Please enter a prompt.")
                    st.stop()
                if not flux2e_images:
                    st.error("Please upload at least one image.")
                    st.stop()

                if flux2e_image_size_mode == "custom":
                    image_size = {"width": int(flux2e_w), "height": int(flux2e_h)}
                else:
                    image_size = flux2e_image_size_mode

                payload = {
                    "prompt": flux2e_prompt.strip(),
                    "image_urls": [file_to_data_uri(f) for f in flux2e_images],
                    "image_size": image_size,
                    "output_format": flux2e_output_format,
                    "sync_mode": bool(flux2e_sync_mode),
                    "safety_tolerance": flux2e_safety_tolerance,
                    "enable_safety_checker": bool(flux2e_enable_safety),
                }
                if int(flux2e_seed) != 0:
                    payload["seed"] = int(flux2e_seed)

                result = call_fal_model(selected_model_id, payload)
                images = result.get("images") or []
                urls = [img.get("url") for img in images if img.get("url")]

                if not urls:
                    st.error("No images returned.")
                    st.stop()

                cols = st.columns(min(len(urls), 2))
                for i, url in enumerate(urls):
                    cols[i % len(cols)].image(url, use_column_width=True)

                add_history_item("image", selected_model_id, urls, meta={"prompt": flux2e_prompt.strip()})

            extra_info.success("Saved to history ✅")

    except Exception as e:
        st.error("Something went wrong while calling the FAL API.")
        st.code(str(e))

