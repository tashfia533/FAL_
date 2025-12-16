import streamlit as st
import requests
import base64
import os
import json
from datetime import datetime, timezone

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Multi-Model Media Platform", layout="wide")

st.title("Multi-Model Media Platform")
st.caption(
    "Switch between different FAL models in one UI, generate/edit media, "
    "and keep a local history (JSON) of outputs."
)

FAL_API_KEY = st.secrets.get("FAL_KEY")
if not FAL_API_KEY:
    st.error("Missing FAL_KEY. Add it in Streamlit Cloud → Settings → Secrets.")
    st.stop()

FAL_BASE_URL = "https://fal.run"
HISTORY_FILE = "history.json"
MAX_HISTORY_ITEMS = 500  # prevent history.json from growing forever

# -----------------------------
# UI STYLE (tiles + thumbnails)
# -----------------------------
st.markdown(
    """
<style>
/* Tile grid */
.tile-wrap {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px;
  padding: 10px;
  background: rgba(255,255,255,0.02);
}
.thumb {
  width: 100%;
  height: 140px;
  object-fit: cover;
  border-radius: 12px;
  border: 1px solid rgba(255,255,255,0.10);
  display: block;
  background: rgba(255,255,255,0.03);
}
video.thumb { background: rgba(255,255,255,0.03); }
.small-muted { opacity: 0.75; font-size: 12px; }
.section-title { margin-top: 8px; margin-bottom: 8px; }
hr.soft { border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 12px 0; }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# ZOOM STATE (shared)
# -----------------------------
if "zoom_media_url" not in st.session_state:
    st.session_state["zoom_media_url"] = None
if "zoom_media_kind" not in st.session_state:
    st.session_state["zoom_media_kind"] = None
if "zoom_media_meta" not in st.session_state:
    st.session_state["zoom_media_meta"] = None

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

def add_history_item(kind, model_id, urls, meta):
    if not urls:
        return
    history = load_history()
    history.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "kind": kind,           # kept for logic, but NOT displayed (per your request)
            "model": model_id,
            "urls": urls,
            "meta": meta or {},
        }
    )
    if len(history) > MAX_HISTORY_ITEMS:
        history = history[-MAX_HISTORY_ITEMS:]
    save_history(history)

def group_history(history):
    """
    Returns: dict(model_id -> list[entry]) newest-first within each model
    """
    grouped = {}
    for entry in reversed(history):  # newest first
        mid = entry.get("model", "unknown")
        grouped.setdefault(mid, []).append(entry)
    return grouped

# -----------------------------
# FAL HELPERS
# -----------------------------
def file_to_data_uri(uploaded_file) -> str:
    if uploaded_file is None:
        return None
    file_bytes = uploaded_file.read()
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    mime = uploaded_file.type or "application/octet-stream"
    return f"data:{mime};base64,{b64}"

def call_fal_model(model_id: str, payload: dict) -> dict:
    url = f"{FAL_BASE_URL}/{model_id}"
    headers = {"Authorization": f"Key {FAL_API_KEY}", "Content-Type": "application/json"}
    resp = requests.post(url, json=payload, headers=headers, timeout=600)
    if resp.status_code != 200:
        raise RuntimeError(f"FAL API error {resp.status_code}: {resp.text[:800]}")
    return resp.json()

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
    "FLUX Kontext Max Multi (Image + Text → Image)": "fal-ai/flux-pro/kontext/max/multi",
}
MODEL_ID_TO_LABEL = {v: k for k, v in MODEL_OPTIONS.items()}

# -----------------------------
# SIDEBAR NAV (Generator | History side-by-side)
# -----------------------------
with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio(
        "Navigation",
        ["Generator", "History"],
        horizontal=True,
        label_visibility="collapsed",
    )

# -----------------------------
# RENDER HELPERS (Tiles + Zoom)
# -----------------------------
def set_zoom(url, kind, meta):
    st.session_state["zoom_media_url"] = url
    st.session_state["zoom_media_kind"] = kind
    st.session_state["zoom_media_meta"] = meta or {}
    st.rerun()

def render_zoom_block(container):
    url = st.session_state.get("zoom_media_url")
    kind = st.session_state.get("zoom_media_kind")
    if not url:
        return

    with container:
        st.markdown('<hr class="soft">', unsafe_allow_html=True)
        st.subheader("🔎 Preview")

        if kind == "image":
            st.image(url, use_container_width=True)
        else:
            st.video(url)

        meta = st.session_state.get("zoom_media_meta") or {}
        if meta:
            with st.expander("Details"):
                st.json(meta)

        cols = st.columns([1, 1, 2])
        with cols[0]:
            if st.button("Close preview"):
                st.session_state["zoom_media_url"] = None
                st.session_state["zoom_media_kind"] = None
                st.session_state["zoom_media_meta"] = None
                st.rerun()

def render_history_tiles(grouped, *, max_tiles_per_model=12, tile_cols=3):
    """
    grouped: dict(model_id -> list[entry]) newest-first
    Shows tiles, and a View button to zoom.
    """
    for model_id, entries in grouped.items():
        label = MODEL_ID_TO_LABEL.get(model_id, model_id)
        st.markdown(f"<div class='section-title'><b>{label}</b></div>", unsafe_allow_html=True)

        # Flatten urls as tiles (newest-first)
        tiles = []
        for entry in entries:
            kind = entry.get("kind", "image")
            meta = entry.get("meta", {}) or {}
            for u in entry.get("urls", []) or []:
                tiles.append((u, kind, meta))
                if len(tiles) >= max_tiles_per_model:
                    break
            if len(tiles) >= max_tiles_per_model:
                break

        if not tiles:
            st.caption("No items yet.")
            st.markdown('<hr class="soft">', unsafe_allow_html=True)
            continue

        cols = st.columns(tile_cols)
        for i, (url, kind, meta) in enumerate(tiles):
            with cols[i % tile_cols]:
                st.markdown("<div class='tile-wrap'>", unsafe_allow_html=True)

                # Thumbnail
                if kind == "image":
                    st.markdown(f"<img class='thumb' src='{url}' />", unsafe_allow_html=True)
                else:
                    # Small fixed-height tile; click View for full player
                    st.markdown(
                        f"<video class='thumb' src='{url}' muted loop playsinline></video>",
                        unsafe_allow_html=True,
                    )

                # Tiny hint (optional)
                prompt = (meta.get("prompt") or meta.get("edit_prompt") or "").strip()
                if prompt:
                    st.markdown(
                        f"<div class='small-muted'>{prompt[:60]}{'…' if len(prompt) > 60 else ''}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown("<div class='small-muted'>&nbsp;</div>", unsafe_allow_html=True)

                # View button
                if st.button("View", key=f"view_{model_id}_{i}_{kind}"):
                    set_zoom(url, kind, meta)

                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<hr class="soft">', unsafe_allow_html=True)

# -----------------------------
# HISTORY PAGE
# -----------------------------
if page == "History":
    st.header("History")

    history = load_history()
    if not history:
        st.info("No history saved yet. Generate something first.")
        st.stop()

    # Optional: filter by model (kept simple)
    all_models = sorted({h.get("model", "unknown") for h in history})
    filter_label = st.selectbox(
        "Filter by model",
        ["All models"] + [MODEL_ID_TO_LABEL.get(mid, mid) for mid in all_models],
    )

    if filter_label != "All models":
        # map label back to id
        wanted_id = None
        for mid in all_models:
            if MODEL_ID_TO_LABEL.get(mid, mid) == filter_label:
                wanted_id = mid
                break
        history = [h for h in history if h.get("model") == wanted_id]

    grouped = group_history(history)

    # Full history: more tiles per model
    render_history_tiles(grouped, max_tiles_per_model=60, tile_cols=4)

    render_zoom_block(st.container())
    st.stop()

# -----------------------------
# GENERATOR PAGE (Left: inputs, Right: result + history panel)
# -----------------------------
selected_model_label = st.selectbox("Choose model", list(MODEL_OPTIONS.keys()))
selected_model_id = MODEL_OPTIONS[selected_model_label]

st.write("---")

left, right = st.columns([1, 1])

# Right side: Result + History (like your reference)
with right:
    st.header("Result")
    output_area = st.empty()
    extra_info = st.empty()

    st.markdown('<hr class="soft">', unsafe_allow_html=True)
    st.subheader("History (quick view)")

    # show grouped-by-model, but fewer tiles
    history_now = load_history()
    if history_now:
        grouped_now = group_history(history_now)
        render_history_tiles(grouped_now, max_tiles_per_model=8, tile_cols=3)
    else:
        st.caption("No history yet.")

    render_zoom_block(st.container())

# Left side: inputs
with left:
    st.header("Input & Settings")

    run_btn = st.button("🚀 Run", type="primary", use_container_width=True)
    reset_btn = st.button("🔄 Reset", use_container_width=True)
    if reset_btn:
        st.rerun()

    st.markdown("---")

    # ---------------- WAN ANIMATE ----------------
    if selected_model_id == "fal-ai/wan/v2.2-14b/animate/move":
        st.subheader("WAN Animate – Video + Image → Animated Video")

        video_file = st.file_uploader(
            "Upload Source Video",
            type=["mp4", "mov", "webm", "m4v", "gif"],
        )
        image_file = st.file_uploader(
            "Upload Reference Image",
            type=["png", "jpg", "jpeg", "webp", "gif", "avif"],
        )

        use_turbo = st.checkbox("Use Turbo", value=True)
        guidance_scale = st.slider("Guidance Scale", 0.0, 20.0, 1.0, 0.1)
        resolution = st.selectbox("Resolution", ["480p", "580p", "720p"], index=0)

        seed = st.number_input("Seed", min_value=0, max_value=2_147_483_647, value=1234)
        steps = st.number_input("Num Inference Steps", min_value=1, max_value=250, value=20)

        enable_safety = st.checkbox("Enable Safety Checker", value=True)
        enable_output_safety = st.checkbox("Enable Output Safety Checker", value=True)

        shift = st.slider("Shift", 1.0, 10.0, 5.0, 0.1)
        video_quality = st.selectbox("Video Quality", ["low", "medium", "high", "maximum"], index=2)
        video_mode = st.selectbox("Video Write Mode", ["fast", "balanced", "small"], index=1)
        return_zip = st.checkbox("Also return frames ZIP", value=False)

    # ---------------- NANO BANANA T2I ----------------
    elif selected_model_id == "fal-ai/nano-banana-pro":
        st.subheader("Nano Banana Pro – Text → Image")

        prompt = st.text_area("Prompt", height=120)
        aspect_ratio = st.selectbox(
            "Aspect Ratio",
            ["21:9", "16:9", "3:2", "4:3", "5:4", "1:1", "4:5", "3:4", "2:3", "9:16"],
            index=5,
        )
        resolution = st.selectbox("Resolution", ["1K", "2K", "4K"], index=0)
        num_images = st.slider("Number of Images", 1, 4, 1)
        output_format = st.selectbox("Output Format", ["png", "jpeg", "webp"], index=0)
        enable_web_search = st.checkbox("Enable Web Search", value=False)

    # ---------------- NANO BANANA EDIT ----------------
    elif selected_model_id == "fal-ai/nano-banana-pro/edit":
        st.subheader("Nano Banana Pro Edit – Image + Text → Image")

        edit_prompt = st.text_area("Edit Prompt", height=120)
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
        num_images = st.slider("Number of Edited Images", 1, 4, 1)
        output_format = st.selectbox("Output Format", ["png", "jpeg", "webp"], index=0)

    # ---------------- SEEDREAM 4.0 T2I ----------------
    elif selected_model_id == "fal-ai/bytedance/seedream/v4/text-to-image":
        st.subheader("Seedream 4.0 – Text → Image")

        sd_prompt = st.text_area("Prompt", height=120)
        sd_width = st.number_input("Width (px)", 512, 4096, 1280, 64)
        sd_height = st.number_input("Height (px)", 512, 4096, 1280, 64)

        sd_num_images = st.slider("Num Images", 1, 4, 1)
        sd_max_images = st.number_input("Max Images", 1, 8, 4)
        sd_seed = st.number_input("Seed (0=random)", 0, 2_147_483_647, 0)
        sd_sync_mode = st.checkbox("Sync Mode", value=False)
        sd_enable_safety = st.checkbox("Enable Safety Checker", value=True)
        sd_enhance_mode = st.selectbox("Enhance Prompt Mode", ["standard"], index=0)
        sd_output_format = st.selectbox("Output Format", ["png", "jpeg", "webp"], index=0)

    # ---------------- SEEDREAM 4.0 EDIT ----------------
    elif selected_model_id == "fal-ai/bytedance/seedream/v4/edit":
        st.subheader("Seedream 4.0 – Edit (Image + Text → Image)")

        sd_edit_prompt = st.text_area("Edit Prompt", height=120)
        sd_edit_images = st.file_uploader(
            "Upload Image(s) to Edit",
            type=["png", "jpg", "jpeg", "webp", "avif"],
            accept_multiple_files=True,
        )
        sd_width = st.number_input("Width (px)", 512, 4096, 1280, 64)
        sd_height = st.number_input("Height (px)", 512, 4096, 1280, 64)

        sd_num_images = st.slider("Num Images", 1, 4, 1)
        sd_max_images = st.number_input("Max Images", 1, 8, 4)
        sd_seed = st.number_input("Seed (0=random)", 0, 2_147_483_647, 0)
        sd_sync_mode = st.checkbox("Sync Mode", value=False)
        sd_enable_safety = st.checkbox("Enable Safety Checker", value=True)
        sd_enhance_mode = st.selectbox("Enhance Prompt Mode", ["standard"], index=0)
        sd_output_format = st.selectbox("Output Format", ["png", "jpeg", "webp"], index=0)

    # ---------------- SEEDREAM 4.5 T2I ----------------
    elif selected_model_id == "fal-ai/bytedance/seedream/v4.5/text-to-image":
        st.subheader("Seedream 4.5 – Text → Image")

        sd45_prompt = st.text_area("Prompt", height=120)
        sd45_image_size = st.selectbox(
            "Image Size",
            ["square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9", "auto_2K", "auto_4K"],
            index=6,
        )
        sd45_num_images = st.slider("Num Images", 1, 6, 1)
        sd45_max_images = st.number_input("Max Images", 1, 10, 1)
        sd45_seed = st.number_input("Seed (0=random)", 0, 2_147_483_647, 0)
        sd45_sync_mode = st.checkbox("Sync Mode", value=False)
        sd45_enable_safety = st.checkbox("Enable Safety Checker", value=True)

    # ---------------- SEEDREAM 4.5 EDIT ----------------
    elif selected_model_id == "fal-ai/bytedance/seedream/v4.5/edit":
        st.subheader("Seedream 4.5 – Edit (Image + Text → Image)")

        sd45_edit_prompt = st.text_area("Edit Prompt", height=120)
        sd45_edit_images = st.file_uploader(
            "Upload Image(s) to Edit",
            type=["png", "jpg", "jpeg", "webp", "avif"],
            accept_multiple_files=True,
        )
        sd45_image_size = st.selectbox(
            "Image Size",
            ["square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9", "auto_2K", "auto_4K"],
            index=7,
        )
        sd45_num_images = st.slider("Num Images", 1, 6, 1)
        sd45_max_images = st.number_input("Max Images", 1, 10, 1)
        sd45_seed = st.number_input("Seed (0=random)", 0, 2_147_483_647, 0)
        sd45_sync_mode = st.checkbox("Sync Mode", value=False)
        sd45_enable_safety = st.checkbox("Enable Safety Checker", value=True)

    # ---------------- FLUX Kontext Max Multi ----------------
    elif selected_model_id == "fal-ai/flux-pro/kontext/max/multi":
        st.subheader("FLUX Kontext Max Multi – Multi-image edit (Image + Text → Image)")

        flux_prompt = st.text_area("Prompt", height=120, placeholder="Put the little duckling on top of the woman's t-shirt.")
        flux_images = st.file_uploader(
            "Upload Image(s)",
            type=["png", "jpg", "jpeg", "webp", "avif"],
            accept_multiple_files=True,
            help="Upload 1–4 images (the model requires image_urls).",
        )

        flux_seed = st.number_input("Seed (0=random)", 0, 2_147_483_647, 0)
        flux_guidance = st.slider("Guidance scale (CFG)", 1.0, 20.0, 3.5, 0.1)
        flux_sync_mode = st.checkbox("Sync Mode", value=False)

        flux_num_images = st.slider("Num Images", 1, 4, 1)
        flux_output_format = st.selectbox("Output Format", ["jpeg", "png"], index=0)

        flux_safety_tol = st.selectbox("Safety Tolerance (API only)", ["1", "2", "3", "4", "5", "6"], index=1)
        flux_enhance = st.checkbox("Enhance Prompt", value=False)

        flux_aspect = st.selectbox(
            "Aspect Ratio",
            ["(auto)", "21:9", "16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16", "9:21"],
            index=0,
        )

# -----------------------------
# RUN LOGIC
# -----------------------------
if run_btn:
    try:
        with st.spinner("Calling FAL API…"):

            # WAN Animate
            if selected_model_id == "fal-ai/wan/v2.2-14b/animate/move":
                if not video_file or not image_file:
                    st.error("Please upload both a video and an image.")
                    st.stop()

                payload = {
                    "video_url": file_to_data_uri(video_file),
                    "image_url": file_to_data_uri(image_file),
                    "guidance_scale": guidance_scale,
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

                video_url = (result.get("video") or {}).get("url")
                if video_url:
                    output_area.video(video_url)
                    add_history_item(
                        kind="video",
                        model_id=selected_model_id,
                        urls=[video_url],
                        meta={"prompt": "WAN Animate", "seed": result.get("seed")},
                    )
                    st.rerun()
                else:
                    st.error("No video URL returned from WAN model.")

            # Nano Banana Pro T2I
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
                if not images:
                    st.error("No images returned.")
                    st.stop()

                urls = [img.get("url") for img in images if img.get("url")]
                cols = st.columns(min(len(urls), 2))
                for i, u in enumerate(urls):
                    cols[i % len(cols)].image(u, use_container_width=True)

                add_history_item(
                    kind="image",
                    model_id=selected_model_id,
                    urls=urls,
                    meta={"prompt": prompt.strip(), "aspect_ratio": aspect_ratio, "resolution": resolution},
                )
                st.rerun()

            # Nano Banana Pro Edit
            elif selected_model_id == "fal-ai/nano-banana-pro/edit":
                if not edit_prompt.strip():
                    st.error("Please enter an edit prompt.")
                    st.stop()
                if not edit_images:
                    st.error("Please upload at least one image.")
                    st.stop()

                image_urls = [file_to_data_uri(f) for f in edit_images]
                payload = {
                    "prompt": edit_prompt.strip(),
                    "image_urls": image_urls,
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
                for i, u in enumerate(urls):
                    cols[i % len(cols)].image(u, use_container_width=True)

                add_history_item(
                    kind="image",
                    model_id=selected_model_id,
                    urls=urls,
                    meta={"prompt": edit_prompt.strip(), "resolution": resolution},
                )
                st.rerun()

            # Seedream 4.0 T2I
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
                for i, u in enumerate(urls):
                    cols[i % len(cols)].image(u, use_container_width=True)

                add_history_item(
                    kind="image",
                    model_id=selected_model_id,
                    urls=urls,
                    meta={"prompt": sd_prompt.strip(), "image_size": {"width": int(sd_width), "height": int(sd_height)}},
                )
                st.rerun()

            # Seedream 4.0 Edit
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
                for i, u in enumerate(urls):
                    cols[i % len(cols)].image(u, use_container_width=True)

                add_history_item(
                    kind="image",
                    model_id=selected_model_id,
                    urls=urls,
                    meta={"prompt": sd_edit_prompt.strip(), "image_size": {"width": int(sd_width), "height": int(sd_height)}},
                )
                st.rerun()

            # Seedream 4.5 T2I
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
                for i, u in enumerate(urls):
                    cols[i % len(cols)].image(u, use_container_width=True)

                add_history_item(
                    kind="image",
                    model_id=selected_model_id,
                    urls=urls,
                    meta={"prompt": sd45_prompt.strip(), "image_size": sd45_image_size},
                )
                st.rerun()

            # Seedream 4.5 Edit
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
                for i, u in enumerate(urls):
                    cols[i % len(cols)].image(u, use_container_width=True)

                add_history_item(
                    kind="image",
                    model_id=selected_model_id,
                    urls=urls,
                    meta={"prompt": sd45_edit_prompt.strip(), "image_size": sd45_image_size},
                )
                st.rerun()

            # FLUX Kontext Max Multi
            elif selected_model_id == "fal-ai/flux-pro/kontext/max/multi":
                if not flux_prompt.strip():
                    st.error("Please enter a prompt.")
                    st.stop()
                if not flux_images:
                    st.error("Please upload at least one image (this model requires image_urls).")
                    st.stop()

                image_urls = [file_to_data_uri(f) for f in flux_images]
                payload = {
                    "prompt": flux_prompt.strip(),
                    "image_urls": image_urls,
                    "guidance_scale": float(flux_guidance),
                    "sync_mode": bool(flux_sync_mode),
                    "num_images": int(flux_num_images),
                    "output_format": flux_output_format,
                    "safety_tolerance": flux_safety_tol,
                    "enhance_prompt": bool(flux_enhance),
                }
                if int(flux_seed) != 0:
                    payload["seed"] = int(flux_seed)
                if flux_aspect != "(auto)":
                    payload["aspect_ratio"] = flux_aspect

                result = call_fal_model(selected_model_id, payload)
                images = result.get("images") or []
                urls = [img.get("url") for img in images if img.get("url")]
                if not urls:
                    st.error("No images returned.")
                    st.stop()

                cols = st.columns(min(len(urls), 2))
                for i, u in enumerate(urls):
                    cols[i % len(cols)].image(u, use_container_width=True)

                add_history_item(
                    kind="image",
                    model_id=selected_model_id,
                    urls=urls,
                    meta={
                        "prompt": flux_prompt.strip(),
                        "guidance_scale": float(flux_guidance),
                        "aspect_ratio": None if flux_aspect == "(auto)" else flux_aspect,
                        "output_format": flux_output_format,
                        "num_images": int(flux_num_images),
                        "safety_tolerance": flux_safety_tol,
                    },
                )
                st.rerun()

    except Exception as e:
        st.error("Something went wrong while calling the FAL API.")
        st.code(str(e))
