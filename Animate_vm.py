import streamlit as st
import requests
import base64
import os
import json
from datetime import datetime, timezone
from collections import defaultdict

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Multi-Model Media Platform",
    layout="wide",
)

st.title("Multi-Model Media Platform")

st.caption(
    "Switch between different models to Edit in one UI, "
    "configure settings, and call the FAL API directly."
)

# Your FAL key must be set in .streamlit/secrets.toml as:
# FAL_KEY = "your_api_key_here"
FAL_API_KEY = st.secrets.get("FAL_KEY")
if not FAL_API_KEY:
    st.error("Missing FAL_KEY. Add it in Streamlit Cloud → Settings → Secrets.")
    st.stop()

FAL_BASE_URL = "https://fal.run"
HISTORY_FILE = "history.json"   # JSON "database"

# -----------------------------
# ZOOM STATE (for History view)
# -----------------------------
if "zoom_media_url" not in st.session_state:
    st.session_state["zoom_media_url"] = None
if "zoom_media_meta" not in st.session_state:
    st.session_state["zoom_media_meta"] = None
if "zoom_media_kind" not in st.session_state:
    st.session_state["zoom_media_kind"] = None

# -----------------------------
# HISTORY HELPERS
# -----------------------------
def load_history():
    """Load history entries from JSON file."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_history(history):
    """Save the full history list back to JSON file."""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"Could not save history: {e}")


def add_history_item(kind, model_id, urls, meta):
    """
    Append a history entry.

    kind: "image" | "video"
    model_id: e.g. "fal-ai/nano-banana-pro"
    urls: list of URLs (image or video)
    meta: dict with extra info, e.g. {"prompt": "...", "seed": 123}
    """
    if not urls:
        return

    history = load_history()
    history.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "model": model_id,
            "urls": urls,
            "meta": meta or {},
        }
    )
    save_history(history)

# -----------------------------
# FAL HELPERS
# -----------------------------
def file_to_data_uri(uploaded_file) -> str:
    """
    Convert a Streamlit UploadedFile to a base64 data URI that FAL accepts
    wherever it expects a *url field (video_url, image_url, image_urls, etc).
    """
    if uploaded_file is None:
        return None
    file_bytes = uploaded_file.read()
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    mime = uploaded_file.type or "application/octet-stream"
    return f"data:{mime};base64,{b64}"


def call_fal_model(model_id: str, payload: dict) -> dict:
    """
    Call a FAL model synchronously via https://fal.run/<model_id>
    using your FAL API key.
    """
    url = f"{FAL_BASE_URL}/{model_id}"

    headers = {
        "Authorization": f"Key {FAL_API_KEY}",
        "Content-Type": "application/json",
    }

    response = requests.post(url, json=payload, headers=headers, timeout=600)

    if response.status_code != 200:
        raise RuntimeError(
            f"FAL API error {response.status_code}: {response.text[:500]}"
        )

    return response.json()


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
}
MODEL_ID_TO_LABEL = {v: k for k, v in MODEL_OPTIONS.items()}

# -----------------------------
# HISTORY PAGE RENDERING
# -----------------------------
def render_history_page():
    st.header("📚 Generation History")

    history = load_history()
    if not history:
        st.info("No history saved yet. Generate something first on the Generator page.")
        return

    # Optional filter: All models or a specific one
    model_filter = st.selectbox(
        "Filter by model",
        ["All models"] + sorted({h["model"] for h in history}),
        index=0,
    )

    # Filter by type (image / video / all)
    kind_filter = st.selectbox(
        "Filter by type",
        ["All", "image", "video"],
        index=0,
    )

    # Apply filters
    filtered = []
    for item in history:
        if model_filter != "All models" and item["model"] != model_filter:
            continue
        if kind_filter != "All" and item["kind"] != kind_filter:
            continue
        filtered.append(item)

    if not filtered:
        st.info("No entries match your filters.")
        return

    # Group filtered entries by model
    grouped = defaultdict(list)
    for entry in filtered:
        grouped[entry["model"]].append(entry)

    # Zoomed item at top
    zoom_url = st.session_state.get("zoom_media_url")
    zoom_kind = st.session_state.get("zoom_media_kind")
    zoom_meta = st.session_state.get("zoom_media_meta")

    if zoom_url:
        st.markdown("---")
        st.subheader("🔎 Selected item")
        if zoom_kind == "image":
            st.image(zoom_url, use_column_width=True)
        elif zoom_kind == "video":
            st.video(zoom_url)

        if zoom_meta:
            with st.expander("Details"):
                st.json(zoom_meta)

        if st.button("Close", key="close_zoom"):
            st.session_state["zoom_media_url"] = None
            st.session_state["zoom_media_meta"] = None
            st.session_state["zoom_media_kind"] = None
            st.rerun()

    st.markdown("---")

    # Order model sections by latest timestamp
    model_order = []
    for mid, entries in grouped.items():
        last_ts = max(e.get("timestamp", "") for e in entries)
        model_order.append((last_ts, mid))
    model_order.sort(reverse=True)

    for _, model_id in model_order:
        entries = sorted(
            grouped[model_id], key=lambda e: e.get("timestamp", ""), reverse=True
        )
        label = MODEL_ID_TO_LABEL.get(model_id, model_id)

        st.markdown(f"## {label}")

        for idx, entry in enumerate(entries):
            ts = entry.get("timestamp", "")[:19].replace("T", " ")
            kind = entry.get("kind", "")
            urls = entry.get("urls", [])
            meta = entry.get("meta", {})

            st.markdown(
                f"**Time (UTC):** {ts}  ·  **Type:** `{kind}`"
            )

            if not urls:
                st.write("_No media URLs found for this entry._")
                continue

            # IMAGE TILES
            if kind == "image":
                cols = st.columns(min(max(len(urls), 1), 4))
                for i, url in enumerate(urls):
                    with cols[i % len(cols)]:
                        st.image(url, width=180)
                        if st.button("🔍 View", key=f"zoom_img_{model_id}_{idx}_{i}"):
                            st.session_state["zoom_media_url"] = url
                            st.session_state["zoom_media_meta"] = meta
                            st.session_state["zoom_media_kind"] = "image"
                            st.rerun()

            # VIDEO TILES
            elif kind == "video":
                cols = st.columns(min(max(len(urls), 1), 3))
                for i, url in enumerate(urls):
                    with cols[i % len(cols)]:
                        # Video rendered in a narrow column, so it appears as a tile
                        st.video(url)
                        if st.button("🔍 View", key=f"zoom_vid_{model_id}_{idx}_{i}"):
                            st.session_state["zoom_media_url"] = url
                            st.session_state["zoom_media_meta"] = meta
                            st.session_state["zoom_media_kind"] = "video"
                            st.rerun()

            with st.expander("Details"):
                st.json(meta)

        st.markdown("---")


# -----------------------------
# SIDEBAR NAV
# -----------------------------
page = st.sidebar.radio("Page", ["Generator", "History"], index=0)

# -----------------------------
# HISTORY PAGE
# -----------------------------
if page == "History":
    render_history_page()
    st.stop()   # don't render generator below


# =============================
# GENERATOR PAGE
# =============================

# Layout for generator
left, right = st.columns([1, 1])

with left:
    st.header("⚙️ Input & Settings")

    selected_model_label = st.selectbox(
        "Choose model",
        list(MODEL_OPTIONS.keys()),
    )
    selected_model_id = MODEL_OPTIONS[selected_model_label]

    run_btn = st.button("🚀 Run", type="primary", use_container_width=True)
    reset_btn = st.button("🔄 Reset", use_container_width=True)

    if reset_btn:
        st.rerun()

    st.markdown("---")

    # --------------- WAN ANIMATE ---------------
    if selected_model_id == "fal-ai/wan/v2.2-14b/animate/move":
        st.subheader("WAN Animate – Video + Image → Animated Video")

        video_file = st.file_uploader(
            "Upload Source Video",
            type=["mp4", "mov", "webm", "m4v", "gif"],
            help="This video’s motion will drive the animation.",
        )

        image_file = st.file_uploader(
            "Upload Reference Image",
            type=["png", "jpg", "jpeg", "webp", "gif", "avif"],
            help="The character/subject from this image will be animated using the video motion.",
        )

        st.markdown("### General Settings")

        use_turbo = st.checkbox(
            "Use Turbo",
            value=True,
            help="Improves quality while keeping generation fast by auto-optimising settings.",
        )

        guidance_scale = st.slider(
            "Guidance Scale",
            min_value=0.0,
            max_value=20.0,
            value=1.0,
            step=0.1,
            help="Higher = sticks closer to motion/style, but may reduce visual richness. Default: 1.0",
        )

        resolution = st.selectbox(
            "Resolution",
            ["480p", "580p", "720p"],
            index=0,
            help="Output video resolution. Higher resolutions look better but take longer and cost more.",
        )

        seed = st.number_input(
            "Seed",
            min_value=0,
            max_value=2_147_483_647,
            value=1234,
            help="Use the same seed to reproduce results. Different seeds = variations.",
        )

        steps = st.number_input(
            "Number of Inference Steps",
            min_value=1,
            max_value=250,
            value=20,
            help="More steps → higher quality but slower. Default: 20.",
        )

        st.markdown("### Safety & Output")

        enable_safety = st.checkbox(
            "Enable Safety Checker",
            value=True,
            help="Checks input for unsafe content before processing.",
        )

        enable_output_safety = st.checkbox(
            "Enable Output Safety Checker",
            value=True,
            help="Checks generated video for unsafe content before returning.",
        )

        shift = st.slider(
            "Shift",
            min_value=1.0,
            max_value=10.0,
            value=5.0,
            step=0.1,
            help="Controls motion intensity. Higher values can produce more dramatic movement.",
        )

        video_quality = st.selectbox(
            "Video Quality",
            ["low", "medium", "high", "maximum"],
            index=2,
            help="Higher quality = better visuals but larger files & more compute.",
        )

        video_mode = st.selectbox(
            "Video Write Mode",
            ["fast", "balanced", "small"],
            index=1,
            help=(
                "fast: quicker but larger files • "
                "balanced: good default • "
                "small: smallest file, slower."
            ),
        )

        return_zip = st.checkbox(
            "Also return frames ZIP",
            value=False,
            help="If enabled, FAL also returns a ZIP of per-frame images.",
        )

    # --------------- NANO BANANA PRO (T2I) ---------------
    elif selected_model_id == "fal-ai/nano-banana-pro":
        st.subheader("Nano Banana Pro – Text → Image")

        prompt = st.text_area(
            "Prompt",
            placeholder="A cinematic shot of a neon-lit street at night, reflections on wet pavement...",
            help="Describe the image you want in natural language.",
        )

        aspect_ratio = st.selectbox(
            "Aspect Ratio",
            ["21:9", "16:9", "3:2", "4:3", "5:4", "1:1", "4:5", "3:4", "2:3", "9:16"],
            index=5,
            help="Shape of the output image. E.g. 16:9 for landscape, 9:16 for vertical/phone.",
        )

        resolution = st.selectbox(
            "Resolution",
            ["1K", "2K", "4K"],
            index=0,
            help="Higher resolutions are sharper but slower and more expensive.",
        )

        num_images = st.slider(
            "Number of Images",
            min_value=1,
            max_value=4,
            value=1,
            help="Generate multiple candidates in one go.",
        )

        output_format = st.selectbox(
            "Output Format",
            ["png", "jpeg", "webp"],
            index=0,
            help="PNG is lossless; JPEG/WebP are usually smaller.",
        )

        enable_web_search = st.checkbox(
            "Enable Web Search",
            value=False,
            help="Let the model pull in up-to-date info from the web if needed.",
        )

    # --------------- NANO BANANA PRO EDIT (I2I) ---------------
    elif selected_model_id == "fal-ai/nano-banana-pro/edit":
        st.subheader("Nano Banana Pro Edit – Image + Text → Edited Image")

        edit_prompt = st.text_area(
            "Edit Prompt",
            placeholder="Make the scene look like golden hour with warm lighting and add a rainbow in the background.",
            help="Describe how you want to transform the uploaded image(s).",
        )

        edit_images = st.file_uploader(
            "Upload Image(s) to Edit",
            type=["png", "jpg", "jpeg", "webp", "avif"],
            accept_multiple_files=True,
            help="You can upload 1–4 images. The model will use them as references to edit.",
        )

        aspect_ratio = st.selectbox(
            "Aspect Ratio",
            ["auto", "21:9", "16:9", "3:2", "4:3", "5:4", "1:1", "4:5", "3:4", "2:3", "9:16"],
            index=0,
            help="auto will infer a good aspect ratio from your inputs.",
        )

        resolution = st.selectbox(
            "Resolution",
            ["1K", "2K", "4K"],
            index=0,
            help="Higher resolutions are sharper but slower and more expensive.",
        )

        num_images = st.slider(
            "Number of Edited Images",
            min_value=1,
            max_value=4,
            value=1,
            help="How many edited outputs to generate.",
        )

        output_format = st.selectbox(
            "Output Format",
            ["png", "jpeg", "webp"],
            index=0,
            help="PNG is lossless; JPEG/WebP are usually smaller.",
        )

    # --------------- SEEDREAM 4.0 (T2I) ---------------
    elif selected_model_id == "fal-ai/bytedance/seedream/v4/text-to-image":
        st.subheader("Seedream 4.0 – Text → Image")

        sd_prompt = st.text_area(
            "Prompt",
            placeholder="A hyper-realistic photo of a sunset over a snowy mountain range, shot on 85mm lens.",
            help="Main description of the image you want Seedream 4.0 to create.",
        )

        st.markdown("### Image Size")

        sd_width = st.number_input(
            "Width (px)",
            min_value=512,
            max_value=4096,
            value=1280,
            step=64,
            help="Width of the generated image in pixels.",
        )

        sd_height = st.number_input(
            "Height (px)",
            min_value=512,
            max_value=4096,
            value=1280,
            step=64,
            help="Height of the generated image in pixels.",
        )

        st.markdown("### Additional Settings")

        sd_num_images = st.slider(
            "Num Images",
            min_value=1,
            max_value=4,
            value=1,
            help="How many images to generate in this request.",
        )

        sd_max_images = st.number_input(
            "Max Images",
            min_value=1,
            max_value=8,
            value=4,
            help="Upper bound for images the model is allowed to generate.",
        )

        sd_seed = st.number_input(
            "Seed",
            min_value=0,
            max_value=2_147_483_647,
            value=0,
            help="Set a seed for reproducible outputs. 0 = random.",
        )

        sd_sync_mode = st.checkbox(
            "Sync Mode",
            value=False,
            help="If true, media may be returned as base64 only and not stored in history.",
        )

        sd_enable_safety = st.checkbox(
            "Enable Safety Checker",
            value=True,
            help="If true, the model will run safety checks on the generated content.",
        )

        sd_enhance_mode = st.selectbox(
            "Enhance Prompt Mode",
            ["standard"],
            index=0,
            help="Prompt enhancement setting. Currently using 'standard' as in the FAL UI.",
        )

        sd_output_format = st.selectbox(
            "Output Format",
            ["png", "jpeg", "webp"],
            index=0,
            help="Format of the generated images.",
        )

    # --------------- SEEDREAM 4.0 EDIT (I2I) ---------------
    elif selected_model_id == "fal-ai/bytedance/seedream/v4/edit":
        st.subheader("Seedream 4.0 Edit – Image + Text → Edited Image")

        sd_edit_prompt = st.text_area(
            "Edit Prompt",
            placeholder="Turn the photo into a cinematic nighttime cityscape with neon reflections.",
            help="Describe how Seedream should transform the input image(s).",
        )

        sd_edit_images = st.file_uploader(
            "Upload Image(s) to Edit",
            type=["png", "jpg", "jpeg", "webp", "avif"],
            accept_multiple_files=True,
            help="Drag and drop or upload 1–4 images to edit.",
        )

        st.markdown("### Image Size")

        sd_width = st.number_input(
            "Width (px)",
            min_value=512,
            max_value=4096,
            value=1280,
            step=64,
            help="Width of the generated edited image in pixels.",
        )

        sd_height = st.number_input(
            "Height (px)",
            min_value=512,
            max_value=4096,
            value=1280,
            step=64,
            help="Height of the generated edited image in pixels.",
        )

        st.markdown("### Additional Settings")

        sd_num_images = st.slider(
            "Num Images",
            min_value=1,
            max_value=4,
            value=1,
            help="How many edited images to generate.",
        )

        sd_max_images = st.number_input(
            "Max Images",
            min_value=1,
            max_value=8,
            value=4,
            help="Upper bound for images the model is allowed to generate.",
        )

        sd_seed = st.number_input(
            "Seed",
            min_value=0,
            max_value=2_147_483_647,
            value=0,
            help="Set a seed for reproducible outputs. 0 = random.",
        )

        sd_sync_mode = st.checkbox(
            "Sync Mode",
            value=False,
            help="If true, media may be returned as base64 only and not stored in history.",
        )

        sd_enable_safety = st.checkbox(
            "Enable Safety Checker",
            value=True,
            help="If true, the model will run safety checks on the generated content.",
        )

        sd_enhance_mode = st.selectbox(
            "Enhance Prompt Mode",
            ["standard"],
            index=0,
            help="Prompt enhancement setting. Currently using 'standard' as in the FAL UI.",
        )

        sd_output_format = st.selectbox(
            "Output Format",
            ["png", "jpeg", "webp"],
            index=0,
            help="Format of the generated images.",
        )

    # --------------- SEEDREAM 4.5 (T2I) ---------------
    elif selected_model_id == "fal-ai/bytedance/seedream/v4.5/text-to-image":
        st.subheader("Seedream 4.5 – Text → Image")

        sd45_prompt = st.text_area(
            "Prompt",
            placeholder="A selfie of a cat at twilight by the Eiffel Tower, holding baklava, 4:3 aspect ratio...",
            help="Text prompt for Seedream 4.5.",
        )

        st.markdown("### Image Size")

        sd45_image_size = st.selectbox(
            "Image Size",
            [
                "square_hd",
                "square",
                "portrait_4_3",
                "portrait_16_9",
                "landscape_4_3",
                "landscape_16_9",
                "auto_2K",
                "auto_4K",
            ],
            index=6,  # auto_2K
            help="Use Auto 2K/4K for automatic high-res sizing, or pick a fixed aspect preset.",
        )

        st.markdown("### Additional Settings")

        sd45_num_images = st.slider(
            "Num Images",
            min_value=1,
            max_value=6,
            value=1,
            help="Number of separate generations to run for this prompt.",
        )

        sd45_max_images = st.number_input(
            "Max Images",
            min_value=1,
            max_value=10,
            value=1,
            help="Upper bound of variations per generation (total images ≤ num_images × max_images).",
        )

        sd45_seed = st.number_input(
            "Seed",
            min_value=0,
            max_value=2_147_483_647,
            value=0,
            help="Random seed. 0 = let the API choose a random seed.",
        )

        sd45_sync_mode = st.checkbox(
            "Sync Mode",
            value=False,
            help="If true, images may be returned as data URIs and not stored in history.",
        )

        sd45_enable_safety = st.checkbox(
            "Enable Safety Checker",
            value=True,
            help="If true, the safety checker is enabled (recommended).",
        )

    # --------------- SEEDREAM 4.5 EDIT (I2I) ---------------
    elif selected_model_id == "fal-ai/bytedance/seedream/v4.5/edit":
        st.subheader("Seedream 4.5 Edit – Image + Text → Edited Image")

        sd45_edit_prompt = st.text_area(
            "Edit Prompt",
            placeholder="Replace the product in image 1 with image 2, copy the title text from image 3 to the top...",
            help="Natural language instructions for multi-image editing.",
        )

        sd45_edit_images = st.file_uploader(
            "Upload Image(s) to Edit",
            type=["png", "jpg", "jpeg", "webp", "avif"],
            accept_multiple_files=True,
            help="Upload up to 10 images. The model can reference them as 'Figure 1', 'Figure 2', etc.",
        )

        st.markdown("### Image Size")

        sd45_image_size = st.selectbox(
            "Image Size",
            [
                "square_hd",
                "square",
                "portrait_4_3",
                "portrait_16_9",
                "landscape_4_3",
                "landscape_16_9",
                "auto_2K",
                "auto_4K",
            ],
            index=7,  # auto_4K
            help="Auto 4K for highest quality, or pick a preset aspect ratio.",
        )

        st.markdown("### Additional Settings")

        sd45_num_images = st.slider(
            "Num Images",
            min_value=1,
            max_value=6,
            value=1,
            help="Number of separate generations to run.",
        )

        sd45_max_images = st.number_input(
            "Max Images",
            min_value=1,
            max_value=10,
            value=1,
            help="Upper bound of images per generation. Total images ≤ num_images × max_images.",
        )

        sd45_seed = st.number_input(
            "Seed",
            min_value=0,
            max_value=2_147_483_647,
            value=0,
            help="Random seed. 0 = let the API choose.",
        )

        sd45_sync_mode = st.checkbox(
            "Sync Mode",
            value=False,
            help="If true, media may be returned as data URIs and not stored in history.",
        )

        sd45_enable_safety = st.checkbox(
            "Enable Safety Checker",
            value=True,
            help="If true, the safety checker is enabled (recommended).",
        )

# RIGHT COLUMN: result only
with right:
    st.header("🧾 Result")
    output_area = st.empty()
    extra_info = st.empty()

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

                video_uri = file_to_data_uri(video_file)
                image_uri = file_to_data_uri(image_file)

                payload = {
                    "video_url": video_uri,
                    "image_url": image_uri,
                    "guidance_scale": guidance_scale,
                    "resolution": resolution,
                    "seed": seed,
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

                frames_zip = result.get("frames_zip") or {}
                frames_zip_url = frames_zip.get("url")

                if video_url:
                    output_area.video(video_url)
                    extra_info.write(f"Seed used: `{result.get('seed')}`")

                    add_history_item(
                        kind="video",
                        model_id=selected_model_id,
                        urls=[video_url],
                        meta={
                            "seed": result.get("seed"),
                            "guidance_scale": guidance_scale,
                            "resolution": resolution,
                        },
                    )

                elif frames_zip_url:
                    zip_bytes = requests.get(frames_zip_url).content
                    output_area.download_button(
                        label="Download Frames ZIP",
                        data=zip_bytes,
                        file_name="frames.zip",
                    )
                else:
                    st.error("No video URL returned from WAN model.")

            # ------- NANO BANANA PRO (T2I) -------
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
                    st.error("No images returned from Nano Banana Pro.")
                else:
                    urls = []
                    cols = st.columns(len(images)) if len(images) > 1 else [output_area]
                    for i, img in enumerate(images):
                        url = img.get("url")
                        if not url:
                            continue
                        urls.append(url)
                        file_name = img.get("file_name", f"image_{i+1}")
                        cols[i % len(cols)].image(url, caption=file_name, use_column_width=True)
                    extra_info.write(result.get("description", ""))

                    add_history_item(
                        kind="image",
                        model_id=selected_model_id,
                        urls=urls,
                        meta={
                            "prompt": prompt.strip(),
                            "aspect_ratio": aspect_ratio,
                            "resolution": resolution,
                        },
                    )

            # ------- NANO BANANA PRO EDIT (I2I) -------
            elif selected_model_id == "fal-ai/nano-banana-pro/edit":
                if not edit_prompt.strip():
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
                    "aspect_ratio": aspect_ratio if aspect_ratio != "auto" else None,
                    "resolution": resolution,
                    "output_format": output_format,
                }

                payload = {k: v for k, v in payload.items() if v is not None}

                result = call_fal_model(selected_model_id, payload)

                images = result.get("images") or []
                if not images:
                    st.error("No images returned from Nano Banana Pro Edit.")
                else:
                    urls = []
                    cols = st.columns(len(images)) if len(images) > 1 else [output_area]
                    for i, img in enumerate(images):
                        url = img.get("url")
                        if not url:
                            continue
                        urls.append(url)
                        file_name = img.get("file_name", f"edit_{i+1}")
                        cols[i % len(cols)].image(url, caption=file_name, use_column_width=True)
                    extra_info.write(result.get("description", ""))

                    add_history_item(
                        kind="image",
                        model_id=selected_model_id,
                        urls=urls,
                        meta={
                            "prompt": edit_prompt.strip(),
                            "resolution": resolution,
                        },
                    )

            # ------- SEEDREAM 4.0 (T2I) -------
            elif selected_model_id == "fal-ai/bytedance/seedream/v4/text-to-image":
                if not sd_prompt.strip():
                    st.error("Please enter a prompt.")
                    st.stop()

                image_size = {"width": int(sd_width), "height": int(sd_height)}

                payload = {
                    "prompt": sd_prompt.strip(),
                    "image_size": image_size,
                    "num_images": int(sd_num_images),
                    "max_images": int(sd_max_images),
                    "seed": int(sd_seed) if sd_seed != 0 else None,
                    "sync_mode": bool(sd_sync_mode),
                    "enable_safety_checker": bool(sd_enable_safety),
                    "enhance_prompt_mode": sd_enhance_mode,
                    "output_format": sd_output_format,
                }

                payload = {k: v for k, v in payload.items() if v is not None}

                result = call_fal_model(selected_model_id, payload)

                images = result.get("images") or []
                if not images:
                    st.error("No images returned from Seedream 4.0.")
                else:
                    urls = []
                    cols = st.columns(len(images)) if len(images) > 1 else [output_area]
                    for i, img in enumerate(images):
                        url = img.get("url")
                        if not url:
                            continue
                        urls.append(url)
                        file_name = img.get("file_name", f"seedream_{i+1}")
                        cols[i % len(cols)].image(url, caption=file_name, use_column_width=True)
                    if "seed" in result:
                        extra_info.write(f"Seed used: `{result['seed']}`")

                    add_history_item(
                        kind="image",
                        model_id=selected_model_id,
                        urls=urls,
                        meta={
                            "prompt": sd_prompt.strip(),
                            "image_size": image_size,
                        },
                    )

            # ------- SEEDREAM 4.0 EDIT (I2I) -------
            elif selected_model_id == "fal-ai/bytedance/seedream/v4/edit":
                if not sd_edit_prompt.strip():
                    st.error("Please enter an edit prompt.")
                    st.stop()

                if not sd_edit_images:
                    st.error("Please upload at least one image to edit.")
                    st.stop()

                image_urls = [file_to_data_uri(f) for f in sd_edit_images]

                image_size = {"width": int(sd_width), "height": int(sd_height)}

                payload = {
                    "prompt": sd_edit_prompt.strip(),
                    "image_urls": image_urls,
                    "image_size": image_size,
                    "num_images": int(sd_num_images),
                    "max_images": int(sd_max_images),
                    "seed": int(sd_seed) if sd_seed != 0 else None,
                    "sync_mode": bool(sd_sync_mode),
                    "enable_safety_checker": bool(sd_enable_safety),
                    "enhance_prompt_mode": sd_enhance_mode,
                    "output_format": sd_output_format,
                }

                payload = {k: v for k, v in payload.items() if v is not None}

                result = call_fal_model(selected_model_id, payload)

                images = result.get("images") or []
                if not images:
                    st.error("No images returned from Seedream 4.0 Edit.")
                else:
                    urls = []
                    cols = st.columns(len(images)) if len(images) > 1 else [output_area]
                    for i, img in enumerate(images):
                        url = img.get("url")
                        if not url:
                            continue
                        urls.append(url)
                        file_name = img.get("file_name", f"seedream_edit_{i+1}")
                        cols[i % len(cols)].image(url, caption=file_name, use_column_width=True)
                    if "seed" in result:
                        extra_info.write(f"Seed used: `{result['seed']}`")

                    add_history_item(
                        kind="image",
                        model_id=selected_model_id,
                        urls=urls,
                        meta={
                            "prompt": sd_edit_prompt.strip(),
                            "image_size": image_size,
                        },
                    )

            # ------- SEEDREAM 4.5 (T2I) -------
            elif selected_model_id == "fal-ai/bytedance/seedream/v4.5/text-to-image":
                if not sd45_prompt.strip():
                    st.error("Please enter a prompt.")
                    st.stop()

                payload = {
                    "prompt": sd45_prompt.strip(),
                    "image_size": sd45_image_size,
                    "num_images": int(sd45_num_images),
                    "max_images": int(sd45_max_images),
                    "seed": int(sd45_seed) if sd45_seed != 0 else None,
                    "sync_mode": bool(sd45_sync_mode),
                    "enable_safety_checker": bool(sd45_enable_safety),
                }

                payload = {k: v for k, v in payload.items() if v is not None}

                result = call_fal_model(selected_model_id, payload)

                images = result.get("images") or []
                if not images:
                    st.error("No images returned from Seedream 4.5.")
                else:
                    urls = []
                    cols = st.columns(len(images)) if len(images) > 1 else [output_area]
                    for i, img in enumerate(images):
                        url = img.get("url")
                        if not url:
                            continue
                        urls.append(url)
                        file_name = img.get("file_name", f"seedream_v45_{i+1}")
                        cols[i % len(cols)].image(url, caption=file_name, use_column_width=True)
                    if "seed" in result:
                        extra_info.write(f"Seed used: `{result['seed']}`")

                    add_history_item(
                        kind="image",
                        model_id=selected_model_id,
                        urls=urls,
                        meta={"prompt": sd45_prompt.strip(), "image_size": sd45_image_size},
                    )

            # ------- SEEDREAM 4.5 EDIT (I2I) -------
            elif selected_model_id == "fal-ai/bytedance/seedream/v4.5/edit":
                if not sd45_edit_prompt.strip():
                    st.error("Please enter an edit prompt.")
                    st.stop()

                if not sd45_edit_images:
                    st.error("Please upload at least one image to edit.")
                    st.stop()

                image_urls = [file_to_data_uri(f) for f in sd45_edit_images]

                payload = {
                    "prompt": sd45_edit_prompt.strip(),
                    "image_urls": image_urls,
                    "image_size": sd45_image_size,
                    "num_images": int(sd45_num_images),
                    "max_images": int(sd45_max_images),
                    "seed": int(sd45_seed) if sd45_seed != 0 else None,
                    "sync_mode": bool(sd45_sync_mode),
                    "enable_safety_checker": bool(sd45_enable_safety),
                }

                payload = {k: v for k, v in payload.items() if v is not None}

                result = call_fal_model(selected_model_id, payload)

                images = result.get("images") or []
                if not images:
                    st.error("No images returned from Seedream 4.5 Edit.")
                else:
                    urls = []
                    cols = st.columns(len(images)) if len(images) > 1 else [output_area]
                    for i, img in enumerate(images):
                        url = img.get("url")
                        if not url:
                            continue
                        urls.append(url)
                        file_name = img.get("file_name", f"seedream_v45_edit_{i+1}")
                        cols[i % len(cols)].image(url, caption=file_name, use_column_width=True)
                    if "seed" in result:
                        extra_info.write(f"Seed used: `{result['seed']}`")

                    add_history_item(
                        kind="image",
                        model_id=selected_model_id,
                        urls=urls,
                        meta={"prompt": sd45_edit_prompt.strip(), "image_size": sd45_image_size},
                    )

    except Exception as e:
        st.error("Something went wrong while calling the FAL API.")
        st.code(str(e))


