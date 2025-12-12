import streamlit as st
import requests
import base64

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Multi-Model Media Platform",
    layout="wide",
)

st.title("🧪 Multi-Model Media Platform")

st.caption(
    "Switch between WAN Animate, Nano Banana Pro, and Nano Banana Pro Edit in one UI, "
    "configure settings, and call the FAL API directly."
)

# Your FAL key must be set in .streamlit/secrets.toml as:
# FAL_KEY = "your_api_key_here"
FAL_API_KEY = st.secrets["FAL_KEY"]

FAL_BASE_URL = "https://fal.run"

# -----------------------------
# HELPERS
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
# MODEL SELECTOR
# -----------------------------
MODEL_OPTIONS = {
    "WAN Animate (Video + Image → Video)": "fal-ai/wan/v2.2-14b/animate/move",
    "Nano Banana Pro (Text → Image)": "fal-ai/nano-banana-pro",
    "Nano Banana Pro Edit (Image + Text → Image)": "fal-ai/nano-banana-pro/edit",
}

selected_model_label = st.radio(
    "Choose model",
    list(MODEL_OPTIONS.keys()),
    horizontal=True,
)

selected_model_id = MODEL_OPTIONS[selected_model_label]

st.write("---")

# -----------------------------
# LAYOUT
# -----------------------------
left, right = st.columns([1, 1])

with right:
    st.header("🧾 Result")
    output_area = st.empty()
    extra_info = st.empty()

# -----------------------------
# LEFT PANEL – DYNAMIC UI
# -----------------------------
with left:
    st.header("⚙️ Input & Settings")

    # Common controls across models
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
                    cols = st.columns(len(images)) if len(images) > 1 else [output_area]
                    for i, img in enumerate(images):
                        url = img.get("url")
                        file_name = img.get("file_name", f"image_{i+1}")
                        if url:
                            cols[i].image(url, caption=file_name, use_column_width=True)
                    extra_info.write(result.get("description", ""))

            # ------- NANO BANANA PRO EDIT (I2I) -------
            elif selected_model_id == "fal-ai/nano-banana-pro/edit":
                if not edit_prompt.strip():
                    st.error("Please enter an edit prompt.")
                    st.stop()

                if not edit_images:
                    st.error("Please upload at least one image to edit.")
                    st.stop()

                # Convert uploaded images to data URIs for image_urls
                image_urls = [file_to_data_uri(f) for f in edit_images]

                payload = {
                    "prompt": edit_prompt.strip(),
                    "image_urls": image_urls,
                    "num_images": int(num_images),
                    "aspect_ratio": aspect_ratio if aspect_ratio != "auto" else None,
                    "resolution": resolution,
                    "output_format": output_format,
                }

                # Remove None entries (e.g. aspect_ratio when auto)
                payload = {k: v for k, v in payload.items() if v is not None}

                result = call_fal_model(selected_model_id, payload)

                images = result.get("images") or []
                if not images:
                    st.error("No images returned from Nano Banana Pro Edit.")
                else:
                    cols = st.columns(len(images)) if len(images) > 1 else [output_area]
                    for i, img in enumerate(images):
                        url = img.get("url")
                        file_name = img.get("file_name", f"edit_{i+1}")
                        if url:
                            cols[i].image(url, caption=file_name, use_column_width=True)
                    extra_info.write(result.get("description", ""))

    except Exception as e:
        st.error("Something went wrong while calling the FAL API.")
        st.code(str(e))
