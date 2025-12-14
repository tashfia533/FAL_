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
    "Switch between different models to edit/generate in one UI, "
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
# HISTORY RENDER FUNCTION
# -----------------------------
def render_history_panel():
    """Render grouped history on the right side, like tiles."""
    history = load_history()
    if not history:
        st.info("No history yet. Generate something on the left.")
        return

    # Group by model
    grouped = defaultdict(list)
    for entry in history:
        grouped[entry["model"]].append(entry)

    # Show selected item (zoomed) at very top if any
    zoom_url = st.session_state.get("zoom_media_url")
    zoom_kind = st.session_state.get("zoom_media_kind")
    zoom_meta = st.session_state.get("zoom_media_meta")

    if zoom_url:
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
            st.experimental_rerun()

        st.markdown("---")

    # Now sections by model
    # Sort models so most recently used model appears first
    # (based on last timestamp of each group)
    model_order = []
    for mid, entries in grouped.items():
        last_ts = max(e.get("timestamp", "") for e in entries)
        model_order.append((last_ts, mid))
    model_order.sort(reverse=True)

    for _, model_id in model_order:
        entries = grouped[model_id]
        label = MODEL_ID_TO_LABEL.get(model_id, model_id)
        st.markdown(f"### {label}")

        # newest first within each model
        entries_sorted = sorted(
            entries, key=lambda e: e.get("timestamp", ""), reverse=True
        )

        for idx, entry in enumerate(entries_sorted):
            ts = entry.get("timestamp", "")[:19].replace("T", " ")
            kind = entry.get("kind", "")
            urls = entry.get("urls", [])
            meta = entry.get("meta", {})

            with st.container():
                st.markdown(
                    f"**Run time (UTC):** {ts}  ·  **Type:** `{kind}`"
                )

                # Tile/grid of media
                if kind == "image":
                    cols = st.columns(min(max(len(urls), 1), 4))
                    for i, url in enumerate(urls):
                        with cols[i % len(cols)]:
                            st.image(url, width=180)
                            if st.button(
                                "🔍 View", key=f"zoom_img_{model_id}_{idx}_{i}"
                            ):
                                st.session_state["zoom_media_url"] = url
                                st.session_state["zoom_media_meta"] = meta
                                st.session_state["zoom_media_kind"] = "image"
