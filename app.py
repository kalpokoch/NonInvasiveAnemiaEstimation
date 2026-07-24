import os

import streamlit as st
import torch
from PIL import Image, ImageOps

from gradcam import compute_gradcam_overlays
from inference import build_batch, load_encoding_map, load_model_and_checkpoint, predict
from model import IMAGE_MODALITY_KEYS
from segmentation import detect_box, draw_box, load_yolo_models, pad_box, crop_to_box

APP_DIR = os.path.dirname(os.path.abspath(__file__))

FIXED_CHECKPOINT_PATH = os.path.join(APP_DIR, "model_checkpoint", "hb_model_sensorfusion_v1_512_best.pth")
USE_TTA = False

MODALITY_LABELS = {
    "eyelid": "Lower eyelid",
    "tongue": "Tongue (underside/back)",
    "palm": "Palm",
    "fingertips": "Fingertips",
}

PATIENTS_DIR = os.path.join(APP_DIR, "Patients")
PATIENT_MODALITY_STEMS = {
    "eyelid": "Lowereyelid",
    "tongue": "Tongueback",
    "palm": "Palm",
    "fingertips": "Fingertip",
}
PATIENT_IMAGE_EXTENSIONS = (".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG")

st.set_page_config(page_title="Hb / Anemia Predictor", page_icon="🩸", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 4rem;
        max-width: 1280px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px;
    }
    div[data-testid="stFileUploaderDropzone"] {
        border-radius: 10px;
    }
    .stButton > button {
        border-radius: 8px;
        padding: 0.55rem 1rem;
    }
    h2, h3 {
        margin-top: 1.75rem;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def list_patients():
    if not os.path.isdir(PATIENTS_DIR):
        return []
    return sorted(d for d in os.listdir(PATIENTS_DIR) if os.path.isdir(os.path.join(PATIENTS_DIR, d)))


def find_patient_modality_path(patient_name, modality_key):
    stem = PATIENT_MODALITY_STEMS[modality_key]
    patient_dir = os.path.join(PATIENTS_DIR, patient_name)
    for ext in PATIENT_IMAGE_EXTENSIONS:
        path = os.path.join(patient_dir, stem + ext)
        if os.path.isfile(path):
            return path
    return None


def open_image_corrected(source):
    """Open an image (path or file-like) and apply its EXIF orientation tag, if any,
    so photos captured sideways/upside-down by a phone camera (pixels stored raw,
    rotation stored as metadata) display and process right-side-up. Most viewers do
    this automatically; PIL does not unless asked."""
    img = Image.open(source)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def make_thumbnail(img, size=(320, 320)):
    """Uniform, center-cropped square thumbnail (like CSS object-fit: cover) so
    preview cards line up regardless of each source photo's own aspect ratio."""
    return ImageOps.fit(img.convert("RGB"), size, method=Image.LANCZOS)


def load_patient_thumbnail(path, size=(320, 320)):
    return make_thumbnail(open_image_corrected(path), size)


@st.cache_resource(show_spinner="Loading model checkpoint...")
def get_model(checkpoint_path, device_str):
    return load_model_and_checkpoint(checkpoint_path, device=device_str)


@st.cache_resource
def get_encoding_map():
    return load_encoding_map()


@st.cache_resource(show_spinner="Loading ROI detectors...")
def get_yolo_models(device_str):
    return load_yolo_models(device_str)


st.title("🩸 Hemoglobin / Anemia Predictor")
st.caption(
    "Multimodal Swin-T sensor-fusion model (eyelid / tongue / palm / fingertips photos + demographics) "
    "predicting hemoglobin (g/dL). Research prototype — not a medical device."
)

encoding_map = get_encoding_map()
if encoding_map.get("_comment"):
    st.warning(
        "⚠️ encoding_map.json in this app is a **placeholder** — the real race/complexion string-to-index "
        "mapping used at training time wasn't found in the project directory. Race and complexion "
        "predictions below use guessed category names but correct index counts (12 race / 3 complexion / "
        "2 gender). Replace streamlit_app/encoding_map.json with the real file if you have it."
    )

if not os.path.isfile(FIXED_CHECKPOINT_PATH):
    st.error(f"Checkpoint not found: {FIXED_CHECKPOINT_PATH}")
    st.stop()

checkpoint_path = FIXED_CHECKPOINT_PATH

with st.sidebar:
    st.header("Model settings")
    device_options = (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]
    device_str = st.selectbox("Device", device_options, index=0)

    use_tta = USE_TTA

    st.divider()
    show_gradcam = st.checkbox(
        "Show Grad-CAM", value=True,
        help="Highlights which regions of each uploaded photo most influenced that modality's own prediction.",
    )

model, ckpt = get_model(checkpoint_path, device_str)
yolo_models = get_yolo_models(device_str)

st.subheader("1. Upload photos")

patients = list_patients()
if patients:
    st.caption("Click a sample patient to load their photos, then untick any you don't want to use. Uploading your own photo for a modality below always overrides the sample.")
    p_cols = st.columns(len(patients) + 1, gap="small")
    for p_col, name in zip(p_cols, patients):
        if p_col.button(name, key=f"patient_btn_{name}", use_container_width=True):
            st.session_state["selected_patient"] = name
            st.session_state.pop("last_prediction", None)
    if p_cols[-1].button("Clear sample", key="clear_patient_btn", use_container_width=True):
        st.session_state["selected_patient"] = None
        st.session_state.pop("last_prediction", None)

selected_patient = st.session_state.get("selected_patient")
patient_use_flags = {}
if selected_patient:
    st.write("")
    st.markdown(f"**Loaded sample: {selected_patient}**")
    pat_cols = st.columns(4, gap="medium")
    for pat_col, key in zip(pat_cols, IMAGE_MODALITY_KEYS):
        with pat_col, st.container(border=True):
            st.markdown(f"**{MODALITY_LABELS[key]}**")
            path = find_patient_modality_path(selected_patient, key)
            if path is not None:
                st.image(load_patient_thumbnail(path), use_container_width=True)
                patient_use_flags[key] = st.checkbox(
                    "Use this photo", value=True, key=f"use_patient_{selected_patient}_{key}"
                )
            else:
                st.caption("Not available for this patient.")
    st.divider()

st.caption("Upload as many modalities as you have. Missing ones are handled via the model's fusion mechanism, but more photos generally give a more reliable prediction.")
st.write("")

cols = st.columns(4, gap="medium")
manual_images = {}
for col, key in zip(cols, IMAGE_MODALITY_KEYS):
    with col, st.container(border=True):
        st.markdown(f"**{MODALITY_LABELS[key]}**")
        file = st.file_uploader(f"upload_{key}", type=["jpg", "jpeg", "png"], key=f"upload_{key}", label_visibility="collapsed")
        if file is not None:
            img = open_image_corrected(file)
            st.image(img, use_container_width=True)
            manual_images[key] = img
        else:
            manual_images[key] = None

# Manual upload always overrides the loaded sample patient for that modality.
uploaded_images = {}
for key in IMAGE_MODALITY_KEYS:
    if manual_images[key] is not None:
        uploaded_images[key] = manual_images[key]
    elif selected_patient and patient_use_flags.get(key):
        path = find_patient_modality_path(selected_patient, key)
        uploaded_images[key] = open_image_corrected(path) if path else None
    else:
        uploaded_images[key] = None

any_uploaded = any(v is not None for v in uploaded_images.values())
if any_uploaded and yolo_models:
    with st.expander("🔍 Show ROI detector output (optional, for demonstration)"):
        st.caption(
            "For eyelid/tongue/palm, a YOLOv8 model auto-detects and crops to the relevant region before "
            "prediction (fingertips has no detector and always uses the fixed crop). This is purely "
            "illustrative and doesn't change the prediction itself."
        )
        detections = {}
        for key in IMAGE_MODALITY_KEYS:
            img = uploaded_images.get(key)
            yolo_model = yolo_models.get(key)
            if img is None or yolo_model is None:
                continue
            detections[key] = (img, detect_box(yolo_model, img))

        if detections:
            st.markdown("**Detected region**")
            row1_cols = st.columns(4, gap="medium")
            for col, key in zip(row1_cols, IMAGE_MODALITY_KEYS):
                if key not in detections:
                    continue
                img, detection = detections[key]
                with col:
                    st.caption(MODALITY_LABELS[key])
                    if detection is None:
                        st.image(make_thumbnail(img), use_container_width=True)
                        st.caption("No detection — raw photo used as-is.")
                    else:
                        x1, y1, x2, y2, conf = detection
                        boxed = draw_box(img, (x1, y1, x2, y2))
                        st.image(make_thumbnail(boxed), use_container_width=True)
                        st.caption(f"Confidence {conf:.2f}")

            st.markdown("**Cropped region actually fed to the model**")
            row2_cols = st.columns(4, gap="medium")
            for col, key in zip(row2_cols, IMAGE_MODALITY_KEYS):
                if key not in detections:
                    continue
                img, detection = detections[key]
                with col:
                    st.caption(MODALITY_LABELS[key])
                    if detection is None:
                        st.caption("N/A (raw photo used)")
                    else:
                        x1, y1, x2, y2, conf = detection
                        padded_box = pad_box((x1, y1, x2, y2), img.size)
                        cropped = crop_to_box(img, padded_box)
                        st.image(make_thumbnail(cropped), use_container_width=True)

st.divider()
st.subheader("2. Demographics")
d1, d2, d3, d4, d5, d6 = st.columns(6, gap="medium")
age = d1.number_input("Age (years)", min_value=0, max_value=120, value=30)
weight_kg = d2.number_input("Weight (kg)", min_value=1.0, max_value=250.0, value=60.0)
height_cm = d3.number_input("Height (cm)", min_value=30.0, max_value=230.0, value=160.0)

race_options = list(encoding_map["race"]["label_encoding"].keys())
complexion_options = list(encoding_map["complexion"]["label_encoding"].keys())
gender_options = list(encoding_map["gender"]["label_encoding"].keys())

race_label = d4.selectbox("Race", race_options)
complexion_label = d5.selectbox("Complexion", complexion_options)
gender_label = d6.selectbox("Gender", gender_options)

n_uploaded = sum(1 for v in uploaded_images.values() if v is not None)

st.divider()
st.subheader("3. Predict")
predict_clicked = st.button("Run prediction", type="primary", disabled=(n_uploaded == 0))
if n_uploaded == 0:
    st.info("Upload at least one photo to enable prediction.")

if predict_clicked:
    demographics = {
        "age": float(age),
        "weight_kg": float(weight_kg),
        "height_cm": float(height_cm),
        "race_idx": encoding_map["race"]["label_encoding"][race_label],
        "complexion_idx": encoding_map["complexion"]["label_encoding"][complexion_label],
        "gender_idx": encoding_map["gender"]["label_encoding"][gender_label],
    }

    with st.spinner("Running inference..."):
        batch, presence, display_images = build_batch(
            uploaded_images, demographics, ckpt, device=device_str, yolo_models=yolo_models
        )
        result = predict(model, batch, presence, ckpt, use_tta=use_tta)

    overlays = {}
    if show_gradcam:
        with st.spinner("Computing Grad-CAM..."):
            overlays = compute_gradcam_overlays(model, batch, presence, display_images, IMAGE_MODALITY_KEYS)

    st.session_state["last_prediction"] = {"result": result, "gradcam_overlays": overlays}

if "last_prediction" in st.session_state:
    result = st.session_state["last_prediction"]["result"]
    overlays = st.session_state["last_prediction"]["gradcam_overlays"]

    hb = result["hb_pred"]
    hb_std = result["hb_pred_std"]

    st.divider()
    st.markdown("### Result")
    m1, m2 = st.columns(2, gap="medium")
    m1.metric("Predicted Hemoglobin", f"{hb:.2f} g/dL", help="Fused via inverse-variance weighting across present modalities.")
    m2.metric("Approx. 95% interval", f"{hb - 1.96 * hb_std:.2f} – {hb + 1.96 * hb_std:.2f} g/dL")

    # WHO-ish anemia thresholds are gender/age dependent; show a rough flag only.
    if hb < 11.0:
        st.error("Predicted Hb is below 11 g/dL — commonly used as a rough anemia threshold. This is a research model, not a diagnosis.")
    elif hb < 12.5:
        st.warning("Predicted Hb is in a borderline-low range.")
    else:
        st.success("Predicted Hb is in a typical non-anemic range.")

    with st.expander("Per-modality breakdown"):
        rows = []
        for key, info in result["per_modality"].items():
            rows.append({
                "modality": key,
                "present": info["present"],
                "hb_pred (g/dL)": round(info["hb_pred"], 2) if info["present"] else "—",
                "hb_std (g/dL)": round(info["hb_std"], 2) if info["present"] else "—",
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption(
            "Lower std = the model is more confident in that modality's own estimate; the fused prediction "
            "weights each present modality by its inverse variance. Modalities marked not present had no photo "
            "uploaded, so their estimate is excluded from fusion (shown as \"—\" here since it's computed from a "
            "placeholder blank image, not the photo you uploaded elsewhere)."
        )

    if show_gradcam and overlays:
        st.markdown("### Grad-CAM: where each photo's prediction came from")
        st.caption(
            "Warm colors (red/yellow) mark regions of the shared Swin-T backbone's last feature map that most "
            "increased that modality's own hemoglobin estimate. Computed per-modality (no TTA), not per fused result."
        )
        gc_cols = st.columns(4, gap="medium")
        for gc_col, key in zip(gc_cols, IMAGE_MODALITY_KEYS):
            with gc_col:
                if key in overlays:
                    st.markdown(f"**{MODALITY_LABELS[key]}**")
                    st.image(overlays[key], use_container_width=True)

    st.caption("This tool is a research prototype trained on a specific dataset. It is not a validated diagnostic device and should not be used for medical decisions.")
