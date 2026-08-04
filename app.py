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

# Tongue is deliberately excluded from the app UI/flow (never uploaded, so its
# presence flag stays 0 and fusion excludes it automatically) even though the
# trained checkpoint still physically has a tongue head -- that's baked into
# the architecture and can't be removed without retraining.
STAGE1_KEYS = ["eyelid"]
STAGE2_KEYS = ["palm", "fingertips"]

AUX_IMAGE_DISPLAY_WIDTH = 320
LOW_HB_RECHECK_THRESHOLD = 10.0
LAB_TEST_SUGGESTION_THRESHOLD = 9.0

MODALITY_LABELS = {
    "eyelid": "Lower eyelid",
    "palm": "Palm",
    "fingertips": "Fingertips",
}

PATIENTS_DIR = os.path.join(APP_DIR, "Patients")
PATIENT_MODALITY_STEMS = {
    "eyelid": "Lowereyelid",
    "palm": "Palm",
    "fingertips": "Fingertip",
}
PATIENT_IMAGE_EXTENSIONS = (".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG")

SEO_TITLE = "HemFinder | AI-Based Hemoglobin Estimation"
SEO_DESCRIPTION = (
    "HemFinder is an AI-based research application for estimating hemoglobin levels "
    "using lower-eyelid images, palm images, and basic clinical information. "
    "Research proof-of-concept only, not a replacement for laboratory blood testing."
)
SEO_KEYWORDS = (
    "hemoglobin estimation, anemia detection, non-invasive hemoglobin, "
    "AI hemoglobin predictor, eyelid conjunctiva anemia, palm pallor anemia, "
    "hemoglobin AI, HemFinder"
)
GOOGLE_SITE_VERIFICATION = "-O14PT_UF-ewNYH-qNEfCoixx5IUR1A31QAx5YhTO7Q"


def _patch_index_html_head():
    """Streamlit serves its own bundled index.html directly, before any of our
    Python/JS runs, for the first HTTP GET of the page -- so client-side head
    injection is invisible to anything that fetches raw HTML without executing
    JS (search-engine verifiers, some crawlers, link-preview bots). Patch real
    <meta> tags straight into that static file on disk once per container so
    they're present from the very first byte served."""
    marker = "<!-- hemfinder-seo-tags -->"
    index_path = os.path.join(os.path.dirname(st.__file__), "static", "index.html")
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            html = f.read()
    except OSError:
        return
    if marker in html:
        return
    tags = "\n".join([
        marker,
        f'<meta name="google-site-verification" content="{GOOGLE_SITE_VERIFICATION}">',
        f'<meta name="description" content="{SEO_DESCRIPTION}">',
        f'<meta name="keywords" content="{SEO_KEYWORDS}">',
        '<meta name="robots" content="index, follow">',
        '<meta name="author" content="HemFinder">',
        '<meta property="og:type" content="website">',
        f'<meta property="og:title" content="{SEO_TITLE}">',
        f'<meta property="og:description" content="{SEO_DESCRIPTION}">',
        '<meta name="twitter:card" content="summary">',
        f'<meta name="twitter:title" content="{SEO_TITLE}">',
        f'<meta name="twitter:description" content="{SEO_DESCRIPTION}">',
    ])
    patched = html.replace("</head>", tags + "\n</head>", 1)
    try:
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(patched)
    except OSError:
        pass


_patch_index_html_head()

st.set_page_config(page_title=SEO_TITLE, page_icon="🩸", layout="wide")

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


def run_prediction(images, demographics, model, ckpt, device_str, yolo_models, use_tta, want_gradcam):
    batch, presence, display_images = build_batch(
        images, demographics, ckpt, device=device_str, yolo_models=yolo_models
    )
    result = predict(model, batch, presence, ckpt, use_tta=use_tta)
    overlays = {}
    if want_gradcam:
        overlays = compute_gradcam_overlays(model, batch, presence, display_images, IMAGE_MODALITY_KEYS)
    return result, overlays


def render_result_card(title, result, overlays, show_gradcam, modality_keys):
    hb = result["hb_pred"]
    hb_std = result["hb_pred_std"]

    st.markdown(f"#### {title}")
    m1, m2 = st.columns(2, gap="medium")
    m1.metric(
        "Predicted Hemoglobin", f"{hb:.2f} g/dL",
        help="Fused via inverse-variance weighting across present modalities.",
    )
    m2.metric("Approx. 95% interval", f"{hb - 1.96 * hb_std:.2f} – {hb + 1.96 * hb_std:.2f} g/dL")

    # WHO-ish anemia thresholds are gender/age dependent; show a rough flag only.
    if hb < LAB_TEST_SUGGESTION_THRESHOLD:
        st.error(
            f"**Abnormal result (below {LAB_TEST_SUGGESTION_THRESHOLD:.0f} g/dL).** "
            "**We recommend a laboratory blood test to confirm this reading.** "
            "This app is a research prototype and cannot substitute for a clinical diagnosis."
        )
    elif hb < 11.0:
        st.error("Predicted Hb is below 11 g/dL — commonly used as a rough anemia threshold. This is a research model, not a diagnosis.")
    elif hb < 12.5:
        st.warning("Predicted Hb is in a borderline-low range.")
    else:
        st.success("Predicted Hb is in a typical non-anemic range.")

    with st.expander("Per-modality breakdown"):
        rows = [
            {
                "modality": key,
                "present": info["present"],
                "hb_pred (g/dL)": round(info["hb_pred"], 2) if info["present"] else "—",
                "hb_std (g/dL)": round(info["hb_std"], 2) if info["present"] else "—",
            }
            for key, info in result["per_modality"].items()
            if key in modality_keys or key == "tabular"
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption(
            "Lower std = the model is more confident in that modality's own estimate; the fused prediction "
            "weights each present modality by its inverse variance."
        )

    present_overlay_keys = [k for k in modality_keys if k in overlays] if show_gradcam else []
    if present_overlay_keys:
        st.markdown("**Grad-CAM**")
        st.caption(
            "Warm colors (red/yellow) mark regions of the shared Swin-T backbone's last feature map that "
            "most increased that modality's own hemoglobin estimate."
        )
        gc_cols = st.columns(len(present_overlay_keys), gap="medium")
        for gc_col, key in zip(gc_cols, present_overlay_keys):
            with gc_col:
                st.markdown(f"**{MODALITY_LABELS[key]}**")
                # Fixed pixel width (not use_container_width) so a lone image
                # in a single-column layout doesn't blow up to the full page
                # width -- every Grad-CAM image renders at the same size
                # regardless of how many modalities are shown alongside it.
                st.image(overlays[key], width=AUX_IMAGE_DISPLAY_WIDTH)


def render_roi_debug_expander(images_subset, yolo_models):
    present = {k: v for k, v in images_subset.items() if v is not None}
    if not present or not yolo_models:
        return
    with st.expander("🔍 Show ROI detector output (optional, for demonstration)"):
        st.caption(
            "For eyelid/palm, a YOLOv8 model auto-detects and crops to the relevant region before "
            "prediction (fingertips has no detector and always uses the fixed crop). This is purely "
            "illustrative and doesn't change the prediction itself."
        )
        detections = {}
        for key, img in present.items():
            yolo_model = yolo_models.get(key)
            if yolo_model is None:
                continue
            detections[key] = (img, detect_box(yolo_model, img))
        if not detections:
            st.caption("No ROI detector available for the uploaded photo(s).")
            return

        st.markdown("**Detected region**")
        row1_cols = st.columns(len(detections), gap="medium")
        for col, (key, (img, detection)) in zip(row1_cols, detections.items()):
            with col:
                st.caption(MODALITY_LABELS[key])
                if detection is None:
                    st.image(make_thumbnail(img), width=AUX_IMAGE_DISPLAY_WIDTH)
                    st.caption("No detection — raw photo used as-is.")
                else:
                    x1, y1, x2, y2, conf = detection
                    boxed = draw_box(img, (x1, y1, x2, y2))
                    st.image(make_thumbnail(boxed), width=AUX_IMAGE_DISPLAY_WIDTH)
                    st.caption(f"Confidence {conf:.2f}")

        st.markdown("**Cropped region actually fed to the model**")
        row2_cols = st.columns(len(detections), gap="medium")
        for col, (key, (img, detection)) in zip(row2_cols, detections.items()):
            with col:
                st.caption(MODALITY_LABELS[key])
                if detection is None:
                    st.caption("N/A (raw photo used)")
                else:
                    x1, y1, x2, y2, conf = detection
                    padded_box = pad_box((x1, y1, x2, y2), img.size)
                    cropped = crop_to_box(img, padded_box)
                    st.image(make_thumbnail(cropped), width=AUX_IMAGE_DISPLAY_WIDTH)


@st.cache_resource(show_spinner="Loading model checkpoint...")
def get_model(checkpoint_path, device_str):
    return load_model_and_checkpoint(checkpoint_path, device=device_str)


@st.cache_resource
def get_encoding_map():
    return load_encoding_map()


@st.cache_resource(show_spinner="Loading ROI detectors...")
def get_yolo_models(device_str):
    return load_yolo_models(device_str)


st.header("HemFinder: AI-Based Non-Invasive Hemoglobin Estimation")

st.text(
    "HemFinder is an AI-based research application for estimating "
    "hemoglobin levels using lower-eyelid images, palm images, "
    "and basic clinical information."
)

st.caption(
    "Research proof-of-concept only. HemFinder is not a replacement "
    "for laboratory blood testing or professional medical diagnosis."
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
        "Show Grad-CAM", value=False,
        help=(
            "Highlights which regions of each uploaded photo most influenced that modality's own prediction. "
            "Off by default: computing it needs a backward pass that roughly doubles peak memory use, which "
            "matters on memory-constrained deployments."
        ),
    )

model, ckpt = get_model(checkpoint_path, device_str)
yolo_models = get_yolo_models(device_str)

st.subheader("1. Upload an eyelid photo")
st.caption(
    "Gently pull down the lower eyelid to expose the inner conjunctiva and photograph it in good "
    "light. This one photo is enough for an initial estimate — you can add more photos afterward "
    "if you want a more robust result."
)

patients = list_patients()
if patients:
    st.caption("Or click a sample patient to try the app with example photos.")
    p_cols = st.columns(len(patients) + 1, gap="small")
    for p_col, name in zip(p_cols, patients):
        if p_col.button(name, key=f"patient_btn_{name}", use_container_width=True):
            st.session_state["selected_patient"] = name
            st.session_state.pop("eyelid_result", None)
            st.session_state.pop("combined_result", None)
    if p_cols[-1].button("Clear sample", key="clear_patient_btn", use_container_width=True):
        st.session_state["selected_patient"] = None
        st.session_state.pop("eyelid_result", None)
        st.session_state.pop("combined_result", None)

selected_patient = st.session_state.get("selected_patient")
sample_eyelid_path = find_patient_modality_path(selected_patient, "eyelid") if selected_patient else None

eyelid_card_col = st.columns(4, gap="medium")[0]
with eyelid_card_col, st.container(border=True):
    st.markdown(f"**{MODALITY_LABELS['eyelid']}**")
    eyelid_file = st.file_uploader(
        "upload_eyelid", type=["jpg", "jpeg", "png"], key="upload_eyelid", label_visibility="collapsed"
    )
    manual_eyelid_img = open_image_corrected(eyelid_file) if eyelid_file is not None else None

    use_sample_eyelid = False
    if manual_eyelid_img is not None:
        st.image(manual_eyelid_img, use_container_width=True)
    elif sample_eyelid_path is not None:
        st.image(load_patient_thumbnail(sample_eyelid_path), use_container_width=True)
        use_sample_eyelid = st.checkbox(
            "Use this sample photo", value=True, key=f"use_sample_eyelid_{selected_patient}"
        )
    else:
        st.caption("No photo yet.")

if manual_eyelid_img is not None:
    eyelid_image = manual_eyelid_img
elif use_sample_eyelid and sample_eyelid_path is not None:
    eyelid_image = open_image_corrected(sample_eyelid_path)
else:
    eyelid_image = None

render_roi_debug_expander({"eyelid": eyelid_image}, yolo_models)

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

demographics = {
    "age": float(age),
    "weight_kg": float(weight_kg),
    "height_cm": float(height_cm),
    "race_idx": encoding_map["race"]["label_encoding"][race_label],
    "complexion_idx": encoding_map["complexion"]["label_encoding"][complexion_label],
    "gender_idx": encoding_map["gender"]["label_encoding"][gender_label],
}

st.divider()
st.subheader("3. Predict")
predict_clicked = st.button("Run prediction", type="primary", disabled=(eyelid_image is None))
if eyelid_image is None:
    st.info("Upload (or select a sample) eyelid photo above to enable prediction.")

if predict_clicked:
    with st.spinner("Running inference..."):
        result, overlays = run_prediction(
            {"eyelid": eyelid_image}, demographics, model, ckpt, device_str, yolo_models, use_tta, show_gradcam
        )
    st.session_state["eyelid_result"] = {"result": result, "gradcam_overlays": overlays}
    st.session_state.pop("combined_result", None)

if "eyelid_result" in st.session_state:
    eyelid_data = st.session_state["eyelid_result"]
    combined_data = st.session_state.get("combined_result")

    current_result = combined_data["result"] if combined_data else eyelid_data["result"]
    current_hb_low = current_result["hb_pred"] - 1.96 * current_result["hb_pred_std"]
    show_recheck_option = current_hb_low < LOW_HB_RECHECK_THRESHOLD

    if show_recheck_option:
        st.divider()
        st.subheader("4. Recheck recommended")
        st.caption(
            f"The lower end of the approx. 95% interval is below {LOW_HB_RECHECK_THRESHOLD:.0f} g/dL. Add "
            "palm and/or fingertip photos for a more robust, multi-site estimate before drawing conclusions. "
            "The eyelid photo from step 1 is reused automatically — no need to re-upload it."
        )

        addon_cols = st.columns(2, gap="medium")
        addon_images = {}
        for col, key in zip(addon_cols, STAGE2_KEYS):
            with col, st.container(border=True):
                st.markdown(f"**{MODALITY_LABELS[key]}**")
                file = st.file_uploader(
                    f"upload_{key}", type=["jpg", "jpeg", "png"], key=f"upload_{key}", label_visibility="collapsed"
                )
                if file is not None:
                    img = open_image_corrected(file)
                    st.image(img, use_container_width=True)
                    addon_images[key] = img
                    continue

                sample_path = find_patient_modality_path(selected_patient, key) if selected_patient else None
                if sample_path is not None:
                    st.image(load_patient_thumbnail(sample_path), use_container_width=True)
                    use_sample = st.checkbox(
                        "Use this sample photo", value=True, key=f"use_sample_{key}_{selected_patient}"
                    )
                    addon_images[key] = open_image_corrected(sample_path) if use_sample else None
                else:
                    st.caption("No photo yet.")
                    addon_images[key] = None

        render_roi_debug_expander(addon_images, yolo_models)

        n_addon = sum(1 for v in addon_images.values() if v is not None)
        combined_clicked = st.button(
            "Run combined prediction", type="primary", disabled=(n_addon == 0), key="combined_predict_btn"
        )
        if n_addon == 0:
            st.caption("Add at least one more photo above to enable a combined prediction.")

        if combined_clicked:
            images = {"eyelid": eyelid_image, **addon_images}
            with st.spinner("Running inference..."):
                result, overlays = run_prediction(
                    images, demographics, model, ckpt, device_str, yolo_models, use_tta, show_gradcam
                )
            used_keys = STAGE1_KEYS + [k for k in STAGE2_KEYS if addon_images.get(k) is not None]
            st.session_state["combined_result"] = {
                "result": result, "gradcam_overlays": overlays, "modality_keys": used_keys,
            }
            combined_data = st.session_state["combined_result"]

    st.divider()
    st.markdown("### Result")

    if combined_data:
        col_a, col_b = st.columns(2, gap="large")
        with col_a:
            render_result_card(
                "Eyelid only", eyelid_data["result"], eyelid_data["gradcam_overlays"], show_gradcam, STAGE1_KEYS
            )
        with col_b:
            combined_label = " + ".join(MODALITY_LABELS[k] for k in combined_data["modality_keys"])
            render_result_card(
                f"Combined ({combined_label})", combined_data["result"], combined_data["gradcam_overlays"],
                show_gradcam, combined_data["modality_keys"],
            )
    else:
        render_result_card(
            "Eyelid-only estimate", eyelid_data["result"], eyelid_data["gradcam_overlays"], show_gradcam, STAGE1_KEYS
        )

st.caption("This tool is a research prototype trained on a specific dataset. It is not a validated diagnostic device and should not be used for medical decisions.")
