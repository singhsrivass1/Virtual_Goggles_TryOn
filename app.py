# app.py
# Streamlit + MediaPipe FaceMesh goggles try-on
# Works for both webcam and uploaded image modes

import streamlit as st
import cv2, os, time
import numpy as np
from PIL import Image
import mediapipe as mp
from models.bisenet_model import BiSeNet

# Optional segmentation imports (PyTorch)
USE_SEG = False
try:
    import torch
    import torchvision.transforms as T
    USE_TORCH = True
except Exception:
    USE_TORCH = False

st.set_page_config(page_title="Virtual Goggles Try-On", layout="wide")
st.title("🕶️ Virtual Goggles Try-On (Streamlit + MediaPipe)")

# ----------------- FaceMesh setup -----------------
mp_face = mp.solutions.face_mesh
# Streaming FaceMesh (for webcam)
face_mesh_stream = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                    min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ----------------- Load optional segmentation model -----------------
SEG_MODEL_PATH = "models/bisenet.pth"
seg_model = None

if os.path.exists(SEG_MODEL_PATH) and USE_TORCH:
    try:
        seg_model = BiSeNet(n_classes=19)
        state = torch.load(SEG_MODEL_PATH, map_location='cpu')
        result = seg_model.load_state_dict(state, strict=False)
        try:
            missing = result.missing_keys
            unexpected = result.unexpected_keys
        except Exception:
            missing, unexpected = result
        seg_model.eval()
        USE_SEG = True
        st.success(f"Segmentation weights loaded ✅ (missing={len(missing)}, unexpected={len(unexpected)})")
    except Exception as e:
        st.warning(f"Could not fully load BiSeNet model: {e}\nFalling back to landmark occlusion.")
else:
    st.info("No segmentation weights found. Using landmark-based occlusion fallback.")

# ----------------- UI setup -----------------
col1, col2 = st.columns([2, 1])
with col1:
    src = st.radio("Input source", ("Webcam (Real-time)", "Upload Image (single)"))
    uploaded_goggles = st.file_uploader("Upload your goggles PNG (transparent)", type=["png"])
    if uploaded_goggles:
        goggles_pil = Image.open(uploaded_goggles).convert("RGBA")
    else:
        default_path = "assets/goggles1.png"
        goggles_pil = Image.open(default_path).convert("RGBA") if os.path.exists(default_path) else None

    st.write("Fine-tune:")
    scale_mult = st.slider("Scale multiplier", 0.6, 2.5, 1.0, 0.01)
    y_offset = st.slider("Vertical offset (px)", -120, 120, 0, 1)
    show_landmarks = st.checkbox("Show detected landmarks", False)

with col2:
    snapshot_btn = st.button("Save Snapshot")
    st.text(f"MediaPipe FaceMesh ready")
    st.text(f"Segmentation enabled: {USE_SEG}")

FRAME_WINDOW = st.image([])

# ----------------- Helper functions -----------------
def cv2_to_pil(img_cv2):
    return Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))

def pil_to_cv2(img_pil):
    return cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)

def compute_eye_centers(pts):
    def avg(indices):
        arr = np.array([pts[i] for i in indices if i < len(pts)], dtype=np.float32)
        return arr.mean(axis=0)
    try:
        left_center = avg([468, 469, 470, 471])
        right_center = avg([473, 474, 475, 476])
    except Exception:
        left_center = avg([33, 133, 160, 159])
        right_center = avg([362, 263, 387, 386])
    return np.array(left_center), np.array(right_center)

def affine_overlay_goggles(frame_bgr, goggles_pil, pts):
    if goggles_pil is None or not pts:
        return frame_bgr

    left_c, right_c = compute_eye_centers(pts)
    center = ((left_c + right_c) / 2).astype(int)
    dx, dy = right_c - left_c
    angle = np.degrees(np.arctan2(dy, dx))
    eye_dist = np.linalg.norm(right_c - left_c)
    target_w = max(1, eye_dist * 2.6 * scale_mult)
    gw, gh = goggles_pil.size
    target_h = int((target_w / gw) * gh)
    g_resized = goggles_pil.resize((int(target_w), int(target_h)), Image.LANCZOS)
    g_rot = g_resized.rotate(angle, expand=True)
    gw2, gh2 = g_rot.size

    x = int(center[0] - gw2 / 2)
    y = int(center[1] - gh2 / 2) + int(y_offset)

    frame_pil = cv2_to_pil(frame_bgr).convert("RGBA")
    overlay = Image.new("RGBA", frame_pil.size, (0, 0, 0, 0))
    overlay.paste(g_rot, (x, y), g_rot)

    frame_np = np.array(frame_pil)
    overlay_np = np.array(overlay)
    alpha = overlay_np[..., 3] / 255.0
    for c in range(3):
        frame_np[..., c] = (1 - alpha) * frame_np[..., c] + alpha * overlay_np[..., c]

    return cv2.cvtColor(frame_np.astype(np.uint8), cv2.COLOR_RGBA2BGR)

# ----------------- Core frame processing -----------------
def get_landmarks(frame_bgr, static=False):
    mesh = mp_face.FaceMesh(static_image_mode=static, max_num_faces=1, refine_landmarks=True)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = mesh.process(rgb)
    mesh.close()
    if not res.multi_face_landmarks:
        return None
    h, w = frame_bgr.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in res.multi_face_landmarks[0].landmark]
    return pts

# ----------------- Main -----------------
if src == "Webcam (Real-time)":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Cannot access webcam.")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh_stream.process(rgb)
            if res.multi_face_landmarks:
                h, w = frame.shape[:2]
                pts = [(int(p.x * w), int(p.y * h)) for p in res.multi_face_landmarks[0].landmark]
                out = affine_overlay_goggles(frame, goggles_pil, pts)
                if show_landmarks:
                    for (x, y) in pts:
                        cv2.circle(out, (x, y), 1, (0, 255, 0), -1)
            else:
                out = frame
            FRAME_WINDOW.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

            if snapshot_btn:
                fname = f"snapshot_{int(time.time())}.png"
                cv2.imwrite(fname, out)
                st.success(f"Saved {fname}")
                snapshot_btn = False
        cap.release()

else:
    uploaded = st.file_uploader("Upload a face image to try on goggles", type=["png", "jpg", "jpeg"])
    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        pts = get_landmarks(img, static=True)
        if pts is None:
            st.warning("⚠️ No face detected in uploaded image. Try a clearer photo.")
            FRAME_WINDOW.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            out = affine_overlay_goggles(img, goggles_pil, pts)
            if show_landmarks:
                for (x, y) in pts:
                    cv2.circle(out, (x, y), 1, (0, 255, 0), -1)
            FRAME_WINDOW.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            if snapshot_btn:
                fname = f"snapshot_{int(time.time())}.png"
                cv2.imwrite(fname, out)
                st.success(f"Saved {fname}")
