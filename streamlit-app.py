import io
import os
import zipfile
from datetime import datetime
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
import streamlit as st

st.set_page_config(page_title="Row Screenshot Exporter", page_icon="🧩", layout="wide")
st.title("🧩 Row Screenshot Exporter")
st.caption("Erstellt Crops pro **Reihe** (40px Rand) und benennt sie z. B. `Untertest - Set X - Aufgabe 1 - Lösung.png`.")

# ===== Helpers =====
def load_cv2_image(file) -> np.ndarray:
    """Read uploaded file (BytesIO) as OpenCV BGR image."""
    bytes_data = file.read()
    img_arr = np.frombuffer(bytes_data, dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    return img

def detect_row_anchors(img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect the left-number anchor boxes used to crop each row."""
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 120)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    anchors: List[Tuple[int, int, int, int]] = []
    min_area = (h * w) * 0.01 * 0.05  # small but ignore noise
    max_area = (h * w) * 0.2  # avoid page-sized regions

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        x, y, bw, bh = cv2.boundingRect(approx)
        area = bw * bh
        if area < min_area or area > max_area:
            continue
        aspect = bw / float(bh)
        if not (0.7 <= aspect <= 1.3):
            continue
        if bw < w * 0.05 or bh < h * 0.05:
            continue
        if x > w * 0.35:
            continue
        anchors.append((x, y, bw, bh))

    anchors.sort(key=lambda b: b[1])

    filtered: List[Tuple[int, int, int, int]] = []
    for box in anchors:
        if not filtered:
            filtered.append(box)
            continue
        x, y, bw, bh = box
        fx, fy, fw, fh = filtered[-1]
        if abs(y - fy) < min(bh, fh) * 0.5:
            if bw * bh > fw * fh:
                filtered[-1] = box
        else:
            filtered.append(box)

    return filtered

def crop_row_region(
    img: np.ndarray,
    box: Tuple[int, int, int, int],
    margin_vertical: int = 40,
    margin_horizontal: int = 40,
) -> np.ndarray:
    """Crop the full task row with a FIXED HEIGHT based solely on the anchor box.
    Height = anchor height (+/- vertical margin). This guarantees the crop never
    cuts content vertically and matches the grey-number box height per row.
    """
    h, w = img.shape[:2]
    x, y, bw, bh = box

    # --- Refine vertical bounds by detecting the OUTER edges of the grey box ---
    # Build a vertical stripe around the anchor to search for the horizontal edges
    xs = max(0, x - 10)
    xe = min(w, x + bw + 10)
    ys = max(0, y - int(1.2 * bh))
    ye = min(h, y + int(1.2 * bh))

    stripe = img[ys:ye, xs:xe]
    stripe_gray = cv2.cvtColor(stripe, cv2.COLOR_BGR2GRAY)
    stripe_blur = cv2.GaussianBlur(stripe_gray, (5, 5), 0)
    stripe_edges = cv2.Canny(stripe_blur, 40, 120)

    # Row-wise edge strength to find strong horizontal lines (grey box borders)
    row_signal = stripe_edges.sum(axis=1).astype(np.float32)
    # Smooth the signal to make peaks clearer
    row_signal = cv2.GaussianBlur(row_signal.reshape(-1, 1), (1, 9), 0).ravel()

    n = len(row_signal)
    # Search windows: first/last 40% of the stripe height
    upper_win = row_signal[: max(5, int(0.4 * n))]
    lower_win = row_signal[min(n - 5, int(0.6 * n)) :]

    if upper_win.size > 0 and lower_win.size > 0:
        top_idx_local = int(np.argmax(upper_win))
        bot_idx_local = int(np.argmax(lower_win)) + int(0.6 * n)
        y_top_edge = ys + top_idx_local
        y_bot_edge = ys + bot_idx_local

        # Sanity: ensure edges are ordered and have reasonable height; otherwise fallback
        if y_bot_edge - y_top_edge >= int(0.6 * bh):
            top = max(0, y_top_edge - margin_vertical)
            bottom = min(h, y_bot_edge + margin_vertical)
        else:
            # Fallback to anchor-based vertical bounds
            top = max(0, y - margin_vertical)
            bottom = min(h, y + bh + margin_vertical)
    else:
        # Fallback if we couldn't compute windows
        top = max(0, y - margin_vertical)
        bottom = min(h, y + bh + margin_vertical)

    # --- Determine horizontal bounds within this fixed-height band ---
    band = img[top:bottom, :]
    band_gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    _, band_thresh = cv2.threshold(band_gray, 240, 255, cv2.THRESH_BINARY_INV)
    col_signal = band_thresh.sum(axis=0)
    col_threshold = 255 * (bottom - top) * 0.004
    content_cols = np.where(col_signal > col_threshold)[0]

    if content_cols.size > 0:
        content_left = int(content_cols[0])
        content_right = int(content_cols[-1])
        left = max(0, content_left - margin_horizontal)
        right = min(w, content_right + margin_horizontal)
    else:
        left = max(0, int(x - margin_horizontal))
        right = min(w, int(x + bw + margin_horizontal))

    if left >= right:
        left = max(0, int(x - margin_horizontal))
        right = min(w, int(x + bw + margin_horizontal))

    return img[top:bottom, left:right].copy()

def export_rows_from_image(img_bgr: np.ndarray, untertest: str, set_text: str, max_rows: int | None = None):
    anchors = detect_row_anchors(img_bgr)
    if max_rows is not None:
        anchors = anchors[:max_rows]

    crops = []
    for box in anchors:
        crop = crop_row_region(img_bgr, box, margin_vertical=40, margin_horizontal=40)
        crops.append(crop)
    return crops

# ===== Sidebar Inputs (Freitext) =====
with st.sidebar:
    st.header("Einstellungen")
    untertest = st.text_input("Untertest (Freitext)", value="Untertest A")
    set_text = st.text_input("Set (Freitext)", value="1")
    max_rows = st.text_input("Anzahl Aufgaben (optional, leer = alle)", value="")
    max_rows_int = None
    if max_rows.strip():
        try:
            max_rows_int = int(max_rows.strip())
        except ValueError:
            st.warning("⚠️ 'Anzahl Aufgaben' muss eine Zahl sein. Ignoriere Eingabe.")
            max_rows_int = None

    uploaded_files = st.file_uploader("Seiten als Bilder hochladen", type=["png","jpg","jpeg","webp"], accept_multiple_files=True)

    start = st.button("Start ▶️", type="primary", use_container_width=True)

# ===== Main Area =====
if start:
    if not uploaded_files:
        st.error("Bitte mindestens ein Bild hochladen.")
        st.stop()

    out_imgs = []
    task_counter = 1
    for uf in uploaded_files:
        img = load_cv2_image(uf)
        crops = export_rows_from_image(img, untertest, set_text, max_rows_int)
        for crop in crops:
            filename = f"{untertest} - Set {set_text} - Aufgabe {task_counter} - Lösung.png"
            out_imgs.append((filename, crop))
            task_counter += 1

    if not out_imgs:
        st.warning("Keine Reihen erkannt. Prüfe das Layout und versuche es erneut.")
    else:
        # Show previews
        st.subheader("Vorschau")
        for name, crop in out_imgs:
            st.markdown(f"**{name}**")
            st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Create ZIP in-memory
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, crop in out_imgs:
                # Encode PNG bytes
                ok, buf = cv2.imencode(".png", crop)
                if ok:
                    zf.writestr(name, buf.tobytes())
        zip_buf.seek(0)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_name = f"Loesungen_{untertest}_Set_{set_text}_{stamp}.zip"
        st.download_button("ZIP herunterladen ⬇️", data=zip_buf, file_name=zip_name, mime="application/zip")

else:
    st.info("Links die Felder ausfüllen, Bilder hochladen und **Start** drücken.")
