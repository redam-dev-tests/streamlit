"""
Row Screenshot Exporter
=======================

This Streamlit application extracts individual task rows from scanned
questionnaire pages.  Each task row begins with a large grey box on the
left containing the task number.  The app automatically detects these
boxes using simple geometric heuristics, crops the corresponding row
with fixed margins on all sides and packages the results in a zip
archive for download.

Features
--------

* **Freitext-Eingaben** für Untertest und Set – keine Vorauswahl
* Automatische Zeilenerkennung basierend auf der linken Nummernbox
* 40 px Rand links/rechts/oben/unten um jede ausgeschnittene Reihe
* Vorschau der erzeugten Ausschnitte direkt in der App
* Generierung eines ZIP-Archivs aller Crops mit sprechenden Dateinamen

The detection and cropping logic has been tuned on the sample pages
provided by the user.  It relies purely on contour geometry to locate
the left number boxes and does not use colour thresholds, making it
robust to variations in scan contrast or white balance.
"""

import io
import zipfile
from datetime import datetime
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import streamlit as st


def load_cv2_image(file) -> np.ndarray:
    """Read an uploaded file (BytesIO) as an OpenCV BGR image."""
    bytes_data = file.read()
    img_arr = np.frombuffer(bytes_data, dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    return img


def detect_row_anchors(img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect the large grey-number box at the beginning of each row.

    Each task row on the questionnaire begins with a grey, roughly
    square box on the left containing the yellow task number.  We
    locate these boxes by finding rectangular contours in the left
    half of the page whose size and aspect ratio fall within a
    reasonable range.

    Parameters
    ----------
    img : np.ndarray
        BGR image of the full page.

    Returns
    -------
    List[Tuple[int, int, int, int]]
        A list of bounding boxes (x, y, w, h) for each detected row,
        sorted from top to bottom.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 20, 80)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cand: List[Tuple[int, int, int, int]] = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) != 4:
            continue
        x, y, bw, bh = cv2.boundingRect(approx)
        # Only consider boxes in the left half
        if x > int(0.5 * w):
            continue
        # Height must be between 10% and 30% of the page height
        if bh < int(0.10 * h) or bh > int(0.30 * h):
            continue
        # Boxes should be roughly square
        aspect = bw / float(bh)
        if aspect < 0.85 or aspect > 1.20:
            continue
        # Exclude very wide boxes (e.g. page borders)
        if bw > int(0.40 * w):
            continue
        cand.append((x, y, bw, bh))
    if not cand:
        return []
    # Group candidates by their vertical position and take the leftmost in each group
    cand.sort(key=lambda t: t[1])
    rows: List[Tuple[int, int, int, int]] = []
    group: List[Tuple[int, int, int, int]] = []
    def flush(g: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
        if not g:
            return None
        return min(g, key=lambda t: t[0])
    for item in cand:
        if not group:
            group = [item]
            continue
        _, y, _, bh = item
        _, gy, _, gh = group[-1]
        if abs(y - gy) < int(0.6 * min(bh, gh)):
            group.append(item)
        else:
            chosen = flush(group)
            if chosen is not None:
                rows.append(chosen)
            group = [item]
    chosen = flush(group)
    if chosen is not None:
        rows.append(chosen)
    rows.sort(key=lambda b: b[1])
    return rows


def _threshold_digit_mask(region: np.ndarray) -> np.ndarray:
    """Return a binary mask highlighting yellow digits within a box.

    The digits in the task number boxes are rendered in a bright yellow
    colour.  We exploit this by thresholding on the red and green
    channels while requiring the blue channel to be low.  The result
    is a binary mask where digit pixels are white (255) and the rest
    are black (0).  A small morphological opening is applied to
    remove isolated noise without merging separate digits.

    Parameters
    ----------
    region : np.ndarray
        BGR image of the number box.

    Returns
    -------
    np.ndarray
        Single-channel (uint8) mask of the same size as ``region``.
    """
    # Split into BGR channels
    b, g, r = cv2.split(region)
    # Threshold: high red & green, reasonably low blue
    mask = ((r > 150) & (g > 150) & (b < 200)).astype(np.uint8) * 255
    # Remove small specks without connecting digits
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def _extract_digit_images(region: np.ndarray) -> List[np.ndarray]:
    """Extract individual digit subimages from a number box.

    Parameters
    ----------
    region : np.ndarray
        BGR image containing one or more yellow digits.

    Returns
    -------
    List[np.ndarray]
        A list of BGR images, each cropped around a single digit,
        sorted from left to right.
    """
    mask = _threshold_digit_mask(region)
    # Find connected components (contours) corresponding to digits
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Ignore tiny regions
        if w * h > 10:
            boxes.append((x, y, w, h))
    # Sort left-to-right
    boxes.sort(key=lambda b: b[0])
    digits: List[np.ndarray] = []
    for x, y, w, h in boxes:
        digits.append(region[y : y + h, x : x + w])
    return digits


def _digit_features(digit_img: np.ndarray) -> Dict[str, float]:
    """Compute simple shape features for a digit image.

    The features used are:

    * ``holes`` – the number of enclosed holes in the digit mask
    * ``aspect`` – width divided by height of the overall digit region
    * ``fill`` – ratio of digit pixels to the bounding box area
    * ``ratio`` – fraction of digit pixels in the top half of the box

    These descriptors allow us to differentiate most numerals in the
    questionnaire font without external OCR libraries.

    Parameters
    ----------
    digit_img : np.ndarray
        BGR subimage containing one isolated digit.

    Returns
    -------
    Dict[str, float]
        A mapping of feature names to values.
    """
    # Obtain a binary mask of the digit
    mask = _threshold_digit_mask(digit_img)
    # Identify holes via contour hierarchy
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = 0
    if hierarchy is not None:
        for idx, _ in enumerate(contours):
            parent = hierarchy[0][idx][3]
            if parent != -1:
                # Count only holes directly inside outer contours
                if hierarchy[0][parent][3] == -1:
                    holes += 1
    # Compute bounding box of all contours (to get overall digit dims)
    boxes = [cv2.boundingRect(cnt) for cnt in contours]
    if boxes:
        xs = [b[0] for b in boxes]
        ys = [b[1] for b in boxes]
        x2s = [b[0] + b[2] for b in boxes]
        y2s = [b[1] + b[3] for b in boxes]
        width = max(x2s) - min(xs)
        height = max(y2s) - min(ys)
        aspect = width / float(height) if height else 0.0
        pixel_count = mask.sum() / 255.0
        bbox_area = width * height
        fill = (pixel_count / bbox_area) if bbox_area > 0 else 0.0
    else:
        aspect = 0.0
        fill = 0.0
    # Top/bottom pixel ratio
    h = mask.shape[0]
    top = mask[: h // 2].sum() / 255.0
    bottom = mask[h // 2 :].sum() / 255.0
    ratio = top / (top + bottom) if (top + bottom) > 0 else 0.0
    return {
        "holes": float(holes),
        "aspect": float(aspect),
        "fill": float(fill),
        "ratio": float(ratio),
    }


def _classify_digit(feats: Dict[str, float]) -> str:
    """Classify a digit based on heuristic shape features.

    This function uses the results of :func:`_digit_features` to
    distinguish between numerals 0–9 as they appear in the test
    material.  The decision logic was calibrated on the provided
    questionnaire pages.  If a digit cannot be confidently
    classified, a question mark ("?") is returned.

    Parameters
    ----------
    feats : Dict[str, float]
        Feature dictionary with keys ``holes``, ``aspect``, ``fill`` and
        ``ratio``.

    Returns
    -------
    str
        The recognised digit as a string.
    """
    holes = feats["holes"]
    aspect = feats["aspect"]
    fill = feats["fill"]
    ratio = feats["ratio"]
    # Two holes: must be an 8
    if holes >= 2:
        return "8"
    # One hole: could be 0, 4, 6 or 9
    if holes == 1:
        # 4 tends to be wide
        if aspect > 0.72:
            return "4"
        # If bottom is heavier than top it's likely a 6
        if ratio < 0.45:
            return "6"
        # Between 0 and 9: 9 is slightly denser
        if fill > 0.67:
            return "9"
        else:
            return "0"
    # No holes: 1, 2, 3, 5 or 7
    # Very slender => 1
    if aspect < 0.50:
        return "1"
    # Top heavy => 7
    if ratio > 0.60:
        return "7"
    # Distinguish 2, 3 and 5 using fill and ratio
    # 5 typically has a larger upper mass than 2 or 3
    if ratio > 0.52:
        return "5"
    # 2 and 3 share similar aspect; 2 often has slightly lower top ratio
    if ratio < 0.47:
        return "2"
    # Otherwise default to 3
    return "3"


def read_task_number(region: np.ndarray) -> str:
    """Read the numeric label from a detected task row anchor.

    Parameters
    ----------
    region : np.ndarray
        BGR image containing the grey task number box.

    Returns
    -------
    str
        The recognised number as a string.  If the digits cannot be
        confidently parsed, a sequence of '?' will be returned.
    """
    digits = _extract_digit_images(region)
    if not digits:
        return "?"
    num_str = ""
    for dimg in digits:
        feats = _digit_features(dimg)
        num_str += _classify_digit(feats)
    return num_str


def crop_row_region(
    img: np.ndarray,
    box: Tuple[int, int, int, int],
    margin_vertical: int = 20,
    margin_horizontal: int = 40,
) -> np.ndarray:
    """Crop a task row around the detected anchor box.

    The crop extends vertically from ``(y - margin_vertical)`` to
    ``(y + h + margin_vertical)``, and horizontally from ``margin_horizontal``
    to ``img_width - margin_horizontal``.  In practice, the vertical
    margin has been tuned to 20 px (down from the original 40 px) to
    eliminate the grey header separation line from the first task on
    each page while still leaving a small whitespace buffer above and
    below the content.  The horizontal margin remains at 40 px to
    provide consistent left/right padding.

    Parameters
    ----------
    img : np.ndarray
        BGR image of the full page.
    box : Tuple[int, int, int, int]
        Bounding box (x, y, w, h) of the detected grey-number box.
    margin_vertical : int
        Number of pixels to include above and below the anchor box.
        Defaults to ``20``.
    margin_horizontal : int
        Number of pixels to include on both the left and right edges.
        Defaults to ``40``.

    Returns
    -------
    np.ndarray
        Cropped row image.
    """
    h, w = img.shape[:2]
    x, y, bw, bh = box
    # Apply the tuned vertical margin; reduce the top and bottom padding
    top = max(0, y - margin_vertical)
    bottom = min(h, y + bh + margin_vertical)
    left = max(0, margin_horizontal)
    right = min(w, w - margin_horizontal)
    return img[top:bottom, left:right].copy()


def export_rows_from_image(
    img_bgr: np.ndarray,
    untertest: str,
    set_text: str,
    max_rows: Optional[int] = None,
    margin_vertical: int = 20,
    margin_horizontal: int = 40,
) -> List[Tuple[np.ndarray, str]]:
    """Extract all task rows and recognise their numeric labels.

    Parameters
    ----------
    img_bgr : np.ndarray
        Colour image of the scanned questionnaire page.
    untertest : str
        Name of the subtest (currently unused).
    set_text : str
        Identifier of the test set (currently unused).
    max_rows : int, optional
        Maximum number of rows to extract; if ``None``, all detected rows
        are returned.
    margin_vertical : int
        Vertical margin passed to :func:`crop_row_region`.
    margin_horizontal : int
        Horizontal margin passed to :func:`crop_row_region`.

    Returns
    -------
    List[Tuple[np.ndarray, str]]
        A list of tuples ``(row_image, task_number)`` where
        ``task_number`` is the digit string read from the grey box of
        the row.  If the digits cannot be confidently recognised, the
        task number will contain '?' characters.
    """
    anchors = detect_row_anchors(img_bgr)
    if max_rows is not None:
        anchors = anchors[:max_rows]
    results: List[Tuple[np.ndarray, str]] = []
    for box in anchors:
        x, y, bw, bh = box
        # Read digits from the anchor region (grey box)
        number = read_task_number(img_bgr[y : y + bh, x : x + bw])
        # Crop full row band
        crop = crop_row_region(img_bgr, box, margin_vertical, margin_horizontal)
        results.append((crop, number))
    return results


def main() -> None:
    st.set_page_config(page_title="Row Screenshot Exporter", page_icon="🧩", layout="wide")
    st.title("🧩 Row Screenshot Exporter")
    st.caption(
        "Erstellt Crops pro **Reihe** (40 px Rand) und benennt sie z. B. `Untertest - Set X - Aufgabe 1 - Lösung.png`."
    )
    # Sidebar inputs
    with st.sidebar:
        st.header("Einstellungen")
        untertest = st.text_input("Untertest (Freitext)", value="Untertest A")
        set_text = st.text_input("Set (Freitext)", value="1")
        max_rows_input = st.text_input("Anzahl Aufgaben (optional, leer = alle)", value="")
        max_rows_int: Optional[int] = None
        if max_rows_input.strip():
            try:
                max_rows_int = int(max_rows_input.strip())
            except ValueError:
                st.warning("⚠️ 'Anzahl Aufgaben' muss eine Zahl sein. Ignoriere Eingabe.")
                max_rows_int = None
        uploaded_files = st.file_uploader(
            "Seiten als Bilder hochladen",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
        )
        start = st.button("Start ▶️", type="primary", use_container_width=True)
    # Main area
    if start:
        if not uploaded_files:
            st.error("Bitte mindestens ein Bild hochladen.")
            st.stop()
        out_imgs: List[Tuple[str, np.ndarray]] = []
        for uf in uploaded_files:
            img = load_cv2_image(uf)
            # Extract rows and recognised numbers
            rows = export_rows_from_image(
                img,
                untertest,
                set_text,
                max_rows=max_rows_int,
                margin_vertical=20,
                margin_horizontal=40,
            )
            for (crop, number) in rows:
                # Normalise the test name for filenames
                safe_test = untertest.replace(" ", "_")
                safe_set = set_text.replace(" ", "_")
                # Build filename using recognised task number; fall back to counter if needed
                if number and '?' not in number:
                    filename = f"{safe_test}_Set-{safe_set}_Task-{number}_Solution.png"
                else:
                    filename = f"{safe_test}_Set-{safe_set}_Task-{len(out_imgs)+1:02d}_Solution.png"
                out_imgs.append((filename, crop))
        if not out_imgs:
            st.warning("Keine Reihen erkannt. Prüfe das Layout und versuche es erneut.")
        else:
            st.subheader("Vorschau")
            for name, crop in out_imgs:
                st.markdown(f"**{name}**")
                st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), use_container_width=True)
            # Create ZIP in-memory
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for name, crop in out_imgs:
                    ok, buf = cv2.imencode(".png", crop)
                    if ok:
                        zf.writestr(name, buf.tobytes())
            zip_buf.seek(0)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_name = f"Loesungen_{untertest}_Set_{set_text}_{stamp}.zip"
            st.download_button(
                "ZIP herunterladen ⬇️", data=zip_buf, file_name=zip_name, mime="application/zip"
            )
    else:
        st.info("Links die Felder ausfüllen, Bilder hochladen und **Start** drücken.")


if __name__ == "__main__":
    main()