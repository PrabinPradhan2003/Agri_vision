import os
import json
import sys
import time
import uuid
from io import BytesIO
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen, Request

from flask import Flask, redirect, render_template, request, url_for, jsonify, send_file
from PIL import Image
from torchvision import transforms

# Ensure repo root is importable (CNN.py depends on plant_labels.py at repo root)
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import CNN
from history_db import add_scan_return_id, get_scan, list_scans, set_feedback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from werkzeug.utils import secure_filename
from typing import Any, Dict, List, Optional, Tuple

try:
    from torchvision import models as tv_models
except Exception:
    tv_models = None


BASE_DIR = _THIS_DIR
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_DB_PATH = BASE_DIR / "history.db"

disease_info = pd.read_csv(BASE_DIR / 'disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv(BASE_DIR / 'supplement_info.csv', encoding='cp1252')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIDENCE_THRESHOLD = float(os.getenv("PREDICTION_CONFIDENCE_THRESHOLD", "0.50"))


INFERENCE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)


def _assess_image_quality(image: Image.Image) -> List[str]:
    """Return human-friendly warnings about the input image quality."""
    warnings: List[str] = []

    gray = image.convert("L")
    arr = np.asarray(gray, dtype=np.float32)

    brightness = float(arr.mean())
    if brightness < 45:
        warnings.append("Image looks too dark; try better lighting.")
    elif brightness > 210:
        warnings.append("Image looks too bright/overexposed; try softer lighting.")

    # Simple blur estimate (variance of gradients).
    gx = np.diff(arr, axis=1)
    gy = np.diff(arr, axis=0)
    sharpness = float(gx.var() + gy.var())
    if sharpness < 35:
        warnings.append("Image looks blurry; try holding the camera steady and focusing on the leaf.")

    return warnings


def _split_points(text: str, max_items: int = 8) -> List[str]:
    if not text:
        return []

    # Prefer newline-separated points; otherwise fall back to sentence-ish splits.
    raw_parts = [p.strip() for p in text.replace("\r", "").split("\n") if p.strip()]
    if len(raw_parts) <= 1:
        raw_parts = [p.strip() for p in text.replace("\r", "").split(".") if p.strip()]

    cleaned: List[str] = []
    for part in raw_parts:
        part = part.strip("-• \t")
        if not part:
            continue
        if part not in cleaned:
            cleaned.append(part)
        if len(cleaned) >= max_items:
            break
    return cleaned


def _df_col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    """Return the first matching column name (case-insensitive), else None."""
    cols = {c.strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in cols:
            return cols[key]
    return None


def _extract_symptoms_from_description(title: str, description: str, max_items: int = 6) -> List[str]:
    """Extract symptom-like sentences so Symptoms isn't just the full Description."""
    if not description:
        return []

    title_l = (title or "").lower()
    if "healthy" in title_l:
        return ["No visible disease symptoms detected (healthy)."]

    # Split into sentence-ish parts.
    parts = [p.strip() for p in description.replace("\r", "").replace("\n", " ").split(".")]
    parts = [p for p in parts if p]

    symptom_keywords = (
        "symptom",
        "appear",
        "lesion",
        "spot",
        "speck",
        "patch",
        "powder",
        "mildew",
        "rust",
        "blight",
        "rot",
        "mold",
        "chlorotic",
        "yellow",
        "brown",
        "black",
        "wilting",
        "curl",
        "distort",
        "defoli",
        "canker",
    )

    picked: List[str] = []
    for p in parts:
        pl = p.lower()
        if any(k in pl for k in symptom_keywords):
            picked.append(p)
        if len(picked) >= max_items:
            break

    if not picked:
        picked = parts[: max(1, min(3, max_items))]

    # Clean and de-duplicate
    cleaned: List[str] = []
    for p in picked:
        p = p.strip(" -\t")
        if p and p not in cleaned:
            cleaned.append(p)
        if len(cleaned) >= max_items:
            break
    return cleaned


def _parse_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if text == "":
            return None
        return float(text)
    except Exception:
        return None


def _fetch_realtime_weather(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """Fetch current weather using Open-Meteo (free, no API key).

    Returns a compact dict safe to pass to templates.
    """

    try:
        # Newer API format
        query = {
            "latitude": f"{lat:.6f}",
            "longitude": f"{lon:.6f}",
            "current": "temperature_2m,relative_humidity_2m,precipitation,rain,weather_code,wind_speed_10m",
            "timezone": "auto",
        }
        url = "https://api.open-meteo.com/v1/forecast?" + urlencode(query)
        req = Request(url, headers={"User-Agent": "Plant-Disease-Detection/1.0"})
        with urlopen(req, timeout=3) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        current = payload.get("current") or {}
        condition = _weather_code_to_text(current.get("weather_code"))
        return {
            "provider": "Open-Meteo",
            "lat": lat,
            "lon": lon,
            "time": current.get("time"),
            "temperature_c": current.get("temperature_2m"),
            "humidity_percent": current.get("relative_humidity_2m"),
            "precip_mm": current.get("precipitation"),
            "rain_mm": current.get("rain"),
            "wind_kph": current.get("wind_speed_10m"),
            "weather_code": current.get("weather_code"),
            "condition": condition,
        }
    except Exception:
        # Older API format fallback (current_weather=true)
        try:
            query = {
                "latitude": f"{lat:.6f}",
                "longitude": f"{lon:.6f}",
                "current_weather": "true",
                "timezone": "auto",
            }
            url = "https://api.open-meteo.com/v1/forecast?" + urlencode(query)
            req = Request(url, headers={"User-Agent": "Plant-Disease-Detection/1.0"})
            with urlopen(req, timeout=3) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            current = payload.get("current_weather") or {}
            condition = _weather_code_to_text(current.get("weathercode"))
            return {
                "provider": "Open-Meteo",
                "lat": lat,
                "lon": lon,
                "time": current.get("time"),
                "temperature_c": current.get("temperature"),
                "wind_kph": current.get("windspeed"),
                "wind_direction": current.get("winddirection"),
                "weather_code": current.get("weathercode"),
                "condition": condition,
            }
        except Exception:
            return None


def _weather_code_to_text(code: Any) -> Optional[str]:
    """Map Open-Meteo weather codes to a short condition string."""
    try:
        if code is None:
            return None
        c = int(code)
    except Exception:
        return None

    mapping = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        56: "Light freezing drizzle",
        57: "Dense freezing drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        66: "Light freezing rain",
        67: "Heavy freezing rain",
        71: "Slight snow",
        73: "Moderate snow",
        75: "Heavy snow",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail",
    }
    return mapping.get(c, "Unknown")


def _reverse_geocode_place(lat: float, lon: float) -> Optional[Dict[str, str]]:
    """Reverse geocode using Nominatim (OpenStreetMap) to get a city/region/country.

    Free but rate-limited. Called once per submission.
    """
    try:
        query = {
            "format": "jsonv2",
            "lat": f"{lat:.6f}",
            "lon": f"{lon:.6f}",
            "zoom": "10",
            "addressdetails": "1",
        }
        url = "https://nominatim.openstreetmap.org/reverse?" + urlencode(query)
        req = Request(
            url,
            headers={
                "User-Agent": "Plant-Disease-Detection (final-year-project)",
                "Accept": "application/json",
            },
        )
        with urlopen(req, timeout=3) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        addr = payload.get("address") or {}
        city = addr.get("city") or addr.get("town") or addr.get("village") or addr.get("hamlet")
        state = addr.get("state") or addr.get("region")
        country = addr.get("country")

        place_parts = [p for p in [city, state, country] if p]
        place = ", ".join(place_parts) if place_parts else None
        if not place:
            return None

        return {"place": place, "city": str(city) if city else "", "state": str(state) if state else "", "country": str(country) if country else ""}
    except Exception:
        return None


# Cache reverse-geocode results to avoid hitting Nominatim repeatedly.
_PLACE_CACHE: Dict[str, Tuple[float, Dict[str, str]]] = {}
_PLACE_CACHE_TTL_SECONDS = 6 * 60 * 60  # 6 hours


def _reverse_geocode_place_cached(lat: float, lon: float) -> Optional[Dict[str, str]]:
    key = f"{lat:.3f},{lon:.3f}"
    now = time.time()
    cached = _PLACE_CACHE.get(key)
    if cached is not None:
        ts, place = cached
        if now - ts <= _PLACE_CACHE_TTL_SECONDS:
            return place

    place = _reverse_geocode_place(lat, lon)
    if place is not None:
        _PLACE_CACHE[key] = (now, place)
    return place


def _general_prevention_tips() -> List[str]:
    return [
        "Crop rotation (avoid repeating the same crop in the same soil)",
        "Use disease-resistant varieties when possible",
        "Maintain proper spacing and airflow between plants",
        "Avoid overhead irrigation; water early so leaves dry faster",
        "Remove infected leaves/plant debris and keep tools clean",
    ]


def _smart_treatment_tips(disease_title: str, realtime: Optional[Dict[str, Any]]) -> List[str]:
    """Weather-aware, safe, general guidance (no chemical dosages)."""
    title_l = (disease_title or "").lower()
    tips: List[str] = []

    if "healthy" in title_l:
        return ["Plant looks healthy — no treatment needed. Keep monitoring and follow general prevention tips."]

    if realtime:
        humid = _parse_float(realtime.get("humidity_percent"))
        rain = _parse_float(realtime.get("rain_mm"))
        precip = _parse_float(realtime.get("precip_mm"))

        if (rain is not None and rain > 1.0) or (precip is not None and precip > 1.0):
            tips.append("Rain/precipitation detected — avoid spraying right now; recheck after conditions dry.")
        if humid is not None and humid >= 75.0:
            tips.append("High humidity increases fungal risk — improve airflow and avoid overhead watering.")

    fungal_keywords = ("mildew", "rust", "blight", "spot", "mold", "powder")
    if any(k in title_l for k in fungal_keywords):
        tips.append("Likely fungal-type disease — remove infected leaves and consider a fungicide per label guidance.")
    else:
        tips.append("Remove heavily infected leaves and sanitize tools to reduce spread.")

    tips.append("If symptoms worsen, consult a local agriculture extension/expert for region-specific advice.")
    return tips


def _weights_version(path_str: Optional[str]) -> Optional[str]:
    try:
        if not path_str:
            return None
        p = Path(path_str)
        if not p.is_file():
            return None
        st = p.stat()
        return f"{p.name} (size={st.st_size} bytes, mtime={int(st.st_mtime)})"
    except Exception:
        return None


def _predict_topk(image: Image.Image, k: int = 3):
    if model is None:
        raise RuntimeError(MODEL_LOAD_ERROR or "Model is not loaded")

    tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).detach().cpu()

    top_probs, top_idxs = probs.topk(k)
    pred_idx = int(top_idxs[0].item())
    return pred_idx, top_idxs.tolist(), top_probs.tolist(), tensor


def _get_last_conv_layer(net: nn.Module) -> Optional[nn.Module]:
    last_conv = None
    for module in net.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    return last_conv


def _build_transfer_model(arch: str, num_classes: int) -> nn.Module:
    if tv_models is None:
        raise RuntimeError("torchvision is required for transfer-learning models but could not be imported")

    arch = (arch or "").lower().strip()
    if arch == "efficientnet_b0":
        model = tv_models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    if arch == "resnet50":
        model = tv_models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported checkpoint arch '{arch}'")


def _generate_gradcam_heatmap(input_tensor: torch.Tensor, class_idx: int) -> Optional[np.ndarray]:
    """Return a normalized Grad-CAM heatmap (H x W, float32 in [0,1])."""
    if model is None:
        return None

    last_conv = _get_last_conv_layer(model)
    if last_conv is None:
        return None

    activations = None
    gradients = None

    def _forward_hook(_module, _inp, out):
        nonlocal activations
        activations = out

    def _backward_hook(_module, _grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    fh = last_conv.register_forward_hook(_forward_hook)
    try:
        bh = last_conv.register_full_backward_hook(_backward_hook)
    except AttributeError:
        bh = last_conv.register_backward_hook(_backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        logits = model(input_tensor)
        score = logits[0, class_idx]
        score.backward(retain_graph=False)

        if activations is None or gradients is None:
            return None

        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * activations).sum(dim=1, keepdim=False)  # (1, H, W)
        cam = F.relu(cam).squeeze(0)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam.detach().cpu().numpy().astype(np.float32)
    finally:
        fh.remove()
        bh.remove()


def _estimate_severity(title: str, heatmap: Optional[np.ndarray]) -> Tuple[Optional[float], Optional[str]]:
    """Estimate affected area % from heatmap; returns (percent, level)."""
    if heatmap is None:
        return None, None

    if title and "healthy" in title.lower():
        return 0.0, "None"

    h = np.asarray(heatmap, dtype=np.float32)
    if h.size == 0:
        return None, None

    affected = float((h > 0.60).mean() * 100.0)

    if affected < 10.0:
        level = "Mild"
    elif affected < 25.0:
        level = "Moderate"
    else:
        level = "Severe"

    return round(affected, 2), level


def _generate_cam_overlay(image: Image.Image, input_tensor: torch.Tensor, class_idx: int) -> Optional[Image.Image]:
    """Generate a simple Grad-CAM overlay image (224x224) for the given class."""
    heat = _generate_gradcam_heatmap(input_tensor, class_idx)
    if heat is None:
        return None

    cam_u8 = (heat * 255.0).astype(np.uint8)
    heat_alpha = Image.fromarray(cam_u8, mode="L").resize((224, 224))
    heat_rgba = Image.new("RGBA", (224, 224), (255, 0, 0, 0))
    heat_rgba.putalpha(heat_alpha.point(lambda a: int(a * 0.55)))

    base = image.resize((224, 224)).convert("RGBA")
    overlay = Image.alpha_composite(base, heat_rgba)
    return overlay.convert("RGB")


def _extract_plant_name(label: str) -> Optional[str]:
    if not label:
        return None

    # Typical labels look like: "Pepper bell : Healthy" or "Tomato : Early Blight"
    if ":" in label:
        plant = label.split(":", 1)[0].strip()
        return plant or None

    # Special case in this dataset
    if "without leaves" in label.lower():
        return None

    # Fallback: first token/phrase
    return label.strip() or None


def _load_model_meta(weights_path: Path) -> Optional[Dict[str, Any]]:
    meta_path = weights_path.with_suffix(weights_path.suffix + ".meta.json")
    if not meta_path.is_file():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_model():
    candidates = []

    env_path = os.getenv("PLANT_DISEASE_MODEL_PATH")
    if env_path:
        candidates.append(Path(env_path))

    # Common filenames referenced in this repo's READMEs
    candidates.extend(
        [
            BASE_DIR / "plant_disease_model_1_latest.pt",
            BASE_DIR / "plant_disease_model_1.pt",
        ]
    )

    for candidate in candidates:
        try:
            if candidate.is_file():
                loaded = torch.load(candidate, map_location=DEVICE)

                # Support both legacy state_dict files and checkpoint dicts.
                if isinstance(loaded, dict) and "state_dict" in loaded:
                    arch = str(loaded.get("arch") or "")
                    num_classes = int(loaded.get("num_classes") or 39)
                    loaded_model = _build_transfer_model(arch, num_classes)
                    loaded_model.load_state_dict(loaded["state_dict"])
                else:
                    loaded_model = CNN.CNN(39)
                    loaded_model.load_state_dict(loaded)

                loaded_model.to(DEVICE)
                loaded_model.eval()
                return loaded_model, str(candidate), None, _load_model_meta(candidate)
        except Exception as exc:
            return (
                None,
                str(candidate),
                f"Failed to load model weights from '{candidate}': {exc}",
                None,
            )

    expected = ", ".join(["plant_disease_model_1_latest.pt", "plant_disease_model_1.pt"])
    return (
        None,
        None,
        (
            "Model weights file not found. "
            f"Download the pre-trained weights and place one of these files in '{BASE_DIR}': {expected}. "
            "You can also set the env var PLANT_DISEASE_MODEL_PATH to an absolute path to the .pt file."
        ),
        None,
    )


model, _model_path, MODEL_LOAD_ERROR, MODEL_META = _load_model()

if MODEL_LOAD_ERROR:
    print(f"[WARN] {MODEL_LOAD_ERROR}")
else:
    print(f"[INFO] Loaded model weights from '{_model_path}' on device '{DEVICE}'")

def prediction(image_path):
    image = Image.open(image_path).convert("RGB")
    pred_idx, _top_idxs, _top_probs, _tensor = _predict_topk(image, k=1)
    return pred_idx


app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')


@app.get('/api/weather')
def api_weather():
    lat = _parse_float(request.args.get('lat'))
    lon = _parse_float(request.args.get('lon'))
    if lat is None or lon is None:
        return jsonify({"error": "lat and lon are required"}), 400

    realtime = _fetch_realtime_weather(lat, lon)
    if realtime is None:
        return jsonify({"error": "weather unavailable"}), 503

    place = _reverse_geocode_place_cached(lat, lon)
    if place is not None:
        realtime.update(place)

    return jsonify(realtime)


@app.route('/history')
def history_page():
    scans = list_scans(HISTORY_DB_PATH, limit=40)
    # Build URLs for images stored in uploads
    scan_dicts = []
    for s in scans:
        scan_dicts.append(
            {
                "id": s.id,
                "created_at": s.created_at,
                "pred_label": s.pred_label,
                "confidence": s.confidence,
                "image_url": url_for('static', filename=f'uploads/{s.image_filename}'),
                "severity_percent": s.severity_percent,
                "severity_level": s.severity_level,
                "feedback_correct": s.feedback_correct,
                "feedback_label": s.feedback_label,
                "report_url": url_for('download_report', scan_id=s.id),
            }
        )
    return render_template('history.html', scans=scan_dicts)


@app.post('/feedback')
def feedback_submit():
    scan_id = request.form.get('scan_id')
    correct = request.form.get('correct')
    label = request.form.get('correct_label') or None

    try:
        sid = int(scan_id)
    except Exception:
        return redirect(url_for('history_page'))

    is_correct = str(correct).lower() in {"1", "true", "yes"}
    try:
        set_feedback(HISTORY_DB_PATH, scan_id=sid, correct=is_correct, correct_label=(None if is_correct else label))
    except Exception:
        pass

    return redirect(url_for('history_page'))


@app.get('/report/<int:scan_id>')
def download_report(scan_id: int):
    scan = get_scan(HISTORY_DB_PATH, scan_id)
    if scan is None:
        return ("Not found", 404)

    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
            Image as RLImage,
            ListFlowable,
            ListItem,
        )
    except Exception:
        return ("PDF support not installed. Install 'reportlab' and retry.", 500)

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40,
        title="Plant Disease Detection — Scan Report",
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=16, spaceAfter=10))
    styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12, spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="Body", parent=styles["BodyText"], fontName="Helvetica", fontSize=10, leading=13, spaceAfter=6))

    def _escape(text: str) -> str:
        return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def prich(markup: str) -> Paragraph:
        """Paragraph that can contain simple ReportLab markup (e.g., <b>).</n+
        Only pass trusted markup here; escape user/data content first.
        """
        return Paragraph(markup or "", styles["Body"])

    def pplain(text: str) -> Paragraph:
        return Paragraph(_escape(text or ""), styles["Body"])

    def bullets(items: List[str]) -> ListFlowable:
        lis = [ListItem(pplain(str(it)), leftIndent=0) for it in items if str(it).strip()]
        return ListFlowable(
            lis,
            bulletType="bullet",
            leftIndent=18,
            bulletFontName="Helvetica",
            bulletFontSize=10,
            bulletDedent=6,
        )

    story = []
    story.append(Paragraph("Plant Disease Detection — Scan Report", styles["H1"]))
    story.append(prich(f"<b>Scan ID:</b> {_escape(str(scan.id))} &nbsp;&nbsp; <b>Time (UTC):</b> {_escape(str(scan.created_at))}"))

    # Summary table
    label = scan.pred_label or "Uncertain"
    conf = (f"{scan.confidence * 100.0:.2f}%" if scan.confidence is not None else "-")
    sev = "-"
    if scan.severity_level:
        sev = scan.severity_level
        if scan.severity_percent is not None:
            sev += f" ({scan.severity_percent:.2f}% area)"

    summary = Table(
        [["Prediction", label], ["Confidence", conf], ["Severity", sev]],
        colWidths=[110, 370],
        hAlign="LEFT",
    )
    summary.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(summary)
    story.append(Spacer(1, 10))

    # Disease-specific details
    if scan.pred_idx is not None:
        try:
            row = disease_info.iloc[int(scan.pred_idx)]
            disease_name_col = _df_col(disease_info, "disease_name", "disease name") or "disease_name"
            description_col = _df_col(disease_info, "description", "desc") or "description"
            steps_col = _df_col(disease_info, "Possible Steps", "possible steps", "steps") or "Possible Steps"
            symptoms_col = _df_col(disease_info, "symptoms", "Symptoms")

            disease_title = str(row.get(disease_name_col, label))
            disease_desc = str(row.get(description_col, ""))
            disease_steps = str(row.get(steps_col, ""))

            story.append(Paragraph("Disease Details", styles["H2"]))

            plant = _extract_plant_name(disease_title)
            if plant:
                story.append(prich(f"<b>Plant:</b> {_escape(plant)}"))

            if disease_desc:
                story.append(prich(f"<b>Description:</b> {_escape(disease_desc)}"))

            if symptoms_col:
                symptoms = _split_points(str(row.get(symptoms_col, "")), max_items=6)
            else:
                symptoms = _extract_symptoms_from_description(disease_title, disease_desc, max_items=6)
            if symptoms:
                story.append(prich("<b>Symptoms:</b>"))
                story.append(bullets(symptoms))

            steps = _split_points(disease_steps, max_items=10)
            if steps:
                story.append(prich("<b>Recovery / Treatment Steps:</b>"))
                story.append(bullets(steps))

            smart = _smart_treatment_tips(disease_title, scan.weather)
            if smart:
                story.append(prich("<b>Smart Tips:</b>"))
                story.append(bullets(smart))

            prev = _general_prevention_tips()
            if prev:
                story.append(prich("<b>Prevention Tips:</b>"))
                story.append(bullets(prev))
        except Exception:
            pass

    # Weather section
    if scan.weather:
        story.append(Paragraph("Weather", styles["H2"]))
        place = scan.weather.get("place") or ""
        cond = scan.weather.get("condition") or ""
        t = scan.weather.get("temperature_c")
        hum = scan.weather.get("humidity_percent")
        wind = scan.weather.get("wind_kph")
        parts = []
        if place:
            parts.append(f"<b>Place:</b> {_escape(place)}")
        if cond:
            parts.append(f"<b>Condition:</b> {_escape(cond)}")
        parts.append(f"<b>Temp:</b> {_escape(str(t))}°C &nbsp;&nbsp; <b>Humidity:</b> {_escape(str(hum))}% &nbsp;&nbsp; <b>Wind:</b> {_escape(str(wind))} km/h")
        story.append(prich("<br/>".join(parts)))

    # Feedback section
    if scan.feedback_correct is not None:
        story.append(Paragraph("Feedback", styles["H2"]))
        if scan.feedback_correct:
            story.append(pplain("Marked correct by user"))
        else:
            story.append(pplain(f"Marked wrong by user. Correct label: {scan.feedback_label or '-'}"))

    # Image at the end (scaled, no overlap)
    img_path = UPLOAD_DIR / scan.image_filename
    if img_path.is_file():
        try:
            story.append(Paragraph("Uploaded Image", styles["H2"]))
            img = RLImage(str(img_path))
            max_w = doc.width
            max_h = 320
            iw = float(getattr(img, "imageWidth", max_w) or max_w)
            ih = float(getattr(img, "imageHeight", max_h) or max_h)
            if iw > 0 and ih > 0:
                scale = min(max_w / iw, max_h / ih, 1.0)
                img.drawWidth = iw * scale
                img.drawHeight = ih * scale
            story.append(img)
        except Exception:
            pass

    doc.build(story)
    buf.seek(0)
    return send_file(
        buf,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"scan_report_{scan.id}.pdf",
    )

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(url_for('ai_engine_page'))

        images = request.files.getlist('image')
        images = [im for im in images if im and im.filename]
        if not images:
            return redirect(url_for('ai_engine_page'))

        image = images[0]

        # Read the uploaded bytes once and use those bytes for BOTH saving and inference.
        # This avoids issues where an old on-disk path is re-used or overwritten.
        image_bytes = image.read()

        original_name = secure_filename(image.filename) or "upload.jpg"
        ext = Path(original_name).suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            ext = ".jpg"

        filename = f"upload_{uuid.uuid4().hex}{ext}"
        saved_path = UPLOAD_DIR / filename
        saved_path.write_bytes(image_bytes)

        # Cache-bust the browser so it always shows the newest uploaded file.
        uploaded_image_url = url_for('static', filename=f'uploads/{filename}', v=str(int(time.time())))

        try:
            pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
            warnings = _assess_image_quality(pil_image)

            lat = _parse_float(request.form.get("lat"))
            lon = _parse_float(request.form.get("lon"))
            realtime = None
            if lat is not None and lon is not None:
                realtime = _fetch_realtime_weather(lat, lon)
                place = _reverse_geocode_place_cached(lat, lon)
                if realtime is not None and place is not None:
                    realtime.update(place)

            pred, top_idxs, top_probs, input_tensor = _predict_topk(pil_image, k=3)
            disease_name_col = _df_col(disease_info, "disease_name", "disease name") or "disease_name"
            top_predictions = [
                {
                    "idx": int(i),
                    "name": str(disease_info.iloc[int(i)][disease_name_col]),
                    "confidence": float(p) * 100.0,
                }
                for i, p in zip(top_idxs, top_probs)
            ]

            cam_url = None
            cams = []
            # Generate explanation overlays for top predictions (can be a bit slow on CPU).
            for rank, cls_idx in enumerate(top_idxs[:3], start=1):
                cam_img = _generate_cam_overlay(pil_image, input_tensor, int(cls_idx))
                if cam_img is None:
                    continue
                cam_name = f"cam_{Path(filename).stem}_top{rank}.jpg"
                cam_path = UPLOAD_DIR / cam_name
                cam_img.save(cam_path, format="JPEG", quality=90)
                cams.append({
                    "rank": rank,
                    "class_idx": int(cls_idx),
                    "label": str(disease_info.iloc[int(cls_idx)][disease_name_col]),
                    "url": url_for('static', filename=f'uploads/{cam_name}'),
                })

            # Keep backward-compat for template variable (top-1)
            if cams:
                cam_url = cams[0]["url"]

            pred_confidence_score = float(top_probs[0])
            pred_confidence_percent = pred_confidence_score * 100.0

            is_uncertain = pred_confidence_score < CONFIDENCE_THRESHOLD
        except Exception as exc:
            # Render the usual result page but with a clear message instead of crashing.
            return render_template(
                'submit.html',
                title='Model not available',
                desc=str(exc),
                prevent='Download the model weights and restart the app.',
                image_url=uploaded_image_url,
                uploaded_image_url=uploaded_image_url,
                top_predictions=[],
                warnings=[],
                cam_url=None,
                cams=[],
                pred_confidence_score=None,
                pred_confidence_percent=None,
                symptoms=[],
                treatment_steps=[],
                prevention_tips=[],
                model_meta=None,
                realtime=None,
                plant_name=None,
                pred=4,
                sname='',
                simage='',
                buy_link='',
            )

        if is_uncertain:
            title = "Uncertain prediction"
            description = (
                "Model confidence is low for this image. "
                "Try taking a clearer photo of a single leaf (good lighting, focused, plain background). "
                "You can also upload multiple images and compare the top predictions."
            )
            prevent = "\n".join(_general_prevention_tips())
            image_url = ""
            symptoms = []
            treatment_steps = []
            prevention_tips = _general_prevention_tips()
            plant_name = None
            severity_percent = None
            severity_level = None
            smart_tips: List[str] = []
        else:
            # Use iloc so we always index by position (model output index).
            row = disease_info.iloc[int(pred)]
            disease_name_col = _df_col(disease_info, "disease_name", "disease name") or "disease_name"
            description_col = _df_col(disease_info, "description", "desc") or "description"
            steps_col = _df_col(disease_info, "Possible Steps", "possible steps", "steps") or "Possible Steps"
            image_url_col = _df_col(disease_info, "image_url", "image url") or "image_url"
            symptoms_col = _df_col(disease_info, "symptoms", "Symptoms")

            title = str(row.get(disease_name_col, ""))
            description = str(row.get(description_col, ""))
            prevent = str(row.get(steps_col, ""))
            image_url = str(row.get(image_url_col, ""))

            plant_name = _extract_plant_name(str(title))

            if symptoms_col:
                symptoms = _split_points(str(row.get(symptoms_col, "")), max_items=6)
            else:
                symptoms = _extract_symptoms_from_description(title, description, max_items=6)
            treatment_steps = _split_points(str(prevent), max_items=10)
            prevention_tips = _general_prevention_tips()

            heat = _generate_gradcam_heatmap(input_tensor, int(pred))
            severity_percent, severity_level = _estimate_severity(title, heat)
            smart_tips = _smart_treatment_tips(title, realtime)

        model_meta = {
            "dataset": "PlantVillage",
            "model": "CNN (PyTorch)",
            "device": str(DEVICE),
            "weights": _model_path,
            "weights_version": _weights_version(_model_path),
            "accuracy": (MODEL_META or {}).get("metrics", {}).get("test_acc"),
        }

        if is_uncertain:
            supplement_name = ""
            supplement_image_url = ""
            supplement_buy_link = ""
        else:
            srow = supplement_info.iloc[int(pred)]
            supplement_name = str(srow.get('supplement name', ''))
            supplement_image_url = str(srow.get('supplement image', ''))
            supplement_buy_link = str(srow.get('buy link', ''))

        # Persist scan history (best-effort)
        scan_id = None
        try:
            scan_id = add_scan_return_id(
                HISTORY_DB_PATH,
                image_filename=filename,
                pred_idx=(None if is_uncertain else int(pred)),
                pred_label=(None if is_uncertain else str(title)),
                confidence=float(pred_confidence_score),
                top_predictions=top_predictions,
                weather=realtime,
                severity_percent=severity_percent,
                severity_level=severity_level,
                model_name=str(model_meta.get("model")),
                model_weights=str(_model_path or ""),
            )
        except Exception:
            pass

        pred_for_template = 4 if is_uncertain else pred

        batch_results = []
        if len(images) > 1:
            disease_name_col = _df_col(disease_info, "disease_name", "disease name") or "disease_name"
            for extra in images[1:6]:
                try:
                    extra_bytes = extra.read()
                    extra_name = secure_filename(extra.filename) or "extra.jpg"
                    extra_ext = Path(extra_name).suffix.lower()
                    if extra_ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                        extra_ext = ".jpg"
                    extra_filename = f"upload_{uuid.uuid4().hex}{extra_ext}"
                    (UPLOAD_DIR / extra_filename).write_bytes(extra_bytes)
                    extra_img = Image.open(BytesIO(extra_bytes)).convert("RGB")
                    e_pred, _tidxs, e_probs, _t = _predict_topk(extra_img, k=1)
                    batch_results.append(
                        {
                            "image_url": url_for('static', filename=f'uploads/{extra_filename}', v=str(int(time.time()))),
                            "pred_label": str(disease_info.iloc[int(e_pred)][disease_name_col]),
                            "confidence_percent": float(e_probs[0]) * 100.0,
                        }
                    )
                except Exception:
                    continue

        all_labels = list((disease_info[_df_col(disease_info, "disease_name", "disease name") or "disease_name"]).astype(str).tolist())

        return render_template('submit.html' , title = title , desc = description , prevent = prevent , 
                       image_url = image_url , uploaded_image_url = uploaded_image_url , pred = pred_for_template ,
                               top_predictions = top_predictions , warnings = warnings , cam_url = cam_url,
                               cams = cams,
                               pred_confidence_score = pred_confidence_score,
                               pred_confidence_percent = pred_confidence_percent,
                               symptoms = symptoms,
                               treatment_steps = treatment_steps,
                               smart_tips = smart_tips,
                               severity_percent = severity_percent,
                               severity_level = severity_level,
                               prevention_tips = prevention_tips,
                               model_meta = model_meta,
                               realtime = realtime,
                               plant_name = plant_name,
                               sname = supplement_name , simage = supplement_image_url , buy_link = supplement_buy_link,
                               scan_id = scan_id,
                               all_labels = all_labels,
                               batch_results = batch_results)

    return redirect(url_for('ai_engine_page'))

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image = list(supplement_info['supplement image']),
                           supplement_name = list(supplement_info['supplement name']), disease = list(disease_info['disease_name']), buy = list(supplement_info['buy link']))

if __name__ == '__main__':
    app.run(debug=True)
