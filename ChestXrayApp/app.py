"""
ClinicalAI Radiology Platform — v2.0
Fixed: patient_id system, Python API retraining, YAML generation,
       extension-safe labels, threading-based background train,
       model reload on activation, image viewer improvements.
"""

import streamlit as st
import sqlite3, os, datetime, json, hashlib, io, re, shutil, threading, subprocess
import numpy as np
import cv2
import requests
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import yaml
from PIL import Image
from ultralytics import YOLO
from fpdf import FPDF

# ================================================================
# CONFIG
# ================================================================
MODEL_PATH = "../cxray14/runs/detect/runs/cxr14_yolov12m/weights/best.pt"
MODEL_DIR = "models"
UPLOAD_DIR = "uploads"
FEEDBACK_IMG = "feedback_dataset/images"
FEEDBACK_LBL = "feedback_dataset/labels"
DB_PATH = "database/scans.db"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:14b"

CLASS_NAMES = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Consolidation",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
]

SEVERITY = {
    "Pneumothorax": "CRITICAL",
    "Cardiomegaly": "HIGH",
    "Pleural effusion": "HIGH",
    "Consolidation": "HIGH",
    "Lung Opacity": "HIGH",
    "Nodule/Mass": "HIGH",
    "Pulmonary fibrosis": "MODERATE",
    "ILD": "MODERATE",
    "Aortic enlargement": "MODERATE",
    "Infiltration": "MODERATE",
    "Atelectasis": "MODERATE",
    "Pleural thickening": "LOW",
    "Calcification": "LOW",
    "Other lesion": "LOW",
}

SEV_COLOR = {
    "CRITICAL": (220, 20, 60),
    "HIGH": (255, 100, 0),
    "MODERATE": (220, 180, 0),
    "LOW": (0, 200, 100),
}
SEV_ORDER = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2, "LOW": 3}

for _d in [UPLOAD_DIR, "database", MODEL_DIR, FEEDBACK_IMG, FEEDBACK_LBL]:
    os.makedirs(_d, exist_ok=True)

# ================================================================
# PAGE CONFIG + CSS
# ================================================================
st.set_page_config(
    page_title="ClinicalAI Radiology",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/icon?family=Material+Icons');
body { font-family: 'Inter', sans-serif; }

/* ── Restore icon fonts — covers every Streamlit icon-span pattern ── */
/*
 * Root cause: `* { font-family: 'Inter' !important }` overrides the
 * icon-ligature fonts (Material Icons, Material Symbols) on every element.
 * Without the correct font the browser renders the raw ligature text
 * (e.g. "arrow_right", "_arrow_right", "upload") instead of a glyph.
 * The selectors below restore the icon font on all known Streamlit patterns.
 */
.material-icons,
[class*="material-icon"],
[class*="material-symbol"],
/* Streamlit expander toggle — data-testid variants */
[data-testid="stExpanderToggleIcon"],
[data-testid="stExpanderToggleIcon"] *,
/* Class-name patterns Streamlit uses for arrows (_arrow_right etc.) */
[class*="_arrow"],
[class*="arrow_"],
[class*="-arrow"],
/* Generic icon class patterns */
[class*="icon_"],
[class*="_icon"],
.icon,
/* File-uploader dropzone icon span */
[data-testid="stFileUploaderDropzone"] button span,
[data-testid="stFileUploaderDropzoneInstructions"] span:first-child,
.stFileUploader button span,
/* Expander / button first-child spans that hold icon glyphs */
.stExpander details summary > span:first-child,
.streamlit-expanderHeader > span:first-child,
button[kind] > span:first-child {
    font-family: 'Material Icons', 'Material Icons Round',
                 'Material Symbols Outlined', 'Material Symbols Rounded' !important;
    font-feature-settings: 'liga' 1 !important;
    -webkit-font-feature-settings: 'liga' 1 !important;
    text-rendering: optimizeLegibility;
    font-size: 18px;
    display: inline-block;
    line-height: 1;
    text-transform: none;
    letter-spacing: normal;
    word-wrap: normal;
    white-space: nowrap;
    direction: ltr;
    -webkit-font-smoothing: antialiased;
}

[data-testid="stAppViewContainer"] { background:#080b14; color:#e2e8f0; }
[data-testid="stSidebar"]          { background:#0d1117; border-right:1px solid #1e2535; }

.glass-card {
    background:linear-gradient(135deg,#111827,#1a2035);
    border:1px solid #1e2d45; border-radius:16px;
    padding:20px 24px; margin:10px 0;
}
.metric-card {
    background:linear-gradient(135deg,#0f172a,#1e2535);
    border:1px solid #2d3a52; border-radius:14px;
    padding:18px 20px; text-align:center; margin:4px 0;
}
.metric-value { font-size:2.2rem; font-weight:700; color:#4f8ef7; margin:0; }
.metric-label { font-size:0.78rem; color:#64748b; letter-spacing:.08em; text-transform:uppercase; }

.alert-critical {
    background:linear-gradient(90deg,#3d0a0a,#1a0505);
    border:1px solid #dc143c; border-left:5px solid #dc143c;
    border-radius:10px; padding:14px 18px; margin:8px 0;
    animation:pulse-border 2s infinite;
}
@keyframes pulse-border { 0%,100%{border-left-color:#dc143c} 50%{border-left-color:#ff4466} }

.alert-high {
    background:linear-gradient(90deg,#2d1a00,#1a1000);
    border:1px solid #ff6400; border-left:5px solid #ff6400;
    border-radius:10px; padding:14px 18px; margin:8px 0;
}
.finding-row {
    background:#111827; border-radius:10px;
    padding:12px 16px; margin:6px 0;
    display:flex; justify-content:space-between; align-items:center;
    border:1px solid #1e2d45;
}
.badge { padding:3px 10px; border-radius:20px; font-size:.72rem; font-weight:600; }
.badge-CRITICAL { background:#3d0a0a; color:#ff4466; border:1px solid #dc143c; }
.badge-HIGH     { background:#2d1a00; color:#ff8040; border:1px solid #ff6400; }
.badge-MODERATE { background:#2d2500; color:#ffd040; border:1px solid #ddb200; }
.badge-LOW      { background:#002d1a; color:#40ff90; border:1px solid #00c864; }

.profile-card {
    background:linear-gradient(135deg,#0f172a,#1e2535);
    border:1px solid #2d3a52; border-radius:14px;
    padding:16px; margin-bottom:8px;
}
.section-header {
    font-size:.7rem; font-weight:600; letter-spacing:.12em;
    text-transform:uppercase; color:#4f6080; margin:16px 0 6px 0;
}
.log-box {
    background:#050810; border:1px solid #1e2d45; border-radius:10px;
    padding:14px 18px; font-family:'Courier New',monospace;
    font-size:.82rem; color:#4ade80;
    max-height:300px; overflow-y:auto; white-space:pre-wrap;
}
.dot-online  { width:8px;height:8px;background:#00c864;border-radius:50%;
               display:inline-block;margin-right:6px;box-shadow:0 0 6px #00c864; }
.dot-offline { width:8px;height:8px;background:#dc143c;border-radius:50%;
               display:inline-block;margin-right:6px; }
h1,h2,h3,h4 { color:#f1f5f9 !important; }
hr { border-color:#1e2535; }
.stButton>button { border-radius:8px !important; font-weight:500 !important; }
.stTabs [data-baseweb="tab-list"] { background:#111827; border-radius:10px; }
.stTabs [aria-selected="true"]    { background:#1e2d45 !important; color:#4f8ef7 !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ================================================================
# DATABASE  —  patient_id-based schema
# ================================================================
@st.cache_resource
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""CREATE TABLE IF NOT EXISTS users(
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        username     TEXT UNIQUE,
        password     TEXT,
        role         TEXT,
        patient_id   TEXT,          -- FK → patients.patient_id  (NULL for doctors/admins)
        full_name    TEXT,
        specialization TEXT,
        hospital     TEXT,
        phone        TEXT,
        email        TEXT,
        created_at   TEXT)""")

    conn.execute("""CREATE TABLE IF NOT EXISTS patients(
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id     TEXT UNIQUE,
        username       TEXT UNIQUE,  -- links to users.username
        full_name      TEXT,
        dob            TEXT,
        gender         TEXT,
        blood_group    TEXT,
        phone          TEXT,
        address        TEXT,
        medical_history TEXT,
        created_at     TEXT)""")

    # scans now stores patient_id (not bare name) + doctor_id
    conn.execute("""CREATE TABLE IF NOT EXISTS scans(
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        doctor_id       TEXT,        -- users.username of the doctor
        patient_id      TEXT,        -- patients.patient_id
        patient_name    TEXT,        -- denormalised for display only
        filename        TEXT,
        findings        TEXT,
        report          TEXT,
        date            TEXT,
        conf            REAL,
        doctor_notes    TEXT,
        reviewed        INTEGER DEFAULT 0,
        edited_findings TEXT,
        edited_report   TEXT)""")

    conn.execute("""CREATE TABLE IF NOT EXISTS feedback(
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        scan_id             INTEGER,
        doctor_id           TEXT,
        image_path          TEXT,
        original_findings   TEXT,
        corrected_findings  TEXT,
        correction_notes    TEXT,
        timestamp           TEXT,
        used_in_training    INTEGER DEFAULT 0)""")

    conn.execute("""CREATE TABLE IF NOT EXISTS models(
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        version       TEXT UNIQUE,
        path          TEXT,
        training_date TEXT,
        dataset_size  INTEGER,
        notes         TEXT,
        active        INTEGER DEFAULT 0)""")

    conn.execute("""CREATE TABLE IF NOT EXISTS alerts(
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        scan_id      INTEGER,
        patient_id   TEXT,
        patient_name TEXT,
        severity     TEXT,
        finding      TEXT,
        doctor_id    TEXT,
        timestamp    TEXT,
        acknowledged INTEGER DEFAULT 0)""")

    # Idempotent migrations — add columns if they don't exist yet
    for _migration in [
        "ALTER TABLE users ADD COLUMN status TEXT DEFAULT 'approved'",
        "ALTER TABLE scans ADD COLUMN scan_status TEXT DEFAULT 'AI Generated'",
    ]:
        try:
            conn.execute(_migration)
        except Exception:
            pass
    conn.commit()
    return conn


db = get_db()


# ---- auth helpers ----
def _hp(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def _seed_default_admin():
    """Create a default admin account if none exists."""
    existing = db.execute("SELECT id FROM users WHERE role='Admin' LIMIT 1").fetchone()
    if not existing:
        db.execute(
            "INSERT INTO users(username,password,role,full_name,status,created_at)"
            " VALUES(?,?,?,?,?,?)",
            (
                "admin",
                _hp("admin123"),
                "Admin",
                "System Administrator",
                "approved",
                str(datetime.datetime.now()),
            ),
        )
        db.commit()


_seed_default_admin()


def register_user(
    username, password, role, full_name="", spec="", hospital="", status="approved"
):
    """Register doctor/admin. Patient accounts are created via patient profile form."""
    try:
        db.execute(
            "INSERT INTO users(username,password,role,full_name,specialization,hospital,status,created_at)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (
                username,
                _hp(password),
                role,
                full_name,
                spec,
                hospital,
                status,
                str(datetime.datetime.now()),
            ),
        )
        db.commit()
        return True
    except Exception:
        return False


def login_user(username, password):
    row = db.execute(
        "SELECT role, patient_id, status FROM users WHERE username=? AND password=?",
        (username, _hp(password)),
    ).fetchone()
    if not row:
        return (None, None)
    role_, pid_, status_ = row[0], row[1], (row[2] or "approved")
    if role_ == "Doctor" and status_ == "pending":
        return ("__pending__", None)
    return (role_, pid_)


def get_user(username):
    return db.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()


# ---- patient helpers ----
def get_patient_by_id(pid):
    return db.execute("SELECT * FROM patients WHERE patient_id=?", (pid,)).fetchone()


def get_patient_by_username(uname):
    return db.execute("SELECT * FROM patients WHERE username=?", (uname,)).fetchone()


def list_patients():
    return db.execute(
        "SELECT patient_id, full_name, dob, gender, blood_group, phone FROM patients ORDER BY full_name"
    ).fetchall()


def next_patient_id():
    row = db.execute(
        "SELECT patient_id FROM patients ORDER BY id DESC LIMIT 1"
    ).fetchone()

    year = datetime.datetime.now().year

    if not row or not row[0]:
        return f"PT-{year}-0001"

    last_id = row[0]

    try:
        last_num = int(last_id.split("-")[-1])
    except:
        last_num = 0

    new_num = last_num + 1
    return f"PT-{year}-{new_num:04d}"


# ================================================================
# MODEL LOADING  —  reloads when active model changes
# ================================================================
def _active_model_path():
    row = db.execute(
        "SELECT path FROM models WHERE active=1 ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row and os.path.exists(row[0]):
        return row[0]
    return MODEL_PATH if os.path.exists(MODEL_PATH) else None


def load_model_from_path(path):
    if path and os.path.exists(path):
        return YOLO(path)
    return None


# Use session-state to allow hot-reload after activation
if "model_path" not in st.session_state:
    st.session_state["model_path"] = _active_model_path()


@st.cache_resource
def _cached_model(path):
    """Cache keyed by path — changing path triggers reload."""
    return load_model_from_path(path)


model = _cached_model(st.session_state["model_path"])


def get_model_info():
    row = db.execute(
        "SELECT version, training_date, dataset_size FROM models WHERE active=1 ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return (
        {"version": row[0], "date": row[1], "dataset": row[2]}
        if row
        else {"version": "v1-base", "date": "Original", "dataset": "VinBigdata"}
    )


# ================================================================
# OLLAMA
# ================================================================
def ollama_report(patient_name, findings, doctor_notes=""):
    if not findings:
        return (
            "No significant radiological findings detected. Lung fields appear clear bilaterally. "
            "Cardiac silhouette within normal limits. No acute cardiopulmonary process identified."
        )
    lines = "\n".join(
        f"- {f['label']} (confidence {f['conf']:.0%}, severity {f['severity']})"
        for f in findings
    )
    notes_block = (
        f"\nClinical context from reviewing physician: {doctor_notes}"
        if doctor_notes
        else ""
    )
    prompt = (
        "You are a senior radiologist writing a formal radiology report.\n"
        "Write a concise clinical impression (4-5 sentences, professional prose, no bullet points).\n\n"
        f"Patient: {patient_name}\nAI-detected findings:\n{lines}{notes_block}\n\n"
        "Include: findings summary, clinical significance, and recommendation."
    )
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 400},
            },
            timeout=90,
        )
        text = r.json().get("response", "").strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text or "Report generation returned empty response."
    except Exception:
        return (
            f"[Ollama offline — run 'ollama serve']\n"
            f"Findings: {', '.join(f['label'] for f in findings)}."
        )


# ================================================================
# DETECTION
# ================================================================
def detect(image: Image.Image, conf_thresh: float):
    img = np.array(image.convert("RGB"))
    res = model.predict(img, conf=conf_thresh, verbose=False)
    boxes = res[0].boxes.xyxy.cpu().numpy()
    classes = res[0].boxes.cls.cpu().numpy()
    scores = res[0].boxes.conf.cpu().numpy()

    findings = []
    draw = img.copy()
    for box, cls, score in zip(boxes, classes, scores):
        x1, y1, x2, y2 = map(int, box)
        label = CLASS_NAMES[int(cls)]
        sev = SEVERITY.get(label, "LOW")
        color = SEV_COLOR[sev]
        findings.append(
            {
                "label": label,
                "conf": float(score),
                "severity": sev,
                "box": [x1, y1, x2, y2],
            }
        )
        cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
        txt = f"{label} {score:.0%}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.rectangle(draw, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            draw,
            txt,
            (x1 + 3, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
        )
    findings.sort(key=lambda x: SEV_ORDER.get(x["severity"], 4))
    return Image.fromarray(draw), findings


def adjust_image(
    arr: np.ndarray, brightness=0, contrast=1.0, invert=False
) -> np.ndarray:
    out = arr.astype(np.float32) * contrast + brightness
    out = np.clip(out, 0, 255).astype(np.uint8)
    return 255 - out if invert else out


# ================================================================
# ALERTS
# ================================================================
def create_alerts(scan_id, patient_id, patient_name, findings, doctor_id):
    for f in findings:
        if f["severity"] in ("CRITICAL", "HIGH"):
            db.execute(
                "INSERT INTO alerts(scan_id,patient_id,patient_name,severity,finding,doctor_id,timestamp,acknowledged)"
                " VALUES(?,?,?,?,?,?,?,0)",
                (
                    scan_id,
                    patient_id,
                    patient_name,
                    f["severity"],
                    f["label"],
                    doctor_id,
                    str(datetime.datetime.now()),
                ),
            )
    db.commit()


def get_unacked_alerts():
    return db.execute(
        "SELECT id,patient_name,severity,finding,doctor_id,timestamp"
        " FROM alerts WHERE acknowledged=0 ORDER BY id DESC"
    ).fetchall()


# ================================================================
# PDF HELPERS
# ================================================================
def sanitize_text(text: str) -> str:
    """Replace Unicode chars that FPDF/latin-1 cannot encode."""
    if not isinstance(text, str):
        text = str(text)
    return (
        text.replace("\u2014", "-")  # em dash
        .replace("\u2013", "-")  # en dash
        .replace("\u201c", '"')  # left double quote
        .replace("\u201d", '"')  # right double quote
        .replace("\u2018", "'")  # left single quote
        .replace("\u2019", "'")  # right single quote
        .replace("\u2026", "...")  # ellipsis
    )


# ================================================================
# PDF
# ================================================================
def make_pdf(
    patient_name,
    patient_id,
    doctor_name,
    findings,
    report,
    date,
    doctor_notes="",
    edited=False,
):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(8, 11, 20)
    pdf.rect(0, 0, 210, 38, "F")
    pdf.set_text_color(79, 142, 247)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_xy(10, 8)
    pdf.cell(0, 10, sanitize_text("ClinicalAI Radiology Report"), ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 116, 139)
    pdf.set_x(10)
    pdf.cell(
        0,
        6,
        sanitize_text(f"Patient ID: {patient_id}  |  Powered by YOLOv12 + Ollama"),
        ln=True,
    )

    pdf.ln(8)
    pdf.set_text_color(40, 40, 40)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(
        0,
        8,
        sanitize_text(
            f"Patient: {patient_name}   |   Physician: {doctor_name}   |   Date: {date}"
        ),
        ln=True,
    )
    if edited:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(79, 142, 247)
        pdf.cell(
            0, 6, sanitize_text("  * Physician-reviewed and edited report"), ln=True
        )

    pdf.ln(3)
    pdf.set_draw_color(30, 45, 69)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 10, "Detected Findings:", ln=True)
    pdf.set_font("Helvetica", "", 11)
    clr = {
        "CRITICAL": (220, 20, 60),
        "HIGH": (200, 80, 0),
        "MODERATE": (180, 140, 0),
        "LOW": (0, 150, 80),
    }
    if findings:
        for f in findings:
            r, g, b = clr.get(f["severity"], (0, 0, 0))
            pdf.set_text_color(r, g, b)
            pdf.cell(
                0,
                8,
                sanitize_text(
                    f"  [{f['severity']}]  {f['label']}  -  Confidence: {f['conf']:.0%}"
                ),
                ln=True,
            )
    else:
        pdf.set_text_color(0, 150, 80)
        pdf.cell(0, 8, sanitize_text("  No abnormalities detected."), ln=True)

    pdf.ln(4)
    pdf.set_draw_color(30, 45, 69)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    pdf.set_text_color(40, 40, 40)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Clinical Impression:", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 8, sanitize_text(report))

    if doctor_notes:
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "Physician Notes:", ln=True)
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(60, 80, 120)
        pdf.multi_cell(0, 7, sanitize_text(doctor_notes))

    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(
        0,
        6,
        sanitize_text(
            "DISCLAIMER: AI-generated. Must be validated by a licensed radiologist."
        ),
        ln=True,
    )

    raw = pdf.output(dest="S")
    # fpdf2 returns bytearray; fpdf1 returns str — handle both
    pdf_bytes = (
        bytes(raw) if isinstance(raw, (bytes, bytearray)) else raw.encode("latin-1")
    )
    return io.BytesIO(pdf_bytes)


# ================================================================
# FEEDBACK & DATASET BUILDER  (BUG-FIXED)
# ================================================================
def _write_yolo_label(image_path: str, findings: list):
    """Write YOLO .txt label — extension-safe, skips empty boxes."""
    base = os.path.splitext(os.path.basename(image_path))[0]  # ← fix #7
    label_path = os.path.join(FEEDBACK_LBL, base + ".txt")
    try:
        img = Image.open(image_path)
        W, H = img.size
        with open(label_path, "w") as lf:
            for f in findings:
                box = f.get("box", [])
                if len(box) != 4:
                    continue
                x1, y1, x2, y2 = box
                cx = ((x1 + x2) / 2) / W
                cy = ((y1 + y2) / 2) / H
                bw = (x2 - x1) / W
                bh = (y2 - y1) / H
                cls_idx = (
                    CLASS_NAMES.index(f["label"]) if f["label"] in CLASS_NAMES else 0
                )
                lf.write(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    except Exception as e:
        st.warning(f"Label write error: {e}")


def _write_dataset_yaml():
    """Write correctly formatted YOLO data.yaml (fix #6)."""
    yaml_data = {
        "path": os.path.abspath("feedback_dataset"),
        "train": "images",
        "val": "images",
        "nc": len(CLASS_NAMES),
        "names": {i: n for i, n in enumerate(CLASS_NAMES)},
    }
    yaml_path = "feedback_dataset/data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)
    return yaml_path


def save_feedback(scan_id, doctor_id, image_path, orig_findings, corr_findings, notes):
    db.execute(
        "INSERT INTO feedback(scan_id,doctor_id,image_path,original_findings,"
        "corrected_findings,correction_notes,timestamp) VALUES(?,?,?,?,?,?,?)",
        (
            scan_id,
            doctor_id,
            image_path,
            json.dumps(orig_findings),
            json.dumps(corr_findings),
            notes,
            str(datetime.datetime.now()),
        ),
    )
    db.commit()
    # copy image to feedback set
    dest = os.path.join(FEEDBACK_IMG, os.path.basename(image_path))
    if os.path.exists(image_path) and not os.path.exists(dest):
        shutil.copy(image_path, dest)
    _write_yolo_label(image_path, corr_findings)


def feedback_stats():
    total = db.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    unused = db.execute(
        "SELECT COUNT(*) FROM feedback WHERE used_in_training=0"
    ).fetchone()[0]
    return total, unused


# ================================================================
# RETRAINING  —  Python API, threading (fixes #5 #8 #9)
# ================================================================
_training_status = {"running": False, "log": ""}


def _run_training(version: str, epochs: int, notes: str):
    """Runs inside a daemon thread — uses YOLO Python API (fix #5)."""
    global _training_status
    _training_status["running"] = True
    log_lines = [
        f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] Training started: {version}"
    ]

    try:
        yaml_path = _write_dataset_yaml()  # fix #6
        log_lines.append(f"Dataset YAML written → {yaml_path}")

        base_path = _active_model_path() or MODEL_PATH
        log_lines.append(f"Base model: {base_path}")
        _training_status["log"] = "\n".join(log_lines)

        train_model = YOLO(base_path)  # fix #5 — Python API
        train_model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=640,
            batch=8,
            project=MODEL_DIR,
            name=version,
            exist_ok=True,
        )

        new_weights = os.path.join(MODEL_DIR, version, "weights", "best.pt")
        log_lines.append(f"Training complete. Weights → {new_weights}")

        if os.path.exists(new_weights):
            db.execute(
                "INSERT OR REPLACE INTO models(version,path,training_date,dataset_size,notes,active)"
                " VALUES(?,?,?,?,?,0)",
                (
                    version,
                    new_weights,
                    str(datetime.datetime.now()),
                    feedback_stats()[0],
                    notes,
                ),
            )
            db.execute(
                "UPDATE feedback SET used_in_training=1 WHERE used_in_training=0"
            )
            db.commit()
            log_lines.append(
                "✅ Model registered in DB. Go to Model Versions to activate."
            )
        else:
            log_lines.append("⚠️ Weights file not found after training.")

    except Exception as e:
        log_lines.append(f"❌ Error: {e}")
    finally:
        _training_status["running"] = False
        _training_status["log"] = "\n".join(log_lines)
        # persist log to file
        log_file = os.path.join(MODEL_DIR, f"log_{version}.txt")
        with open(log_file, "w") as lf:
            lf.write(_training_status["log"])


def trigger_training(version, epochs, notes):
    if _training_status["running"]:
        return False
    t = threading.Thread(  # fix #8 — threading not os.system
        target=_run_training, args=(version, epochs, notes), daemon=True
    )
    t.start()
    return True


# ================================================================
# SESSION STATE
# ================================================================
defaults = {
    "logged_in": False,
    "username": "",
    "role": "",
    "patient_id": None,
    "nav": "🔬 Run Detection",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ================================================================
# LOGIN / REGISTER
# ================================================================
if not st.session_state.logged_in:
    _, col, _ = st.columns([1, 1.6, 1])
    with col:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(
            """
        <div style='text-align:center;margin-bottom:24px'>
            <span style='font-size:3rem'>🫁</span>
            <h1 style='margin:8px 0 4px'>ClinicalAI Radiology</h1>
            <p style='color:#64748b'>AI-Assisted Chest X-ray Diagnostic Platform</p>
        </div>""",
            unsafe_allow_html=True,
        )

        t1, t2 = st.tabs(["🔐 Login", "📝 Register"])
        with t1:
            u = st.text_input("Username", key="lu")
            p = st.text_input("Password", type="password", key="lp")
            if st.button("Login", use_container_width=True, type="primary"):
                role, pid = login_user(u, p)
                if role == "__pending__":
                    st.warning(
                        "⏳ Your account is pending Admin approval. Please check back later."
                    )
                elif role:
                    _nav0 = (
                        "📋 My Reports"
                        if role == "Patient"
                        else "📋 Patient History"
                        if role == "Admin"
                        else "🔬 Run Detection"
                    )
                    st.session_state.update(
                        {
                            "logged_in": True,
                            "username": u,
                            "role": role,
                            "patient_id": pid,
                            "nav": _nav0,
                        }
                    )
                    st.rerun()
                else:
                    st.error("Invalid credentials.")

        with t2:
            st.info(
                "Doctor self-registration only. Your account will need Admin approval before you can log in."
            )
            ru = st.text_input("Username*", key="ru")
            rp = st.text_input("Password*", type="password", key="rp")
            rfn = st.text_input("Full Name*", key="rfn")
            c1, c2 = st.columns(2)
            with c1:
                rsp = st.text_input("Specialization")
            with c2:
                rho = st.text_input("Hospital / Clinic")
            st.caption(
                "Patient accounts are created by doctors inside Patient Profiles."
            )
            if st.button(
                "Request Doctor Account", use_container_width=True, type="primary"
            ):
                if not ru.strip() or not rp.strip() or not rfn.strip():
                    st.error("Username, password and full name are required.")
                elif register_user(ru, rp, "Doctor", rfn, rsp, rho, status="pending"):
                    st.success(
                        "✅ Request submitted! An Admin will approve your account."
                    )
                else:
                    st.error("Username already taken.")
    st.stop()

# ================================================================
# SIDEBAR
# ================================================================
uname = st.session_state.username
role = st.session_state.role
upid = st.session_state.patient_id  # None for doctors/admins
user = get_user(uname)
full_name = user[5] if user and user[5] else uname

with st.sidebar:
    icon = "👨‍⚕️" if role == "Doctor" else ("🔬" if role == "Admin" else "👤")
    st.markdown(
        f"""
    <div class='profile-card'>
        <div style='font-size:2rem'>{icon}</div>
        <div style='font-weight:600;color:#e2e8f0'>{full_name}</div>
        <div style='font-size:.78rem;color:#64748b'>{role} · {user[6] if user and user[6] else ""}</div>
        <div style='font-size:.75rem;color:#4f6080'>{user[7] if user and user[7] else ""}</div>
    </div>""",
        unsafe_allow_html=True,
    )

    # Alert badge
    alerts = get_unacked_alerts()
    if alerts:
        st.markdown(
            f"""
        <div class='alert-critical' style='padding:10px 14px'>
            🚨 <b>{len(alerts)} Alert{"s" if len(alerts) > 1 else ""}</b>
            <div style='font-size:.78rem;color:#ff9999;margin-top:4px'>Unacknowledged critical findings</div>
        </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='section-header'>Navigation</div>", unsafe_allow_html=True)

    # ── Role-based navigation ──────────────────────────────────────
    _doctor_pages = [
        "🔬 Run Detection",
        "📋 Patient History",
        "👥 Patient Profiles",
        "🔄 Scan Comparison",
        "📊 Dashboard",
    ]
    _admin_pages = [
        "📋 Patient History",
        "👥 Patient Profiles",
        "📊 Dashboard",
        "🧠 AI Training",
        "👤 Manage Users",
    ]
    _patient_pages = ["📋 My Reports"]

    if role == "Doctor":
        pages = _doctor_pages
    elif role == "Admin":
        pages = _admin_pages
    else:  # Patient
        pages = _patient_pages

    # If current nav is not in the allowed pages, reset to first allowed
    if st.session_state["nav"] not in pages:
        st.session_state["nav"] = pages[0]

    for page in pages:
        active = st.session_state["nav"] == page
        if st.button(
            page, use_container_width=True, type="primary" if active else "secondary"
        ):
            st.session_state["nav"] = page
            st.rerun()

    nav = st.session_state["nav"]

    st.markdown(
        "<div class='section-header'>System Status</div>", unsafe_allow_html=True
    )
    try:
        requests.get("http://localhost:11434", timeout=2)
        st.markdown(
            "<span class='dot-online'></span><span style='font-size:.82rem;color:#00c864'>Ollama Online</span>",
            unsafe_allow_html=True,
        )
        st.caption(f"LLM: {OLLAMA_MODEL}")
    except Exception:
        st.markdown(
            "<span class='dot-offline'></span><span style='font-size:.82rem;color:#dc143c'>Ollama Offline</span>",
            unsafe_allow_html=True,
        )
        st.caption("Run: `ollama serve`")

    mi = get_model_info()
    st.markdown(
        f"<span class='dot-online'></span><span style='font-size:.82rem;color:#4f8ef7'>YOLO {mi['version']}</span>",
        unsafe_allow_html=True,
    )
    st.caption(f"Dataset: {mi['dataset']}")

    if _training_status["running"]:
        st.warning("⚙️ Retraining in progress…")

    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        for k in defaults:
            st.session_state[k] = defaults[k]
        st.rerun()

# ================================================================
# PAGE — RUN DETECTION
# ================================================================
if nav == "🔬 Run Detection":
    # ── Permission guard ──────────────────────────────────────────
    if role != "Doctor":
        st.error("🚫 Access denied. Only Doctors can run detections.")
        st.stop()

    st.markdown("# 🔬 X-ray Detection & Analysis")

    if model is None:
        st.error("⚠️ YOLO model not found. Check MODEL_PATH in config.")
        st.stop()

    left, right = st.columns([1, 1.2])

    with left:
        st.markdown(
            "<div class='section-header'>Patient Selection</div>",
            unsafe_allow_html=True,
        )
        pts = list_patients()
        options = ["— Select patient —"] + [f"{p[1]} ({p[0]})" for p in pts]
        sel = st.selectbox("Patient", options)
        if sel == "— Select patient —":
            st.info("Register the patient in **Patient Profiles** first.")
            sel_pid, sel_pname = None, ""
        else:
            sel_pname = sel.split(" (")[0]
            sel_pid = sel.split("(")[1].rstrip(")")

        conf = st.slider("Confidence Threshold", 0.10, 0.90, 0.40, 0.05)
        doc_note = st.text_area(
            "Clinical Notes (optional)",
            placeholder="Age, symptoms, relevant history…",
            height=80,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        upload = st.file_uploader(
            "Upload Chest X-ray",
            type=["png", "jpg", "jpeg"],
        )

        if upload:
            img_orig = Image.open(upload)
            st.markdown(
                "<div class='section-header'>Image Viewer</div>", unsafe_allow_html=True
            )
            _iv1, _iv2, _iv3 = st.columns(3)
            with _iv1:
                brightness = st.slider("☀️ Brightness", -80, 80, 0, 5)
            with _iv2:
                contrast = st.slider("🔆 Contrast", 0.5, 2.5, 1.0, 0.1)
            with _iv3:
                invert = st.checkbox("🔄 Invert (X-ray negative)")
            arr_adj = adjust_image(
                np.array(img_orig.convert("RGB")), brightness, contrast, invert
            )
            st.image(arr_adj, caption="Preview", use_container_width=True)

    with right:
        if upload and sel_pid:
            if st.button("🚀 Analyse X-ray", use_container_width=True, type="primary"):
                with st.spinner("Running YOLO detection…"):
                    ann, findings = detect(img_orig, conf)

                st.image(ann, caption="AI Detection Result", use_container_width=True)

                # Critical / high banners
                for f in findings:
                    if f["severity"] == "CRITICAL":
                        st.markdown(
                            f"""
                        <div class='alert-critical'>
                            🚨 <b>CRITICAL: {f["label"]}</b>
                            <div style='font-size:.82rem;margin-top:4px;color:#ffaaaa'>
                            Confidence {f["conf"]:.0%} — Immediate attention required</div>
                        </div>""",
                            unsafe_allow_html=True,
                        )
                    elif f["severity"] == "HIGH":
                        st.markdown(
                            f"""
                        <div class='alert-high'>
                            ⚠️ <b>HIGH: {f["label"]}</b>
                            <div style='font-size:.82rem;margin-top:4px;color:#ffcc88'>
                            Confidence {f["conf"]:.0%} — Priority review recommended</div>
                        </div>""",
                            unsafe_allow_html=True,
                        )

                st.markdown("### Findings")
                if findings:
                    for f in findings:
                        sev = f["severity"]
                        st.markdown(
                            f"""
                        <div class='finding-row'>
                            <span style='color:#e2e8f0;font-weight:500'>🔹 {f["label"]}</span>
                            <span style='color:#94a3b8'>{f["conf"]:.0%}</span>
                            <span class='badge badge-{sev}'>{sev}</span>
                        </div>""",
                            unsafe_allow_html=True,
                        )
                else:
                    st.success("✅ No abnormalities detected")

                st.markdown("### 🤖 Clinical Impression")
                with st.spinner(f"Generating report with {OLLAMA_MODEL}…"):
                    report = ollama_report(sel_pname, findings, doc_note)
                st.markdown(
                    f"""
                <div class='glass-card'>
                    <p style='color:#cbd5e1;line-height:1.7;margin:0'>{report}</p>
                </div>""",
                    unsafe_allow_html=True,
                )

                # Save
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"{sel_pid}_{ts}.png"
                ann.save(os.path.join(UPLOAD_DIR, fname))

                db.execute(
                    "INSERT INTO scans(doctor_id,patient_id,patient_name,filename,"
                    "findings,report,date,conf,doctor_notes) VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        uname,
                        sel_pid,
                        sel_pname,
                        fname,
                        json.dumps(findings),
                        report,
                        str(datetime.datetime.now()),
                        conf,
                        doc_note,
                    ),
                )
                db.commit()
                scan_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
                create_alerts(scan_id, sel_pid, sel_pname, findings, uname)
                st.success("💾 Saved to database")

                pdf_buf = make_pdf(
                    sel_pname,
                    sel_pid,
                    full_name,
                    findings,
                    report,
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    doc_note,
                )
                st.download_button(
                    "📥 Download PDF Report",
                    data=pdf_buf,
                    file_name=f"{sel_pid}_report_{ts}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

        elif upload and not sel_pid:
            st.warning("Please select a patient before analysing.")

# ================================================================
# PAGE — PATIENT HISTORY  (patient sees only their own)
# ================================================================
elif nav in ("📋 Patient History", "📋 My Reports"):
    # ── Permission guard ──────────────────────────────────────────
    if role == "Patient" and nav != "📋 My Reports":
        st.error("🚫 Access denied.")
        st.stop()

    st.markdown("# 📋 Scan History")

    # Acknowledge alerts
    alerts = get_unacked_alerts()
    if alerts and role in ("Doctor", "Admin"):
        with st.expander(f"🚨 {len(alerts)} Unacknowledged Alert(s)", expanded=True):
            for aid, apname, asev, afind, adoc, ats in alerts:
                css = "alert-critical" if asev == "CRITICAL" else "alert-high"
                icon = "🚨" if asev == "CRITICAL" else "⚠️"
                st.markdown(
                    f"""
                <div class='{css}'>
                    {icon} <b>{asev}: {afind}</b> — {apname}
                    <div style='font-size:.78rem;margin-top:3px;color:#aaa'>Dr.{adoc} · {ats[:16]}</div>
                </div>""",
                    unsafe_allow_html=True,
                )
                if st.button("✅ Acknowledge", key=f"ack_{aid}"):
                    db.execute("UPDATE alerts SET acknowledged=1 WHERE id=?", (aid,))
                    db.commit()
                    st.rerun()

    # Query — patients filtered by patient_id (fix #1)
    if role in ("Doctor", "Admin"):
        with st.container():
            c1, c2, c3 = st.columns([3, 2, 2])
            with c1:
                search = st.text_input("🔍 Search patient name")
            with c2:
                f_sev = st.multiselect(
                    "Severity", ["CRITICAL", "HIGH", "MODERATE", "LOW"]
                )
            with c3:
                f_rev = st.selectbox("Status", ["All", "Reviewed", "Pending"])
        st.markdown("<br>", unsafe_allow_html=True)

        q = (
            "SELECT id,doctor_id,patient_id,patient_name,filename,findings,"
            "report,date,doctor_notes,reviewed,edited_findings,edited_report,"
            "COALESCE(scan_status,'AI Generated') FROM scans"
        )
        params = []
        conds = []
        if search.strip():
            conds.append("patient_name LIKE ?")
            params.append(f"%{search}%")
        if f_rev == "Reviewed":
            conds.append("reviewed=1")
        elif f_rev == "Pending":
            conds.append("reviewed=0")
        if conds:
            q += " WHERE " + " AND ".join(conds)
        q += " ORDER BY id DESC"
    else:
        # Patient login — filter by patient_id, Finalized reports only
        q = (
            "SELECT id,doctor_id,patient_id,patient_name,filename,findings,"
            "report,date,doctor_notes,reviewed,edited_findings,edited_report,"
            "COALESCE(scan_status,'AI Generated')"
            " FROM scans WHERE patient_id=?"
            " AND (scan_status='Finalized' OR (scan_status IS NULL AND reviewed=1))"
            " ORDER BY id DESC"
        )
        params = [upid]
        f_sev = []

    rows = db.execute(q, params).fetchall()
    if not rows:
        st.info("No records found.")
    else:
        for (
            scan_id,
            doc_id,
            pid,
            pname,
            fname,
            fjson,
            rep,
            date,
            dnotes,
            reviewed,
            edited_f,
            edited_r,
            scan_status_val,
        ) in rows:
            findings = json.loads(fjson) if fjson else []
            active_find = json.loads(edited_f) if edited_f else findings
            active_report = edited_r if edited_r else rep

            if f_sev:
                sevs = {f.get("severity") for f in active_find}
                if not (sevs & set(f_sev)):
                    continue

            top_sev = active_find[0]["severity"] if active_find else "NONE"
            _sicon = {"Finalized": "✅", "Draft Review": "📝", "AI Generated": "🤖"}
            rev_icon = _sicon.get(scan_status_val, "⏳") + f" {scan_status_val}"
            icon_map = {"CRITICAL": "🚨", "HIGH": "⚠️"}
            h_icon = icon_map.get(top_sev, "📋")

            # ── Scan History row — column layout (no text overlap) ──
            with st.container():
                _c1, _c2, _c3, _c4 = st.columns([3, 2, 2, 1])
                _c1.markdown(f"{h_icon} **{pname}**")
                _c2.markdown(f"`{pid}`")
                _c3.markdown(f"🗓 {date[:16]}")
                _c4.markdown(f"👨‍⚕️ {doc_id}")
            st.caption(rev_icon)
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("View details", expanded=False):
                tabs = st.tabs(["📸 Findings", "📝 Doctor Review", "🔁 Corrections"])

                # ---- TAB 1: Findings ----
                with tabs[0]:
                    ca, cb = st.columns(2)
                    with ca:
                        ip = os.path.join(UPLOAD_DIR, fname)
                        if os.path.exists(ip):
                            _bc1, _bc2, _bc3 = st.columns(3)
                            with _bc1:
                                bri = st.slider(
                                    "☀️ Brightness", -80, 80, 0, 5, key=f"br_{scan_id}"
                                )
                            with _bc2:
                                ctr = st.slider(
                                    "🔆 Contrast",
                                    0.5,
                                    2.5,
                                    1.0,
                                    0.1,
                                    key=f"ct_{scan_id}",
                                )
                            with _bc3:
                                inv = st.checkbox("🔄 Invert", key=f"inv_{scan_id}")
                            adj = adjust_image(
                                np.array(Image.open(ip).convert("RGB")), bri, ctr, inv
                            )
                            st.image(adj, use_container_width=True)
                    with cb:
                        for f in active_find:
                            sev = f.get("severity", "LOW")
                            st.markdown(
                                f"""
                            <div class='finding-row'>
                                <span>{f["label"]}</span>
                                <span style='color:#94a3b8'>{f["conf"]:.0%}</span>
                                <span class='badge badge-{sev}'>{sev}</span>
                            </div>""",
                                unsafe_allow_html=True,
                            )
                        if not active_find:
                            st.success("No findings")
                        st.info(active_report)
                        if dnotes:
                            st.caption(f"Notes: {dnotes}")
                        pdf = make_pdf(
                            pname,
                            pid,
                            doc_id,
                            active_find,
                            active_report,
                            date[:16],
                            dnotes or "",
                            edited=bool(edited_r),
                        )
                        st.download_button(
                            "📥 PDF",
                            pdf,
                            file_name=f"{pid}_{date[:10]}.pdf",
                            mime="application/pdf",
                            key=f"dl_{scan_id}",
                        )

                # ---- TAB 2: Doctor Review ----
                with tabs[1]:
                    if role == "Doctor":
                        st.markdown("#### Edit Report & Findings")
                        new_rep = st.text_area(
                            "Clinical impression",
                            value=active_report,
                            height=160,
                            key=f"rep_{scan_id}",
                        )
                        new_notes = st.text_area(
                            "Physician addendum",
                            value=dnotes or "",
                            height=70,
                            key=f"nt_{scan_id}",
                        )

                        st.markdown("**Findings editor:**")
                        state_key = f"edited_{scan_id}"
                        if state_key not in st.session_state:
                            st.session_state[state_key] = list(active_find)
                        edited = st.session_state[state_key]
                        to_rm = []
                        for i, f in enumerate(edited):
                            hc1, hc2, hc3, hc4 = st.columns([3, 1, 2, 1])
                            hc1.text(f["label"])
                            hc2.text(f"{f['conf']:.0%}")
                            with hc3:
                                ns = st.selectbox(
                                    "",
                                    ["CRITICAL", "HIGH", "MODERATE", "LOW"],
                                    index=["CRITICAL", "HIGH", "MODERATE", "LOW"].index(
                                        f["severity"]
                                    ),
                                    key=f"sev_{scan_id}_{i}",
                                )
                                edited[i]["severity"] = ns
                            with hc4:
                                if st.button("✖", key=f"rm_{scan_id}_{i}"):
                                    st.session_state[state_key].pop(i)
                                    st.rerun()

                        ac1, ac2, ac3 = st.columns([3, 2, 1])
                        with ac1:
                            add_l = st.selectbox(
                                "Add finding", CLASS_NAMES, key=f"al_{scan_id}"
                            )
                        with ac2:
                            add_s = st.selectbox(
                                "Severity",
                                ["CRITICAL", "HIGH", "MODERATE", "LOW"],
                                key=f"as_{scan_id}",
                            )
                        with ac3:
                            st.markdown("<br>", unsafe_allow_html=True)
                            if st.button("➕", key=f"add_{scan_id}"):
                                edited.append(
                                    {
                                        "label": add_l,
                                        "conf": 1.0,
                                        "severity": add_s,
                                        "box": [],
                                    }
                                )
                                st.rerun()

                        # Save Draft / Approve Report buttons
                        _is_finalized = scan_status_val == "Finalized"
                        if _is_finalized:
                            st.success(
                                "🔒 This report has been finalized and is read-only."
                            )
                            _pdf = make_pdf(
                                pname,
                                pid,
                                doc_id,
                                active_find,
                                active_report,
                                date[:16],
                                dnotes or "",
                                edited=True,
                            )
                            st.download_button(
                                "📥 Download Finalized PDF",
                                _pdf,
                                file_name=f"{pid}_{date[:10]}_final.pdf",
                                mime="application/pdf",
                                key=f"fp_{scan_id}",
                            )
                        else:
                            _b1, _b2 = st.columns(2)
                            with _b1:
                                if st.button(
                                    "💾 Save Draft",
                                    key=f"sd_{scan_id}",
                                    use_container_width=True,
                                ):
                                    db.execute(
                                        "UPDATE scans SET reviewed=0,edited_findings=?,"
                                        "edited_report=?,doctor_notes=?,scan_status='Draft Review'"
                                        " WHERE id=?",
                                        (
                                            json.dumps(edited),
                                            new_rep,
                                            new_notes,
                                            scan_id,
                                        ),
                                    )
                                    db.commit()
                                    st.success("💾 Draft saved — still editable.")
                                    st.rerun()
                            with _b2:
                                if st.button(
                                    "✅ Approve Report",
                                    key=f"sv_{scan_id}",
                                    type="primary",
                                    use_container_width=True,
                                ):
                                    db.execute(
                                        "UPDATE scans SET reviewed=1,edited_findings=?,"
                                        "edited_report=?,doctor_notes=?,scan_status='Finalized'"
                                        " WHERE id=?",
                                        (
                                            json.dumps(edited),
                                            new_rep,
                                            new_notes,
                                            scan_id,
                                        ),
                                    )
                                    db.commit()
                                    # Auto-save corrections to feedback dataset
                                    _ip = os.path.join(UPLOAD_DIR, fname)
                                    if findings != edited:
                                        save_feedback(
                                            scan_id,
                                            uname,
                                            _ip,
                                            findings,
                                            edited,
                                            "Auto-saved on report approval",
                                        )
                                    st.success("✅ Report finalized and locked!")
                                    st.rerun()
                    else:
                        st.info(
                            "Doctor review is restricted to physicians. Admins have view-only access."
                        )

                # ---- TAB 3: Corrections ----
                with tabs[2]:
                    if role == "Doctor":
                        st.markdown("#### Submit Correction for Retraining")
                        st.markdown(
                            """
                        <div class='glass-card' style='border-left:3px solid #4f8ef7'>
                            Corrected findings are saved as YOLO-format training data.
                            Collect enough corrections, then trigger retraining in <b>AI Training</b>.
                        </div>""",
                            unsafe_allow_html=True,
                        )
                        cn = st.text_area(
                            "Describe corrections",
                            placeholder="e.g. Removed false-positive Pneumothorax…",
                            key=f"cn_{scan_id}",
                            height=70,
                        )
                        use_ed = st.checkbox(
                            "Use edited findings as corrected labels",
                            value=True,
                            key=f"ue_{scan_id}",
                        )
                        if st.button(
                            "📤 Submit to Dataset", key=f"fb_{scan_id}", type="primary"
                        ):
                            corr = (
                                json.loads(edited_f)
                                if (use_ed and edited_f)
                                else findings
                            )
                            ip = os.path.join(UPLOAD_DIR, fname)
                            save_feedback(scan_id, uname, ip, findings, corr, cn)
                            db.execute(
                                "UPDATE scans SET reviewed=1 WHERE id=?", (scan_id,)
                            )
                            db.commit()
                            t, u_ = feedback_stats()
                            st.success(
                                f"✅ Submitted! Dataset: {t} total, {u_} unused."
                            )
                    else:
                        st.info(
                            "Correction submission is restricted to physicians. Admins have view-only access."
                        )

            st.divider()

# ================================================================
# PAGE — PATIENT PROFILES
# ================================================================
elif nav == "👥 Patient Profiles":
    # ── Permission guard ──────────────────────────────────────────
    if role not in ("Doctor", "Admin"):
        st.error(
            "🚫 Access denied. Patient profiles are restricted to Doctors and Admins."
        )
        st.stop()

    st.markdown("# 👥 Patient Profile System")
    t1, t2 = st.tabs(["➕ Register Patient", "📋 All Patients"])

    with t1:
        st.markdown("### Register New Patient")
        c1, c2 = st.columns(2)
        with c1:
            auto_id = next_patient_id()
            pid_in = st.text_input(
                "Patient ID", value=auto_id, help="Auto-generated. You may edit it."
            )
            pname = st.text_input("Full Name*")
            dob = st.date_input(
                "Date of Birth",
                value=datetime.date(1990, 1, 1),
                min_value=datetime.date(1900, 1, 1),
                max_value=datetime.date.today(),
            )
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        with c2:
            blood = st.selectbox(
                "Blood Group",
                ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-", "Unknown"],
            )
            phone = st.text_input("Phone")
            addr = st.text_area("Address", height=68)
            hist = st.text_area(
                "Medical History",
                height=68,
                placeholder="Hypertension, Diabetes, Smoker…",
            )

        st.markdown("---")
        st.markdown("**Create Login Account for Patient (optional)**")
        create_login = st.checkbox(
            "Allow this patient to log in and view their reports"
        )
        p_uname = p_pass = ""
        if create_login:
            lc1, lc2 = st.columns(2)
            with lc1:
                p_uname = st.text_input("Patient username")
            with lc2:
                p_pass = st.text_input("Patient password", type="password")

        if st.button("💾 Save Patient", type="primary"):
            if not pname.strip():
                st.error("Patient name is required.")
            else:
                try:
                    db.execute(
                        "INSERT OR REPLACE INTO patients"
                        "(patient_id,username,full_name,dob,gender,blood_group,phone,address,medical_history,created_at)"
                        " VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (
                            pid_in,
                            p_uname or None,
                            pname,
                            str(dob),
                            gender,
                            blood,
                            phone,
                            addr,
                            hist,
                            str(datetime.datetime.now()),
                        ),
                    )
                    if create_login and p_uname and p_pass:
                        db.execute(
                            "INSERT OR IGNORE INTO users"
                            "(username,password,role,patient_id,full_name,created_at)"
                            " VALUES(?,?,?,?,?,?)",
                            (
                                p_uname,
                                _hp(p_pass),
                                "Patient",
                                pid_in,
                                pname,
                                str(datetime.datetime.now()),
                            ),
                        )
                    db.commit()
                    st.success(f"✅ Patient {pname} registered with ID: {pid_in}")
                except Exception as e:
                    st.error(f"Error: {e}")

    with t2:
        all_pts = db.execute(
            "SELECT patient_id,username,full_name,dob,gender,blood_group,phone,medical_history,created_at"
            " FROM patients ORDER BY full_name"
        ).fetchall()
        srch = st.text_input("🔍 Search")

        if not all_pts:
            st.info("No patients registered yet.")

        for pid, puname, pname, dob, gender, blood, phone, hist, created in all_pts:
            if srch and srch.lower() not in pname.lower():
                continue
            sc = db.execute(
                "SELECT COUNT(*) FROM scans WHERE patient_id=?", (pid,)
            ).fetchone()[0]

            # ── Patient card: summary row with columns ─────────────
            with st.container():
                hc1, hc2, hc3 = st.columns([3, 2, 1])
                hc1.markdown(f"**👤 {pname}**")
                hc2.markdown(f"`{pid}`")
                hc3.markdown(f"🗂 **{sc}** scans")

            with st.expander("View / Edit patient details", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**DOB:** {dob}")
                    st.markdown(
                        f"**Gender:** {gender} &nbsp;|&nbsp; **Blood:** {blood}"
                    )
                    st.markdown(f"**Phone:** {phone or '—'}")
                    st.markdown(f"**Login:** {puname or '—'}")
                    st.markdown(f"**Registered:** {created[:10] if created else '—'}")
                with c2:
                    st.markdown("**Medical History:**")
                    st.info(hist or "None recorded.")
                last = db.execute(
                    "SELECT date,findings FROM scans WHERE patient_id=? ORDER BY id DESC LIMIT 5",
                    (pid,),
                ).fetchall()
                if last:
                    st.markdown("**Recent scans:**")
                    for sd, sf in last:
                        fl = json.loads(sf) if sf else []
                        top = fl[0]["label"] if fl else "Clear"
                        extra = f" +{len(fl) - 1}" if len(fl) > 1 else ""
                        st.caption(f"• {sd[:16]} — {top}{extra}")

            st.markdown("<br>", unsafe_allow_html=True)

# ================================================================
# PAGE — SCAN COMPARISON
# ================================================================
elif nav == "🔄 Scan Comparison":
    # ── Permission guard ──────────────────────────────────────────
    if role != "Doctor":
        st.error("🚫 Access denied. Scan comparison is restricted to Doctors.")
        st.stop()

    st.markdown("# 🔄 Scan Comparison")

    pts_with_scans = db.execute(
        "SELECT DISTINCT patient_id, patient_name FROM scans ORDER BY patient_name"
    ).fetchall()
    if not pts_with_scans:
        st.info("No scans available yet.")
    else:
        sel_pt = st.selectbox(
            "Select Patient", [f"{p[1]} ({p[0]})" for p in pts_with_scans]
        )
        sel_pid_cmp = sel_pt.split("(")[1].rstrip(")")
        scans = db.execute(
            "SELECT id,date,filename,findings,report FROM scans WHERE patient_id=? ORDER BY id DESC",
            (sel_pid_cmp,),
        ).fetchall()

        if len(scans) < 2:
            st.warning("Need at least 2 scans for comparison.")
        else:
            labels = [f"Scan {i + 1}: {s[1][:16]}" for i, s in enumerate(scans)]
            c1, c2 = st.columns(2)
            with c1:
                i1 = st.selectbox(
                    "Scan A (earlier)",
                    range(len(scans)),
                    format_func=lambda i: labels[i],
                    index=min(1, len(scans) - 1),
                )
            with c2:
                i2 = st.selectbox(
                    "Scan B (later)",
                    range(len(scans)),
                    format_func=lambda i: labels[i],
                    index=0,
                )
            s1, s2 = scans[i1], scans[i2]
            f1 = json.loads(s1[3]) if s1[3] else []
            f2 = json.loads(s2[3]) if s2[3] else []

            im1, im2 = st.columns(2)
            for col, (scan, label, bkey, ckey) in zip(
                [im1, im2], [(s1, "Scan A", "cb1", "cc1"), (s2, "Scan B", "cb2", "cc2")]
            ):
                with col:
                    st.markdown(f"**{scan[1][:16]} ({label})**")
                    ip = os.path.join(UPLOAD_DIR, scan[2])
                    if os.path.exists(ip):
                        bv = st.slider("☀️", -80, 80, 0, 5, key=bkey)
                        cv = st.slider("🔆", 0.5, 2.5, 1.0, 0.1, key=ckey)
                        arr = adjust_image(
                            np.array(Image.open(ip).convert("RGB")), bv, cv
                        )
                        st.image(arr, use_container_width=True)

            st.markdown("### 📊 Findings Diff")
            lb1, lb2 = {f["label"] for f in f1}, {f["label"] for f in f2}
            new_f = lb2 - lb1
            resolved = lb1 - lb2
            same = lb1 & lb2
            dc1, dc2, dc3 = st.columns(3)
            with dc1:
                st.markdown("**🆕 New in Scan B**")
                for l in new_f:
                    fd = next((f for f in f2 if f["label"] == l), {})
                    sev = fd.get("severity", "LOW")
                    st.markdown(
                        f"<span class='badge badge-{sev}'>{l}</span>",
                        unsafe_allow_html=True,
                    )
                if not new_f:
                    st.caption("None")
            with dc2:
                st.markdown("**✅ Resolved**")
                for l in resolved:
                    st.markdown(
                        f"<span style='color:#00c864'>↓ {l}</span>",
                        unsafe_allow_html=True,
                    )
                if not resolved:
                    st.caption("None")
            with dc3:
                st.markdown("**⏺ Unchanged**")
                for l in same:
                    st.markdown(
                        f"<span style='color:#94a3b8'>= {l}</span>",
                        unsafe_allow_html=True,
                    )
                if not same:
                    st.caption("None")

            if same:
                st.markdown("### 📈 Confidence Trend")
                rows_t = []
                for l in same:
                    ca_ = next((f["conf"] for f in f1 if f["label"] == l), 0)
                    cb_ = next((f["conf"] for f in f2 if f["label"] == l), 0)
                    rows_t.append({"Finding": l, "Scan A": ca_, "Scan B": cb_})
                df_t = pd.DataFrame(rows_t)
                fig = go.Figure(
                    [
                        go.Bar(
                            name="Scan A",
                            x=df_t["Finding"],
                            y=df_t["Scan A"],
                            marker_color="#4f8ef7",
                        ),
                        go.Bar(
                            name="Scan B",
                            x=df_t["Finding"],
                            y=df_t["Scan B"],
                            marker_color="#00c864",
                        ),
                    ]
                )
                fig.update_layout(
                    barmode="group",
                    paper_bgcolor="#111827",
                    plot_bgcolor="#111827",
                    font_color="white",
                    height=300,
                )
                st.plotly_chart(fig, use_container_width=True)

            if new_f & {"Pneumothorax", "Cardiomegaly", "Pleural effusion"}:
                st.markdown(
                    "<div class='alert-critical'>🚨 <b>Progression detected</b> — New critical findings in Scan B</div>",
                    unsafe_allow_html=True,
                )
            elif resolved:
                st.markdown(
                    "<div style='background:#002d1a;border:1px solid #00c864;border-radius:10px;padding:12px 16px'>✅ <b>Improvement observed</b></div>",
                    unsafe_allow_html=True,
                )

# ================================================================
# PAGE — DASHBOARD
# ================================================================
elif nav == "📊 Dashboard":
    # ── Permission guard ──────────────────────────────────────────
    if role not in ("Doctor", "Admin"):
        st.error("🚫 Access denied. Dashboard is restricted to Doctors and Admins.")
        st.stop()

    st.markdown("# 📊 Analytics Dashboard")
    rows = db.execute(
        "SELECT patient_id,patient_name,findings,date,reviewed FROM scans ORDER BY id DESC"
    ).fetchall()
    if not rows:
        st.info("No data yet.")
    else:
        all_f = []
        for _, _, fj, _, _ in rows:
            try:
                all_f.extend(json.loads(fj))
            except:
                pass

        rev_count = sum(1 for r in rows if r[4])
        c1, c2, c3, c4, c5 = st.columns(5)
        for col, val, label, color in [
            (c1, len(rows), "Total Scans", "#4f8ef7"),
            (c2, len({r[0] for r in rows}), "Patients", "#4f8ef7"),
            (c3, len(all_f), "Findings", "#4f8ef7"),
            (
                c4,
                sum(1 for f in all_f if f.get("severity") == "CRITICAL"),
                "Critical",
                "#dc143c",
            ),
            (c5, rev_count, "Reviewed", "#00c864"),
        ]:
            col.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-value' style='color:{color}'>{val}</div>"
                f"<div class='metric-label'>{label}</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            cnt = {}
            for f in all_f:
                cnt[f.get("label", "?")] = cnt.get(f.get("label", "?"), 0) + 1
            df = pd.DataFrame(cnt.items(), columns=["Finding", "Count"]).sort_values(
                "Count", ascending=False
            )
            fig = px.bar(
                df,
                x="Count",
                y="Finding",
                orientation="h",
                color="Count",
                color_continuous_scale="Blues",
                title="Finding Frequency",
            )
            fig.update_layout(
                paper_bgcolor="#111827",
                plot_bgcolor="#111827",
                font_color="white",
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            sc = {"CRITICAL": 0, "HIGH": 0, "MODERATE": 0, "LOW": 0}
            for f in all_f:
                sc[f.get("severity", "LOW")] += 1
            fig2 = go.Figure(
                go.Pie(
                    labels=list(sc.keys()),
                    values=list(sc.values()),
                    marker_colors=["#dc143c", "#ff6400", "#ddb200", "#00c864"],
                    hole=0.45,
                )
            )
            fig2.update_layout(
                title="Severity Distribution",
                paper_bgcolor="#111827",
                font_color="white",
                height=420,
            )
            st.plotly_chart(fig2, use_container_width=True)

        if len(rows) > 3:
            df_t = pd.DataFrame([(r[3][:10],) for r in rows if r[3]], columns=["date"])
            df_t["date"] = pd.to_datetime(df_t["date"])
            dg = df_t.groupby("date").size().reset_index(name="count")
            fig3 = px.line(
                dg,
                x="date",
                y="count",
                title="Scan Volume Over Time",
                markers=True,
                color_discrete_sequence=["#4f8ef7"],
            )
            fig3.update_layout(
                paper_bgcolor="#111827",
                plot_bgcolor="#111827",
                font_color="white",
                height=280,
            )
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown("### Recent Scans")
        tbl = []
        for pid, pname, fj, date, rev in rows[:25]:
            try:
                fl = json.loads(fj)
                top = fl[0]["label"] if fl else "Clear"
                sev = fl[0]["severity"] if fl else "—"
            except:
                top, sev = "—", "—"
            tbl.append(
                {
                    "Patient ID": pid,
                    "Name": pname,
                    "Date": date[:16],
                    "Top Finding": top,
                    "Severity": sev,
                    "Reviewed": "✅" if rev else "⏳",
                }
            )
        st.dataframe(pd.DataFrame(tbl), use_container_width=True, height=380)

# ================================================================
# PAGE — AI TRAINING
# ================================================================
elif nav == "🧠 AI Training":
    # ── Permission guard ──────────────────────────────────────────
    if role != "Admin":
        st.error("🚫 Access denied. AI Training is restricted to Admins.")
        st.stop()

    st.markdown("# 🧠 AI Training & Model Management")

    total_fb, unused_fb = feedback_stats()
    all_models = db.execute(
        "SELECT version,path,training_date,dataset_size,notes,active FROM models ORDER BY id DESC"
    ).fetchall()

    mc1, mc2, mc3, mc4 = st.columns(4)
    for col, val, label, color in [
        (mc1, total_fb, "Corrections", "#4f8ef7"),
        (mc2, unused_fb, "Ready for Training", "#ddb200"),
        (mc3, len(all_models), "Model Versions", "#4f8ef7"),
        (mc4, get_model_info()["version"], "Active Model", "#00c864"),
    ]:
        col.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-value' style='color:{color};font-size:1.6rem'>{val}</div>"
            f"<div class='metric-label'>{label}</div></div>",
            unsafe_allow_html=True,
        )

    t1, t2, t3 = st.tabs(
        ["🚀 Trigger Retraining", "📦 Model Versions", "📊 Correction Analytics"]
    )

    with t1:
        st.markdown(
            """
        <div class='glass-card'>
            <b>Human-in-the-Loop Pipeline</b><br><br>
            Doctor reviews AI findings → submits corrections →
            YOLO-format labels are saved → you trigger retraining here →
            new model is registered → you review & activate it.<br><br>
            <span style='color:#64748b'>The model is never auto-deployed — you always approve it first.</span>
        </div>""",
            unsafe_allow_html=True,
        )

        lc, rc = st.columns(2)
        with lc:
            new_ver = st.text_input(
                "New version name", value=f"v{len(all_models) + 2}-finetuned"
            )
            epochs_in = st.slider("Epochs", 5, 100, 20)
            min_samp = st.slider("Min corrections to unlock training", 5, 200, 20)
            train_note = st.text_area(
                "Training notes", height=70, placeholder="What changed, why retrained…"
            )
        with rc:
            st.markdown("**Recent corrections:**")
            fb_rows = db.execute(
                "SELECT doctor_id,timestamp,correction_notes FROM feedback ORDER BY id DESC LIMIT 8"
            ).fetchall()
            for dr, ts, cn in fb_rows:
                st.caption(f"• Dr.{dr} · {ts[:16]}  — {(cn or 'no notes')[:55]}")

        if unused_fb < min_samp:
            st.warning(
                f"⚠️ {unused_fb}/{min_samp} corrections ready. Submit more from Scan History."
            )
        else:
            st.success(f"✅ {unused_fb} corrections available — ready to train.")

        bc1, bc2 = st.columns(2)
        with bc1:
            if st.button(
                "🚀 Start Retraining",
                type="primary",
                use_container_width=True,
                disabled=(unused_fb < min_samp or _training_status["running"]),
            ):
                ok = trigger_training(new_ver, epochs_in, train_note)
                if ok:
                    st.success("🎉 Retraining started in background thread!")
                    st.info(
                        "Monitor progress below. Activate model after training completes."
                    )
                else:
                    st.warning("Training already running.")
        with bc2:
            if st.button("📤 Export Dataset Info", use_container_width=True):
                info = {
                    "images": len(os.listdir(FEEDBACK_IMG)),
                    "labels": len(os.listdir(FEEDBACK_LBL)),
                    "corrections": total_fb,
                    "class_names": CLASS_NAMES,
                    "yaml_path": os.path.abspath("feedback_dataset/data.yaml"),
                    "exported": str(datetime.datetime.now()),
                }
                st.json(info)
                st.download_button(
                    "💾 Download JSON",
                    json.dumps(info, indent=2),
                    "dataset_info.json",
                    "application/json",
                )

        # Live training log
        if _training_status["log"]:
            st.markdown("**Training Log:**")
            st.markdown(
                f"<div class='log-box'>{_training_status['log']}</div>",
                unsafe_allow_html=True,
            )
        elif _training_status["running"]:
            st.info("Training running… refresh page to see log updates.")

    with t2:
        st.markdown("### Model Version Control")
        # Always show base model
        st.markdown(
            f"""
        <div class='glass-card' style='border-left:4px solid {"#00c864" if not all_models or not any(m[5] for m in all_models) else "#4f6080"}'>
            <b>v1-base</b> — Original YOLOv12 trained on VinBigdata<br>
            <span style='color:#64748b'>Path: {MODEL_PATH}</span>
        </div>""",
            unsafe_allow_html=True,
        )

        for ver, path, tdate, dsize, notes, active in all_models:
            status = "🟢 ACTIVE" if active else "⚫ Inactive"
            with st.expander(f"{status}  {ver}  —  {(tdate or '')[:16]}"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Version:** {ver}")
                    st.markdown(f"**Trained:** {(tdate or '—')[:16]}")
                    st.markdown(f"**Dataset size:** {dsize}")
                    st.markdown(f"**Notes:** {notes or '—'}")
                with c2:
                    exists = path and os.path.exists(path)
                    st.markdown(
                        f"**Weights:** {'✅ Found' if exists else '⏳ Not found'}"
                    )
                    if not active and exists:
                        if st.button(
                            f"✅ Activate {ver}", key=f"act_{ver}", type="primary"
                        ):
                            db.execute("UPDATE models SET active=0")
                            db.execute(
                                "UPDATE models SET active=1 WHERE version=?", (ver,)
                            )
                            db.commit()
                            # Reload model in session (fix #9)
                            st.session_state["model_path"] = path
                            st.cache_resource.clear()
                            st.success(f"✅ {ver} activated and model reloaded!")
                            st.rerun()
                    if active and not (path == st.session_state["model_path"]):
                        if st.button("🔄 Reload Weights", key=f"rel_{ver}"):
                            st.session_state["model_path"] = path
                            st.cache_resource.clear()
                            st.rerun()
                log_f = os.path.join(MODEL_DIR, f"log_{ver}.txt")
                if os.path.exists(log_f):
                    with open(log_f) as lf:
                        log_txt = lf.read()
                    st.markdown("**Training Log:**")
                    st.markdown(
                        f"<div class='log-box'>{log_txt[-2500:]}</div>",
                        unsafe_allow_html=True,
                    )

    with t3:
        st.markdown("### Correction Analytics")
        fb_all = db.execute(
            "SELECT doctor_id,timestamp,original_findings,corrected_findings FROM feedback ORDER BY id DESC"
        ).fetchall()
        if not fb_all:
            st.info("No corrections yet.")
        else:
            fp_removed = []
            for _, _, ofj, cfj in fb_all:
                try:
                    orig = {f["label"] for f in json.loads(ofj)}
                    corr = {f["label"] for f in json.loads(cfj)}
                    fp_removed.extend(orig - corr)
                except:
                    pass
            if fp_removed:
                cnt = {}
                for l in fp_removed:
                    cnt[l] = cnt.get(l, 0) + 1
                df_fp = pd.DataFrame(
                    cnt.items(), columns=["Finding", "False Positives Removed"]
                )
                df_fp = df_fp.sort_values("False Positives Removed", ascending=False)
                fig_fp = px.bar(
                    df_fp,
                    x="False Positives Removed",
                    y="Finding",
                    orientation="h",
                    title="Most Common AI Errors (Removed by Doctors)",
                    color="False Positives Removed",
                    color_continuous_scale="Reds",
                )
                fig_fp.update_layout(
                    paper_bgcolor="#111827",
                    plot_bgcolor="#111827",
                    font_color="white",
                    height=350,
                )
                st.plotly_chart(fig_fp, use_container_width=True)

            doc_cnt = {}
            for dr, _, _, _ in fb_all:
                doc_cnt[dr] = doc_cnt.get(dr, 0) + 1
            df_doc = pd.DataFrame(
                doc_cnt.items(), columns=["Doctor", "Corrections"]
            ).sort_values("Corrections", ascending=False)
            st.markdown("**Contributions by Doctor:**")
            st.dataframe(df_doc, use_container_width=True)

# ================================================================
# PAGE — MANAGE USERS  (Admin only)
# ================================================================
elif nav == "👤 Manage Users":
    if role != "Admin":
        st.error("🚫 Access denied. User management is restricted to Admins.")
        st.stop()

    st.markdown("# 👤 User Management")
    st.markdown("<br>", unsafe_allow_html=True)

    all_users = db.execute(
        "SELECT id, username, full_name, role, specialization, hospital,"
        " COALESCE(status,'approved'), created_at FROM users ORDER BY role, username"
    ).fetchall()

    # ── Pending Doctors ───────────────────────────────────────────────
    pending = [u for u in all_users if u[3] == "Doctor" and u[6] == "pending"]
    st.markdown("### ⏳ Pending Doctor Approvals")
    if not pending:
        st.success("✅ No pending approvals.")
    for uid, uname_, fname_, urole, spec, hosp, ustatus, ucreated in pending:
        with st.container():
            pc1, pc2, pc3, pc4 = st.columns([3, 2, 2, 1])
            pc1.markdown(f"👨‍⚕️ **{fname_ or uname_}** (`{uname_}`)")
            pc2.markdown(f"{spec or '—'}")
            pc3.markdown(f"{hosp or '—'}")
            with pc4:
                if st.button(
                    "✅ Approve",
                    key=f"app_{uid}",
                    type="primary",
                    use_container_width=True,
                ):
                    db.execute("UPDATE users SET status='approved' WHERE id=?", (uid,))
                    db.commit()
                    st.success(f"✅ {uname_} approved!")
                    st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("---")

    # ── All Users ───────────────────────────────────────────────────
    st.markdown("### 👥 All Users")
    _role_filter = st.selectbox("Filter by role", ["All", "Doctor", "Admin", "Patient"])
    for uid, uname_, fname_, urole, spec, hosp, ustatus, ucreated in all_users:
        if _role_filter != "All" and urole != _role_filter:
            continue
        _status_badge = (
            "🟢 Approved"
            if ustatus == "approved"
            else "🟡 Pending"
            if ustatus == "pending"
            else ustatus
        )
        with st.container():
            uc1, uc2, uc3, uc4, uc5 = st.columns([2, 2, 1, 1, 1])
            uc1.markdown(f"**{fname_ or uname_}** `{uname_}`")
            uc2.markdown(f"{spec or ''} {hosp or ''}")
            uc3.markdown(urole)
            uc4.markdown(_status_badge)
            with uc5:
                if uname_ != uname:  # cannot delete yourself
                    if st.button("🗑️", key=f"del_{uid}", help=f"Delete {uname_}"):
                        db.execute("DELETE FROM users WHERE id=?", (uid,))
                        db.commit()
                        st.warning(f"User {uname_} deleted.")
                        st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)
