import os
import json
from typing import Dict, Any
from flask import (
    Flask, render_template, request, redirect, url_for,
    send_from_directory, flash, jsonify, render_template_string
)
from detect_workflow import detect_by_part_id

# =========================
# App & Config Management
# =========================
app = Flask(__name__)
app.secret_key = "amropick-minimal-ui"  # change if you like

CONFIG_PATH = os.environ.get("AMROPICK_CONFIG", "config.yaml")

# Optional YAML support (fallback to JSON if PyYAML is not installed)
try:
    import yaml  # type: ignore
    _YAML_OK = True
except Exception:
    yaml = None  # type: ignore
    _YAML_OK = False


def _default_config() -> Dict[str, Any]:
    """Hard defaults (used if no file or keys missing)."""
    return {
        # Directory with object models; /api detects will default to model_dir/Plate{part_id}.ply if model_path isn't passed
        "model_dir": "object_models",

        # Whether to render Open3D overlay by default (HTML form can override)
        "render_overlay_default": False,

        # Pose output options (reserved for future use)
        "output_units": "deg",  # degrees for rx, ry, rz

        # ICP / alignment related knobs (your pipeline may or may not read these internally)
        "icp_threshold": 0.01,
        "max_icp_iterations": 50,
    }


def _load_config() -> Dict[str, Any]:
    """Load config from CONFIG_PATH; merge with defaults."""
    cfg = _default_config()
    if os.path.isfile(CONFIG_PATH):
        try:
            if _YAML_OK and CONFIG_PATH.lower().endswith((".yml", ".yaml")):
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    file_cfg = yaml.safe_load(f) or {}
            else:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    file_cfg = json.load(f)
            if isinstance(file_cfg, dict):
                cfg.update(file_cfg)
        except Exception as e:
            print(f"[CFG] Failed to load {CONFIG_PATH}: {e}", flush=True)
    return cfg


def _save_config(cfg: Dict[str, Any]) -> None:
    """Persist config to CONFIG_PATH, prefer YAML if possible."""
    try:
        if _YAML_OK and CONFIG_PATH.lower().endswith((".yml", ".yaml")):
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
        else:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
        print(f"[CFG] Saved configuration -> {CONFIG_PATH}", flush=True)
    except Exception as e:
        print(f"[CFG] Failed to save {CONFIG_PATH}: {e}", flush=True)


def _coerce_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "on"}


def _coerce_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _coerce_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _pose_dict(pose_tuple):
    tx, ty, tz, rx, ry, rz = pose_tuple
    return {
        "tx": tx, "ty": ty, "tz": tz,
        "rx": rx, "ry": ry, "rz": rz,
        "tuple": [tx, ty, tz, rx, ry, rz],
    }

# =========================================
# Web UI
# =========================================

@app.route("/", methods=["GET"])
def index():
    """Simple HTML form for manual testing."""
    cfg = _load_config()
    return render_template("index.html", cfg=cfg)


# ------- NEW: Configuration Page (UI) -------
_CONFIG_FORM_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>AMROPick Config</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }
    .wrap { max-width: 840px; margin: 0 auto; }
    h1 { margin-bottom: 0.5rem; }
    .sub { color:#666; margin-bottom: 1.5rem; }
    form { display: grid; grid-template-columns: 1fr 2fr; gap: 12px 16px; align-items: center; }
    label { font-weight: 600; }
    input[type="text"], input[type="number"] { padding: 8px; border: 1px solid #ccc; border-radius: 6px; width: 100%; }
    input[type="checkbox"] { transform: scale(1.2); }
    .row { grid-column: 1 / -1; margin-top: 8px; }
    .btns { display:flex; gap:12px; margin-top: 8px; }
    .btn { background:#111; color:#fff; border:none; padding:10px 14px; border-radius:8px; cursor:pointer; }
    .btn.secondary { background:#666; }
    .note { color:#555; font-size: 0.9rem; }
    .flash { background:#eef9f0; border:1px solid #bde5c8; color:#256a3f; padding:10px 12px; border-radius:8px; margin-bottom:16px; }
    a { color:#0b57d0; text-decoration:none; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>AMROPick – Configuration</h1>
    <div class="sub">Editing: <code>{{ config_path }}</code></div>

    {% with messages = get_flashed_messages() %}
      {% if messages %}
        {% for m in messages %}
          <div class="flash">{{ m }}</div>
        {% endfor %}
      {% endif %}
    {% endwith %}

    <form method="post" action="{{ url_for('config_page') }}">
      <label for="model_dir">Model directory</label>
      <input id="model_dir" name="model_dir" type="text" value="{{ cfg.model_dir }}">

      <label for="render_overlay_default">Render overlay by default</label>
      <input id="render_overlay_default" name="render_overlay_default" type="checkbox" {% if cfg.render_overlay_default %}checked{% endif %}>

      <label for="output_units">Output units (rx,ry,rz)</label>
      <input id="output_units" name="output_units" type="text" value="{{ cfg.output_units }}">

      <label for="icp_threshold">ICP threshold</label>
      <input id="icp_threshold" name="icp_threshold" type="number" step="0.0001" value="{{ cfg.icp_threshold }}">

      <label for="max_icp_iterations">Max ICP iterations</label>
      <input id="max_icp_iterations" name="max_icp_iterations" type="number" step="1" value="{{ cfg.max_icp_iterations }}">

      <div class="row btns">
        <button class="btn" type="submit">Save</button>
        <a class="btn secondary" href="{{ url_for('index') }}">← Back</a>
        <a class="btn secondary" href="{{ url_for('config_export') }}">Export JSON</a>
      </div>
    </form>

    <p class="note">You can also GET/POST <code>/api/config</code> to script changes.</p>
  </div>
</body>
</html>
"""

@app.route("/config", methods=["GET", "POST"])
def config_page():
    cfg = _load_config()
    if request.method == "POST":
        # Read and coerce values
        cfg["model_dir"] = (request.form.get("model_dir") or cfg["model_dir"]).strip()
        cfg["render_overlay_default"] = _coerce_bool(request.form.get("render_overlay_default", cfg["render_overlay_default"]))
        cfg["output_units"] = (request.form.get("output_units") or cfg["output_units"]).strip() or "deg"
        cfg["icp_threshold"] = _coerce_float(request.form.get("icp_threshold", cfg["icp_threshold"]), cfg["icp_threshold"])
        cfg["max_icp_iterations"] = _coerce_int(request.form.get("max_icp_iterations", cfg["max_icp_iterations"]), cfg["max_icp_iterations"])

        # Basic validation
        if cfg["icp_threshold"] <= 0:
            cfg["icp_threshold"] = 0.01
            flash("icp_threshold must be > 0; reset to 0.01")
        if cfg["max_icp_iterations"] < 1:
            cfg["max_icp_iterations"] = 50
            flash("max_icp_iterations must be >= 1; reset to 50")

        _save_config(cfg)
        flash("Configuration saved.")

        # Redirect-POST pattern
        return redirect(url_for("config_page"))

    # GET -> render simple inline page (no template file needed)
    return render_template_string(_CONFIG_FORM_HTML, cfg=cfg, config_path=os.path.abspath(CONFIG_PATH))


# Optional helper to export current config via browser
@app.route("/config/export", methods=["GET"])
def config_export():
    return jsonify(_load_config())


# =========================================
# Detection (HTML)
# =========================================

@app.route("/detect", methods=["POST"])
def detect():
    """HTML form submit -> runs detection and renders result page."""
    print("[WEB] /detect (HTML) start", flush=True)
    cfg = _load_config()

    part_str = (request.form.get("part_id") or "").strip()
    model_path = (request.form.get("model_path") or "").strip()
    render_overlay_form = request.form.get("render_overlay")
    render_overlay = _coerce_bool(render_overlay_form) if render_overlay_form is not None else bool(cfg.get("render_overlay_default", False))

    if not part_str.isdigit():
        flash("Part ID must be an integer (e.g., 1, 2, 3…).")
        return redirect(url_for("index"))

    part_id = int(part_str)

    # If empty model_path -> default to model_dir/Plate{part_id}.ply
    model_fs_path = model_path if model_path else os.path.join(cfg.get("model_dir", "object_models"), f"Plate{part_id}.ply")

    result = detect_by_part_id(
        part_id=part_id,
        model_fs_path=model_fs_path,
        render_overlay=render_overlay
    )

    if not isinstance(result, dict) or "error" in result:
        err = result.get("error") if isinstance(result, dict) else "Unknown error."
        flash(err)
        print(f"[WEB] /detect (HTML) error: {err}", flush=True)
        return redirect(url_for("index"))

    print("[WEB] /detect (HTML) done", flush=True)
    return render_template(
        "index.html",
        result=result,
        part_id=part_id,
        model_path=model_path,
        cfg=cfg
    )


# =========================================
# Detection (API)
# =========================================

@app.route("/api/detect", methods=["POST", "GET"])
def api_detect():
    """
    REST API to run detection for a wanted part by ID.

    Accepts:
      - JSON body: {"part_id": 3, "model_path": "object_models/Plate3.ply", "render_overlay": false}
        (model_path optional; if omitted -> {model_dir}/Plate{part_id}.ply)
      - OR query params (GET): /api/detect?part_id=3&model_path=...&render_overlay=1

    Returns JSON with poses and a URL to the preview image.
    """
    print("[WEB] /api/detect start", flush=True)
    cfg = _load_config()

    data = {}
    if request.method == "POST" and request.is_json:
        data = request.get_json(silent=True) or {}
    else:
        # allow GET with query params or POST with form data
        data = {**request.args, **request.form}

    part_raw = (str(data.get("part_id", "")).strip())
    if not part_raw.isdigit():
        return jsonify({"status": "error", "error": "part_id (int) is required"}), 400

    part_id = int(part_raw)
    model_path = str(data.get("model_path", "")).strip() or None

    if model_path is None:
        model_dir = cfg.get("model_dir", "object_models")
        model_path = os.path.join(model_dir, f"Plate{part_id}.ply")

    if "render_overlay" in data:
        render_overlay = _coerce_bool(data.get("render_overlay"))
    else:
        render_overlay = bool(cfg.get("render_overlay_default", False))

    # Run pipeline
    result = detect_by_part_id(
        part_id=part_id,
        model_fs_path=model_path,
        render_overlay=render_overlay
    )

    if not isinstance(result, dict) or "error" in result:
        err = result.get("error") if isinstance(result, dict) else "Unknown error."
        print(f"[WEB] /api/detect error: {err}", flush=True)
        return jsonify({"status": "error", "error": err}), 400

    overlay_url = url_for("files", filepath=result["overlay_png"], _external=True)

    resp = {
        "status": "ok",
        "part_id": part_id,
        "model_path": model_path,
        "pose_camera": _pose_dict(result["pose_camera"]),
        "pose_robot": _pose_dict(result["pose_robot"]),
        "overlay_png": result["overlay_png"],
        "overlay_url": overlay_url,
        "config_used": {
            "render_overlay_default": bool(cfg.get("render_overlay_default", False)),
            "model_dir": cfg.get("model_dir", "object_models"),
        }
    }

    print("[WEB] /api/detect done", flush=True)
    return jsonify(resp), 200


# =========================================
# Config (API)
# =========================================

@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    """GET returns current config; POST updates whitelisted fields."""
    if request.method == "GET":
        return jsonify(_load_config())

    # POST -> update
    data = request.get_json(silent=True) or {}
    cfg = _load_config()

    # Only allow known fields to be updated
    allowed = {"model_dir", "render_overlay_default", "output_units", "icp_threshold", "max_icp_iterations"}
    for k, v in data.items():
        if k not in allowed:
            continue
        if k == "render_overlay_default":
            cfg[k] = _coerce_bool(v)
        elif k == "icp_threshold":
            cfg[k] = _coerce_float(v, cfg["icp_threshold"])
        elif k == "max_icp_iterations":
            cfg[k] = _coerce_int(v, cfg["max_icp_iterations"])
        else:
            cfg[k] = v

    # Basic validation
    if cfg["icp_threshold"] <= 0:
        cfg["icp_threshold"] = 0.01
    if cfg["max_icp_iterations"] < 1:
        cfg["max_icp_iterations"] = 50

    _save_config(cfg)
    return jsonify({"status": "ok", "config": cfg})


# =========================================
# File server / health
# =========================================

@app.route("/files/<path:filepath>")
def files(filepath: str):
    """Serve files saved during detection (e.g., overlay PNG)."""
    directory = os.path.dirname(filepath) or "."
    filename = os.path.basename(filepath)
    return send_from_directory(directory, filename, as_attachment=False)


@app.route("/ping", methods=["GET"])
def ping():
    return "ok", 200


if __name__ == "__main__":
    # Run with: python app.py
    # Use debug=True for auto-reload during development.
    app.run(host="0.0.0.0", port=5000, debug=True)
