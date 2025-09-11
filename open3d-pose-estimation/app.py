# app.py
import os
from flask import (
    Flask, render_template, request, redirect, url_for,
    send_from_directory, flash, jsonify
)
from detect_workflow import detect_by_part_id

app = Flask(__name__)
app.secret_key = "amropick-minimal-ui"  # change if you like


def _pose_dict(pose_tuple):
    tx, ty, tz, rx, ry, rz = pose_tuple
    return {
        "tx": tx, "ty": ty, "tz": tz,
        "rx": rx, "ry": ry, "rz": rz,
        "tuple": [tx, ty, tz, rx, ry, rz],
    }


@app.route("/", methods=["GET"])
def index():
    """Simple HTML form for manual testing."""
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    """HTML form submit -> runs detection and renders result page."""
    print("[WEB] /detect (HTML) start", flush=True)

    part_str = (request.form.get("part_id") or "").strip()
    model_path = (request.form.get("model_path") or "").strip()

    if not part_str.isdigit():
        flash("Part ID must be an integer (e.g., 1, 2, 3…).")
        return redirect(url_for("index"))

    part_id = int(part_str)
    model_fs_path = model_path if model_path else None

    result = detect_by_part_id(
        part_id=part_id,
        model_fs_path=model_fs_path,
        render_overlay=False  # faster & safer for headless; set True if you need Open3D overlay
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
        model_path=model_path
    )


@app.route("/api/detect", methods=["POST", "GET"])
def api_detect():
    """
    REST API to run detection for a wanted part by ID.

    Accepts:
      - JSON body: {"part_id": 3, "model_path": "object_models/Plate3.ply", "render_overlay": false}
        (model_path optional; if omitted -> object_models/Plate{part_id}.ply)
      - OR query params (GET): /api/detect?part_id=3&model_path=...&render_overlay=1

    Returns JSON with poses and a URL to the preview image.
    """
    print("[WEB] /api/detect start", flush=True)

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
    render_overlay = str(data.get("render_overlay", "0")).lower() in {"1", "true", "yes"}

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
        "model_path": model_path or f"object_models/Plate{part_id}.ply",
        "pose_camera": _pose_dict(result["pose_camera"]),
        "pose_robot": _pose_dict(result["pose_robot"]),
        "overlay_png": result["overlay_png"],
        "overlay_url": overlay_url,
    }

    print("[WEB] /api/detect done", flush=True)
    return jsonify(resp), 200


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
