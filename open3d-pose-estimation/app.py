# app.py
import os
from flask import (
    Flask, render_template, request, redirect, url_for,
    send_from_directory, flash
)
from detect_workflow import detect_by_part_id

# ---- Flask setup ----
app = Flask(__name__)
app.secret_key = "amropick-minimal-ui"  # change if you like


@app.route("/", methods=["GET"])
def index():
    """Simple form page: Part ID + optional model path."""
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    """Run detection/alignment for the given Part ID."""
    print("[WEB] /detect start", flush=True)

    # 1) Read form inputs
    part_str = (request.form.get("part_id") or "").strip()
    model_path = (request.form.get("model_path") or "").strip()

    if not part_str.isdigit():
        flash("Part ID must be an integer (e.g., 1, 2, 3...).")
        return redirect(url_for("index"))

    part_id = int(part_str)
    model_fs_path = model_path if model_path else None

    # 2) Call pipeline (render_overlay=False to avoid offscreen renderer hangs)
    result = detect_by_part_id(
        part_id=part_id,
        model_fs_path=model_fs_path,
        render_overlay=False
    )

    # 3) Handle result / errors
    if not isinstance(result, dict) or "error" in result:
        err = result.get("error") if isinstance(result, dict) else "Unknown error."
        flash(err)
        print(f"[WEB] /detect error: {err}", flush=True)
        return redirect(url_for("index"))

    print("[WEB] /detect done", flush=True)
    return render_template(
        "index.html",
        result=result,
        part_id=part_id,
        model_path=model_path
    )


@app.route("/files/<path:filepath>")
def files(filepath: str):
    """
    Serve files that detection saved (e.g., overlay PNG).
    We accept absolute or relative paths; send_from_directory needs dir + filename.
    """
    directory = os.path.dirname(filepath) or "."
    filename = os.path.basename(filepath)
    return send_from_directory(directory, filename, as_attachment=False)


# Optional: simple health check
@app.route("/ping")
def ping():
    return "ok", 200


if __name__ == "__main__":
    # Run with: python app.py
    # Set debug=True for auto-reload on code changes.
    app.run(host="0.0.0.0", port=5000, debug=True)
