from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO, StringIO
from pathlib import Path
import os

import joblib
import pandas as pd
from flask import Flask, render_template, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from ml_engine import infer_column, predict_email, train_classifier

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INSTANCE_DIR = BASE_DIR / "instance"
MODEL_PATH = INSTANCE_DIR / "mailguard_model.joblib"
ALLOWED_EXTENSIONS = {"csv"}
MAX_CSV_BYTES = 10 * 1024 * 1024


def allowed_csv(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_or_create_model(app: Flask) -> None:
    if MODEL_PATH.exists():
        try:
            stored = joblib.load(MODEL_PATH)
            app.config["MODEL"] = stored["model"]
            app.config["MODEL_INFO"] = stored["info"]
            return
        except Exception:
            # A corrupt or incompatible local model is safely replaced by the demo model.
            pass

    demo_path = DATA_DIR / "sample_dataset.csv"
    frame = pd.read_csv(demo_path)
    model, metrics = train_classifier(frame, text_column="text", label_column="label")
    info = {
        "metrics": metrics,
        "source": "Built-in demonstration dataset",
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "is_demo": True,
    }
    app.config["MODEL"] = model
    app.config["MODEL_INFO"] = info
    INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "info": info}, MODEL_PATH)


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.environ.get("MAILGUARD_SECRET_KEY", "local-development-only-change-me"),
        MAX_CONTENT_LENGTH=MAX_CSV_BYTES,
    )
    if test_config:
        app.config.update(test_config)
    INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
    if not (test_config and app.config.get("MODEL") is not None and app.config.get("MODEL_INFO") is not None):
        load_or_create_model(app)

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            model_info=app.config["MODEL_INFO"],
            result=None,
            error=None,
            subject="",
            email_body="",
            page_title="Dashboard",
        )

    @app.post("/predict")
    def predict():
        subject = request.form.get("subject", "").strip()
        email_body = request.form.get("email_body", "").strip()
        combined = "Subject: " + subject + "\n\n" + email_body if subject else email_body
        try:
            result = predict_email(app.config["MODEL"], combined)
            error = None
        except ValueError as exc:
            result, error = None, str(exc)
        return render_template(
            "index.html",
            model_info=app.config["MODEL_INFO"],
            result=result,
            error=error,
            subject=subject,
            email_body=email_body,
            page_title="Dashboard",
        )

    @app.post("/train")
    def train():
        uploaded = request.files.get("dataset")
        if not uploaded or not uploaded.filename:
            return render_template("index.html", model_info=app.config["MODEL_INFO"], result=None,
                                   error="Choose a CSV dataset to train the model.", subject="", email_body="",
                                   page_title="Dashboard")
        filename = secure_filename(uploaded.filename)
        if not allowed_csv(filename):
            error = "Only .csv files are supported for training."
        else:
            try:
                frame = pd.read_csv(uploaded, encoding="utf-8-sig", on_bad_lines="error")
                text_column = request.form.get("text_column", "").strip() or None
                label_column = request.form.get("label_column", "").strip() or None
                model, metrics = train_classifier(frame, text_column=text_column, label_column=label_column)
                info = {
                    "metrics": metrics,
                    "source": f"Uploaded dataset ({filename})",
                    "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                    "is_demo": False,
                }
                app.config["MODEL"] = model
                app.config["MODEL_INFO"] = info
                INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
                joblib.dump({"model": model, "info": info}, MODEL_PATH)
                return render_template("train_result.html", model_info=info, trained=True, error=None)
            except (ValueError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
                error = str(exc) if isinstance(exc, ValueError) else "The CSV could not be read. Save it as a UTF-8 CSV and try again."
        return render_template("index.html", model_info=app.config["MODEL_INFO"], result=None,
                               error=error, subject="", email_body="", page_title="Dashboard")

    @app.post("/batch")
    def batch():
        uploaded = request.files.get("dataset")
        if not uploaded or not uploaded.filename:
            return render_template("batch.html", error="Choose a CSV file to analyze.", preview=None)
        filename = secure_filename(uploaded.filename)
        if not allowed_csv(filename):
            return render_template("batch.html", error="Only .csv files are supported.", preview=None)
        try:
            frame = pd.read_csv(uploaded, encoding="utf-8-sig", on_bad_lines="error")
            requested_column = request.form.get("text_column", "").strip() or None
            text_column = requested_column or infer_column(frame.columns, ("text", "message", "email", "body", "content", "mail", "email_text"))
            if not text_column or text_column not in frame.columns:
                raise ValueError("Could not identify the email-text column. Add a 'text', 'message', 'email', or 'body' column, or enter its exact name.")
            if frame.empty:
                raise ValueError("The uploaded CSV contains no data rows.")
            if len(frame) > 50000:
                raise ValueError("Batch analysis is limited to 50,000 rows per upload.")
            texts = frame[text_column].fillna("").astype(str)
            predictions = [predict_email(app.config["MODEL"], item) if item.strip() else {
                "label": "not analyzed", "spam_probability": "", "confidence": "", "indicators": []
            } for item in texts]
            output = frame.copy()
            output["mailguard_prediction"] = [p["label"] for p in predictions]
            output["spam_probability_percent"] = [p["spam_probability"] for p in predictions]
            output["model_confidence_percent"] = [p["confidence"] for p in predictions]
            output["suspicious_indicators"] = ["; ".join(p["indicators"]) for p in predictions]
            buffer = StringIO()
            output.to_csv(buffer, index=False)
            data = BytesIO(buffer.getvalue().encode("utf-8-sig"))
            data.seek(0)
            output_name = f"mailguard_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            return send_file(data, mimetype="text/csv", as_attachment=True, download_name=output_name)
        except ValueError as exc:
            return render_template("batch.html", error=str(exc), preview=None)
        except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
            return render_template("batch.html", error="The CSV could not be read. Save it as a UTF-8 CSV and try again.", preview=None)

    @app.get("/batch")
    def batch_page():
        return render_template("batch.html", error=None, preview=None)

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        return render_template("batch.html", error="Upload too large. The maximum CSV size is 10 MB.", preview=None), 413

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": app.config.get("MODEL") is not None}

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
