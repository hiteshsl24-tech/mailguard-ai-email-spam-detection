from io import BytesIO
from pathlib import Path
import pandas as pd

from app import create_app

BASE_DIR = Path(__file__).resolve().parents[1]


def make_test_app(tmp_path):
    frame = pd.read_csv(BASE_DIR / "data" / "sample_dataset.csv")
    from ml_engine import train_classifier
    model, metrics = train_classifier(frame, "text", "label")
    return create_app({
        "TESTING": True,
        "MODEL": model,
        "MODEL_INFO": {
            "metrics": metrics,
            "source": "test fixture",
            "trained_at": "test",
            "is_demo": True,
        },
    })


def test_dashboard_loads(tmp_path):
    client = make_test_app(tmp_path).test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"MailGuard" in response.data
    assert b"Single email analysis" in response.data


def test_predict_rejects_empty_message(tmp_path):
    client = make_test_app(tmp_path).test_client()
    response = client.post("/predict", data={"subject": "", "email_body": ""})
    assert response.status_code == 200
    assert b"Enter an email subject or body" in response.data


def test_predict_returns_classification(tmp_path):
    client = make_test_app(tmp_path).test_client()
    response = client.post("/predict", data={
        "subject": "Urgent: verify your account",
        "email_body": "Your account is suspended. Enter your password now at http://example.invalid"
    })
    assert response.status_code == 200
    assert b"CLASSIFICATION RESULT" in response.data
    assert b"Spam probability" in response.data


def test_batch_csv_download(tmp_path):
    client = make_test_app(tmp_path).test_client()
    csv_content = b"text\nPlease review the class notes.\nClaim your prize now at http://example.invalid\n"
    response = client.post("/batch", data={
        "dataset": (BytesIO(csv_content), "emails.csv"),
    }, content_type="multipart/form-data")
    assert response.status_code == 200
    assert "text/csv" in response.content_type
    assert b"mailguard_prediction" in response.data


def test_train_rejects_non_csv(tmp_path):
    client = make_test_app(tmp_path).test_client()
    response = client.post("/train", data={"dataset": (BytesIO(b"hello"), "file.txt")}, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"Only .csv files" in response.data
