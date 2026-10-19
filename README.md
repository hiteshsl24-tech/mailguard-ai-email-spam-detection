# MailGuard AI — Email Spam Detection Website

A beginner-friendly Python data-mining project with a browser-based interface. It classifies pasted email text, supports supervised retraining from an uploaded CSV, and adds spam predictions to a batch CSV. It runs locally from VS Code and does not require Node.js.

> **SIH 2026 context:** The closest matching SIH 2026 reference is **SIH26106 — “AI-Powered Email Threat Detection, GeoLocation and Forensic Intelligence Platform”**, listed under Software / Blockchain & Cybersecurity. This project is a deliberately smaller educational prototype for the text-classification part of email threat screening. It is **not** a complete implementation of SIH26106 and must not be represented as one. The referenced problem statement also discusses header and protocol analysis, SPF/DKIM/DMARC, domain/IP intelligence, origin traceability, campaign correlation, and forensic reporting; those features are not implemented here.

## Features

- Browser-based dashboard built with Flask, HTML, and CSS.
- Single-message screening using an optional subject and a body.
- Email text features from TF-IDF (unigrams and bigrams) and a Multinomial Naive Bayes classifier.
- Simple, transparent text indicators for urgency, credential requests, prize/payment lures, URLs, and attachment wording. These are not proof that an email is malicious.
- Training from a CSV dataset with common column-name auto-detection or explicitly entered column names.
- Evaluation metrics: accuracy, spam precision, spam recall, F1-score, and a confusion matrix computed from a held-out split.
- Batch CSV analysis with a downloadable results CSV.
- Local model persistence; raw uploaded training CSVs and pasted messages are not stored by the app.
- Basic upload limits and input validation.

## Requirements

- Windows 10/11, macOS, or Linux.
- Python 3.10 or newer recommended.
- VS Code with the Python extension.
- Internet is only needed once to install Python packages. The app itself runs locally; the optional Google Fonts import in `static/style.css` falls back to system fonts when offline.

## Run it in VS Code (Windows PowerShell)

1. Download and extract the ZIP file.
2. Open VS Code, select **File → Open Folder**, and select the extracted `email-spam-detection` folder.
3. Open **Terminal → New Terminal**. Make sure the terminal is inside that project folder. You should see `app.py` and `requirements.txt` when you run `dir`.
4. Check that Python is available:

   ```powershell
   py --version
   ```

   If `py` is not recognized, try `python --version`. If neither command works, install Python from [python.org](https://www.python.org/downloads/) and enable **Add Python to PATH** in the installer.

5. Create a virtual environment:

   ```powershell
   py -m venv .venv
   ```

   If you used `python` in Step 4 instead, use `python -m venv .venv`.

6. Activate it:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use this one-time setting for the current terminal, then activate again:

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv\Scripts\Activate.ps1
   ```

7. Install the packages:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

8. Start the website:

   ```powershell
   python app.py
   ```

9. Open this address in your browser: **http://127.0.0.1:5000**
10. To stop the local web server, click the terminal and press `Ctrl+C`.

If you close VS Code and come back later, open the same folder, activate `.venv` again, and run `python app.py`.

## Dataset format

The training page accepts a CSV file with at least 20 non-empty, labeled rows and examples from both classes. Typical format:

```csv
label,text
spam,"Verify your account immediately using this link"
ham,"Can we move tomorrow's project meeting to 3 PM?"
```

Common text headers such as `text`, `message`, `email`, `body`, and `content` are detected. Common label headers such as `label`, `category`, `class`, `target`, and `type` are detected. If automatic detection fails, enter the exact column names in the form.

Accepted label examples include `spam` and `ham`, `spam` and `legitimate`, or `1` and `0` (where this project treats `1` as spam and `0` as ham). Unrecognized labels trigger a validation message; the app does not guess arbitrary classes.

`data/sample_dataset.csv` is a small **synthetic teaching/demo dataset** made for testing the workflow, not a validated benchmark. Use a larger, licensed, representative corpus for an academic evaluation. Keep the source, license, label definitions, class balance, and train/test methodology in your final report.

## How the data-mining model works

1. **Input validation:** remove empty messages and map known labels to two classes (`spam` / `ham`).
2. **Train/test split:** split examples into an 80% training subset and a 20% held-out test subset (stratified where possible, fixed random seed for reproducibility).
3. **Feature extraction:** TF-IDF represents word and two-word phrase importance numerically.
4. **Classification:** Multinomial Naive Bayes learns word-feature patterns associated with spam and legitimate messages.
5. **Evaluation:** metrics are calculated on the held-out subset. After evaluation, a fresh classifier is fitted on the full uploaded dataset for subsequent predictions.
6. **Persistence:** the trained model and summary metrics are stored at `instance/mailguard_model.joblib`. This is generated when the app runs and is excluded from version control.

### Metric interpretation

- **Accuracy:** fraction of test emails classified correctly.
- **Precision (spam):** of emails predicted as spam, the fraction that were spam.
- **Recall (spam):** of actual spam emails in the test set, the fraction detected by the model.
- **F1-score:** harmonic mean of spam precision and spam recall.
- **Confusion matrix:** counts correct and incorrect predictions for the two classes.

The model score/probability is not a calibrated security guarantee. Dataset leakage, repeated templates, small sample sizes, language differences, class imbalance, or changing attack patterns can make performance look better than it is. For a fair evaluation, keep a separate test set from a representative source and inspect false positives and false negatives.

## Project structure

```text
email-spam-detection/
├── app.py                  # Flask routes and website startup
├── ml_engine.py            # Dataset validation, training, prediction, indicators
├── requirements.txt        # Python dependencies
├── data/
│   └── sample_dataset.csv  # Synthetic starter data
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── train_result.html
│   └── batch.html
├── static/
│   ├── style.css
│   └── sample_dataset.csv  # Downloadable copy for the website
├── tests/
│   └── test_app.py
└── instance/               # Local trained model generated at run time
```

## Test the project

With the virtual environment active:

```powershell
python -m pytest -q
```

## Security and privacy limitations

- Do not upload confidential email dumps or messages containing sensitive personal information into a demo app unless you have authorization and an approved handling process.
- The app processes pasted text in server memory to classify it and does not intentionally log or persist the text. Use local-only deployment while learning.
- Uploaded training CSVs are not saved as raw files; the fitted model and its metrics are saved. Model files created by `joblib` must only be loaded from a trusted source.
- Batch CSV upload is limited to 10 MB and 50,000 rows.
- This prototype does not open URLs, download attachments, contact external threat-intelligence services, or access an email inbox.
- A legitimate-looking prediction does not establish that a message is safe. Do not use this prototype as an automatic blocker or as forensic evidence.

## SIH-style improvement roadmap

If the aim is to build toward SIH26106, add and test these as separate, real modules rather than claiming they are already present:

1. Parse raw `.eml` artifacts with the standard-library `email` package and preserve original headers safely.
2. Extract `Received`, `Return-Path`, `Reply-To`, `Message-ID`, and authentication results; clearly state when evidence is missing.
3. Add sender-domain similarity checks and safe URL parsing/defanging. Never fetch suspicious URLs from the analysis server by default.
4. Integrate SPF/DKIM/DMARC validation where technically justified, with transparent limitations.
5. Add IP/domain reputation and geolocation only using licensed/approved sources; display the confidence and origin of each result.
6. Add case IDs, analyst review, audit logs, data retention settings, and privacy masking before handling real institutional emails.
7. Build a campaign view to correlate repeated indicators across cases and produce a report with evidence provenance.
8. Evaluate on a representative, separate dataset; report per-class metrics, confusion matrix, false-positive/false-negative review, and limitations.

## Reference

- SIH 2026 problem statement archive page for [SIH26106 — AI-Powered Email Threat Detection, GeoLocation and Forensic Intelligence Platform](https://sih2026.vuce.in/ps/SIH26106). This is a community-maintained archive that links to the SIH portal; verify current official notices and requirements in the SIH portal before any formal submission.
