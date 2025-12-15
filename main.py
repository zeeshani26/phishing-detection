import json
import logging
import os
import re
from email import policy
from email.parser import BytesParser
from io import BytesIO
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

import PyPDF2
import google.generativeai as genai
from docx import Document
from dotenv import load_dotenv
from flask import Flask, render_template, request
from google.api_core import exceptions as google_exceptions

# Initialize Flask app
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB upload limit to prevent abuse

logger = logging.getLogger(__name__)

# Load environment variables from .env if present
load_dotenv()

# Set up the Google API Key from environment
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise RuntimeError("GOOGLE_API_KEY environment variable is not set.")
genai.configure(api_key=google_api_key)

TEMPLATE_NAME = "index.html"
ALLOWED_EXTENSIONS = {"pdf", "txt", "docx", "eml"}

MODEL_INIT_ERROR = ""


def _fetch_available_models() -> Set[str]:
    """Return the set of model ids that support generateContent."""
    try:
        models = genai.list_models()
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(
            "Unable to list Gemini models via list_models(); continuing with fallbacks. Details: %s",
            exc,
        )
        return set()

    available: Set[str] = set()
    for entry in models:
        supported_methods = getattr(entry, "supported_generation_methods", []) or []
        if "generateContent" in supported_methods:
            name = getattr(entry, "name", "")
            if name:
                available.add(name.split("/")[-1])
    return available


def _resolve_model() -> genai.GenerativeModel:
    """Select the best available Gemini model, trying fallbacks if necessary."""
    preferred_model = os.getenv("GEMINI_MODEL")
    available_models = _fetch_available_models()

    candidate_models = [
        preferred_model,
        "gemini-2.0-flash-exp",
        "gemini-2.0-flash",
        "gemini-2.0-flash-001",
        "gemini-2.0-flash-lite-latest",
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash",
        "gemini-1.5-flash-001",
        "gemini-1.0-pro",
        "gemini-pro",
    ]
    errors: list[str] = []

    for candidate in filter(None, candidate_models):
        normalized = candidate.split("/")[-1]
        if available_models and normalized not in available_models:
            errors.append(f"{candidate}: not returned by list_models()")
            continue
        try:
            logger.info("Attempting to initialize Gemini model '%s'", candidate)
            candidate_model = genai.GenerativeModel(candidate)
            # Lightweight readiness check so we fail fast if the model isn't enabled.
            candidate_model.count_tokens("healthcheck")
            return candidate_model
        except google_exceptions.NotFound as exc:
            logger.warning("Gemini model '%s' not found: %s", candidate, exc)
            errors.append(f"{candidate}: not found")
        except google_exceptions.PermissionDenied as exc:
            logger.warning("Gemini model '%s' permission denied: %s", candidate, exc)
            errors.append(f"{candidate}: permission denied")
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Failed to initialize Gemini model '%s'", candidate, exc_info=exc)
            errors.append(f"{candidate}: {exc}")

    raise RuntimeError(
        "None of the configured Gemini models could be initialized. "
        "Tried -> " + ", ".join(errors)
    )


try:
    model = _resolve_model()
except RuntimeError as err:  # pylint: disable=broad-except
    model = None
    MODEL_INIT_ERROR = (
        "Gemini model is unavailable. Update your google-generativeai package "
        "(`pip install -U google-generativeai`), ensure your API key has access, "
        "or set GEMINI_MODEL to a supported model (e.g., gemini-1.0-pro). "
        f"Details: {err}"
    )
    logger.error("Gemini model initialization failed: %s", MODEL_INIT_ERROR)


def render_index(**context):
    """Render the homepage template, injecting any model availability warning."""
    if MODEL_INIT_ERROR:
        context.setdefault("model_error", MODEL_INIT_ERROR)
    return render_template(TEMPLATE_NAME, **context)


def _strip_code_fence(payload: str) -> str:
    cleaned = payload.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    return cleaned


def _extract_json_segment(cleaned: str) -> str:
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    return match.group(0) if match else cleaned


def _parse_model_json(raw_text: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw_text:
        return None
    cleaned = _strip_code_fence(raw_text)
    candidate = _extract_json_segment(cleaned)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        logger.debug("Failed to parse JSON from model response: %s", raw_text)
        return None
    return parsed if isinstance(parsed, dict) else None


def _coerce_score(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.replace("%", "").strip()
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    try:
        score_int = int(round(score))
    except OverflowError:
        return None
    return max(0, min(100, score_int))


def _ensure_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        parts = re.split(r"[;\n]+", value)
        return [part.strip() for part in parts if part.strip()]
    return []


def _build_email_analysis(raw_text: str) -> Optional[Dict[str, Any]]:
    parsed = _parse_model_json(raw_text)
    if not parsed:
        return None

    classification = str(
        parsed.get("classification")
        or parsed.get("verdict")
        or parsed.get("label")
        or ""
    ).strip()
    if not classification:
        return None

    risk_score = _coerce_score(parsed.get("risk_score") or parsed.get("risk"))
    confidence = _coerce_score(parsed.get("confidence"))
    summary = parsed.get("summary") or parsed.get("highlight") or parsed.get("rationale")
    indicators = (
        parsed.get("key_findings")
        or parsed.get("indicators")
        or parsed.get("signals")
    )
    recommendations = parsed.get("recommended_actions") or parsed.get("actions")

    return {
        "classification": classification.lower(),
        "display_classification": classification,
        "risk_score": risk_score,
        "confidence": confidence,
        "summary": summary.strip() if isinstance(summary, str) else "",
        "indicators": _ensure_list(indicators),
        "recommendations": _ensure_list(recommendations),
        "raw": raw_text,
    }


def _build_url_analysis(raw_text: str) -> Optional[Dict[str, Any]]:
    parsed = _parse_model_json(raw_text)
    if not parsed:
        return None

    classification = str(
        parsed.get("classification")
        or parsed.get("category")
        or parsed.get("label")
        or ""
    ).strip()
    if not classification:
        return None

    risk_score = _coerce_score(parsed.get("risk_score") or parsed.get("risk"))
    confidence = _coerce_score(parsed.get("confidence"))
    summary = parsed.get("verdict_reasoning") or parsed.get("summary") or parsed.get("explanation")
    indicators = parsed.get("signals") or parsed.get("indicators") or parsed.get("evidence")
    recommendations = parsed.get("recommended_actions") or parsed.get("actions")

    return {
        "classification": classification.lower(),
        "display_classification": classification,
        "risk_score": risk_score,
        "confidence": confidence,
        "summary": summary.strip() if isinstance(summary, str) else "",
        "indicators": _ensure_list(indicators),
        "recommendations": _ensure_list(recommendations),
        "raw": raw_text,
    }


def predict_fake_or_real_email_content(text: str) -> str:
    """Classify the supplied email text as real or scam using the Gemini model."""
    if not text.strip():
        return "Unable to classify empty text."

    if model is None:
        return MODEL_INIT_ERROR

    prompt = f"""
    You are an expert in fraud analysis. Evaluate the email or message content below and respond in JSON with this exact shape:
    {{
      "classification": "scam" | "legitimate" | "suspicious",
      "risk_score": 0-100 (integer, 100 = most risky),
      "confidence": 0-100 (integer confidence in your verdict),
      "summary": "one-sentence insight about why you chose this verdict",
      "key_findings": ["bullet highlighting evidence", "..."],
      "recommended_actions": ["next step for user", "..."]
    }}

    Requirements:
    - Always fill every field. If unsure, choose your best estimate.
    - The response must be valid JSON without extra commentary.

    Content to evaluate:
    ---
    {text}
    ---
    """

    try:
        response = model.generate_content(prompt)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Failed to classify email content: %s", exc)
        return "Classification failed due to an unexpected error."

    raw_text = response.text if response else ""
    analysis = _build_email_analysis(raw_text or "")
    if analysis:
        return analysis
    return raw_text.strip() if raw_text else "Classification failed."


def url_detection(url: str) -> str:
    """Classify the supplied URL into threat categories using the Gemini model."""
    if model is None:
        return MODEL_INIT_ERROR

    prompt = f"""
    You are an advanced URL threat analyst. Evaluate the URL below and respond in JSON with:
    {{
      "classification": "benign" | "phishing" | "malware" | "defacement" | "suspicious",
      "risk_score": 0-100,
      "confidence": 0-100,
      "verdict_reasoning": "one-sentence explanation",
      "signals": ["indicator 1", "indicator 2"],
      "recommended_actions": ["step 1", "step 2"]
    }}

    Rules:
    - Always produce valid JSON only.
    - Risk score should align with the confidence and classification (higher = more dangerous).

    URL to evaluate: {url}
    """

    try:
        response = model.generate_content(prompt)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Failed to classify URL: %s", exc)
        return "Detection failed due to an unexpected error."

    raw_text = response.text if response else ""
    analysis = _build_url_analysis(raw_text or "")
    if analysis:
        return analysis
    return raw_text.strip() if raw_text else "Detection failed."


def _strip_html(html: str) -> str:
    return re.sub(r"<[^>]+>", " ", html)


def _extract_eml_text(raw_bytes: bytes) -> str:
    message = BytesParser(policy=policy.default).parsebytes(raw_bytes)
    texts: List[str] = []

    for part in message.walk():
        content_type = part.get_content_type()
        if content_type == "text/plain":
            texts.append(part.get_content())
    if not texts:
        for part in message.walk():
            if part.get_content_type() == "text/html":
                texts.append(_strip_html(part.get_content()))
    flattened = "\n".join(texts)
    return flattened.strip()


def extract_text_from_upload(file_storage) -> tuple[str, str]:
    """Extract text content from an uploaded PDF or TXT file."""
    filename = file_storage.filename or ""
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    raw_bytes = file_storage.read()
    if hasattr(file_storage, "stream"):
        file_storage.stream.seek(0)
    else:
        file_storage.seek(0)

    if not raw_bytes:
        return "", "Uploaded file is empty."

    if extension == "pdf":
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(raw_bytes))
            extracted = " ".join(page.extract_text() or "" for page in pdf_reader.pages).strip()
            if extracted:
                return extracted, ""
            return "", "Unable to extract text from the PDF."
        except (PyPDF2.errors.PdfReadError, ValueError) as exc:
            logger.exception("Failed to read PDF: %s", exc)
            return "", "Unable to read the PDF file. Please try another document."
    if extension == "txt":
        try:
            extracted = raw_bytes.decode("utf-8", errors="ignore").strip()
            if extracted:
                return extracted, ""
            return "", "Unable to extract text from the text file."
        except UnicodeDecodeError as exc:
            logger.exception("Failed to decode TXT: %s", exc)
            return "", "Unable to decode the text file. Please ensure it is UTF-8 encoded."
    if extension == "docx":
        try:
            document = Document(BytesIO(raw_bytes))
            extracted = "\n".join(p.text for p in document.paragraphs if p.text).strip()
            if extracted:
                return extracted, ""
            return "", "Unable to extract text from the DOCX file."
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Failed to read DOCX: %s", exc)
            return "", "Unable to read the DOCX file. Please upload a standard Word document."
    if extension == "eml":
        try:
            extracted = _extract_eml_text(raw_bytes)
            if extracted:
                return extracted, ""
            return "", "Unable to extract text from the EML file."
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Failed to parse EML: %s", exc)
            return "", "Unable to parse the email file. Please try another message."

    return "", "Invalid file type. Please upload a PDF, TXT, DOCX, or EML file."


def is_supported_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def is_valid_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


# Routes
@app.route("/")
def home():
    return render_index()


@app.route("/scam/", methods=["POST"])
def detect_scam():
    uploaded_file = request.files.get("file")
    if not uploaded_file or not uploaded_file.filename:
        return render_index(file_message="No file uploaded.")

    if not is_supported_file(uploaded_file.filename):
        return render_index(
            file_message="Unsupported file type. Please upload PDF, TXT, DOCX, or EML files."
        )

    extracted_text, error_message = extract_text_from_upload(uploaded_file)
    if error_message:
        return render_index(file_message=error_message)

    message = predict_fake_or_real_email_content(extracted_text)
    if isinstance(message, dict):
        return render_index(file_analysis=message)
    return render_index(file_message=message)


@app.route("/predict", methods=["POST"])
def predict_url():
    url = request.form.get("url", "").strip()

    if not url:
        return render_index(url_message="Please provide a URL to classify.")

    if not is_valid_url(url):
        return render_index(
            url_message="Invalid URL format. Include http:// or https://", input_url=url
        )

    classification = url_detection(url)
    if isinstance(classification, dict):
        return render_index(input_url=url, url_analysis=classification)
    return render_index(input_url=url, url_message=classification)


if __name__ == "__main__":
    app.run(debug=True)
