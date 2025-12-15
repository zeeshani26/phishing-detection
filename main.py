import logging
import os
from io import BytesIO
from typing import Set
from urllib.parse import urlparse

import PyPDF2
import google.generativeai as genai
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
ALLOWED_EXTENSIONS = {"pdf", "txt"}

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


def predict_fake_or_real_email_content(text: str) -> str:
    """Classify the supplied email text as real or scam using the Gemini model."""
    if not text.strip():
        return "Unable to classify empty text."

    if model is None:
        return MODEL_INIT_ERROR

    prompt = f"""
    You are an expert in identifying scam messages in text, email etc. Analyze the given text and classify it as:

    - **Real/Legitimate** (Authentic, safe message)
    - **Scam/Fake** (Phishing, fraud, or suspicious message)

    **for the following Text:**
    {text}

    **Return a clear message indicating whether this content is real or a scam. 
    If it is a scam, mention why it seems fraudulent. If it is real, state that it is legitimate.**

    **Only return the classification message and nothing else.**
    Note: Don't return empty or null, you only need to return message for the input text
    """

    try:
        response = model.generate_content(prompt)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Failed to classify email content: %s", exc)
        return "Classification failed due to an unexpected error."

    return response.text.strip() if response and getattr(response, "text", "").strip() else "Classification failed."


def url_detection(url: str) -> str:
    """Classify the supplied URL into threat categories using the Gemini model."""
    if model is None:
        return MODEL_INIT_ERROR

    prompt = f"""
    You are an advanced AI model specializing in URL security classification. Analyze the given URL and classify it as one of the following categories:

    1. Benign**: Safe, trusted, and non-malicious websites such as google.com, wikipedia.org, amazon.com.
    2. Phishing**: Fraudulent websites designed to steal personal information. Indicators include misspelled domains (e.g., paypa1.com instead of paypal.com), unusual subdomains, and misleading content.
    3. Malware**: URLs that distribute viruses, ransomware, or malicious software. Often includes automatic downloads or redirects to infected pages.
    4. Defacement**: Hacked or defaced websites that display unauthorized content, usually altered by attackers.

    **Example URLs and Classifications:**
    - **Benign**: "https://www.microsoft.com/"
    - **Phishing**: "http://secure-login.paypa1.com/"
    - **Malware**: "http://free-download-software.xyz/"
    - **Defacement**: "http://hacked-website.com/"

    **Input URL:** {url}

    **Output Format:**  
    - Return only a string class name
    - Example output for a phishing site:  

    Analyze the URL and return the correct classification (Only name in lowercase such as benign etc.
    Note: Don't return empty or null, at any cost return the corrected class
    """

    try:
        response = model.generate_content(prompt)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Failed to classify URL: %s", exc)
        return "Detection failed due to an unexpected error."

    return response.text.strip() if response and getattr(response, "text", "").strip() else "Detection failed."


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

    return "", "Invalid file type. Please upload a PDF or TXT file."


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
        return render_index(message="No file uploaded.")

    if not is_supported_file(uploaded_file.filename):
        return render_index(message="Unsupported file type. Please upload PDF or TXT files.")

    extracted_text, error_message = extract_text_from_upload(uploaded_file)
    if error_message:
        return render_index(message=error_message)

    message = predict_fake_or_real_email_content(extracted_text)
    return render_index(message=message)


@app.route("/predict", methods=["POST"])
def predict_url():
    url = request.form.get("url", "").strip()

    if not url:
        return render_index(message="Please provide a URL to classify.")

    if not is_valid_url(url):
        return render_index(message="Invalid URL format. Include http:// or https://", input_url=url)

    classification = url_detection(url)
    return render_index(input_url=url, predicted_class=classification)


if __name__ == "__main__":
    app.run(debug=True)
