# Detect Scam Emails & Malicious URLs

## Prerequisites
- Python 3.9 or newer
- A Google AI Studio project with billing enabled and a Gemini API key

## Setup (PowerShell)
```powershell
# (Optional) create a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Provision environment variables
@"
GOOGLE_API_KEY=your-real-api-key
"@ | Out-File -FilePath .env -Encoding utf8 -Force

# Load the .env file into the current shell (optional convenience step)
Get-Content .env | ForEach-Object {
    if ($_ -match '^(?<key>[^=]+)=(?<value>.+)$') {
        Set-Item -Path env:$($Matches['key'].Trim()) -Value $Matches['value'].Trim()
    }
}
```

## Run the app
```powershell
python main.py
```

Open your browser to [http://127.0.0.1:5000](http://127.0.0.1:5000) to access the UI. Upload PDF/TXT files or provide URLs for classification.

## Notes
- Do not check API keys into source control; use environment variables or a secrets manager in production.
- `.gitignore` already excludes `.env`, `.venv`, and build artifacts to help keep secrets local only.
- Install dependencies from `requirements.txt` in automated environments (GitHub Actions, Docker, etc.).
- On startup the app runs `list_models()` and then attempts (in order) any explicit `GEMINI_MODEL`, followed by `gemini-2.0-flash-exp`, `gemini-2.0-flash`, `gemini-2.0-flash-001`, `gemini-2.0-flash-lite-latest`, `gemini-1.5-flash-latest`, `gemini-1.5-flash`, `gemini-1.5-flash-001`, `gemini-1.0-pro`, and `gemini-pro`. Override with `GEMINI_MODEL=your-model-id` if you have custom access. If you still see “model unavailable”, update the SDK (`pip install -U google-generativeai`) or request access to one of the supported models.
- For production deployments disable `debug=True` in `main.py`, serve over HTTPS, and store secrets in a secure location such as `.env` (managed via `python-dotenv`) or your hosting provider’s secret store.