# Start script for Escriba Médico Soberano (TFM)
# - Creates/activates venv
# - Installs dependencies and spaCy model (first run)
# - Starts Streamlit app

param(
    [switch]$ForceInstall
)

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "   Escriba Médico — Inicio Rápido (TFM)" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

$venvDir = ".\.venv"
$venvPython = "$venvDir\Scripts\python.exe"

# Create venv if not present
if (-not (Test-Path $venvPython)) {
    Write-Host "[INFO] Creando entorno virtual (.venv)..." -ForegroundColor Yellow
    python -m venv $venvDir
}

# Activate venv in this session
$activateScript = "$venvDir\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    Write-Host "[INFO] Activando entorno virtual..." -ForegroundColor Yellow
    . $activateScript
} else {
    Write-Host "[ERROR] No se encontró $activateScript" -ForegroundColor Red
    exit 1
}

# Install requirements if needed
$needsInstall = $ForceInstall -or (-not (Get-Command pip -ErrorAction SilentlyContinue))
if (-not $needsInstall) {
    # Try import of key modules
    try {
        python - << 'PY'
import sys
mods = ["streamlit","faster_whisper","torch","ollama","transformers","sentence_transformers","spacy"]
missing = []
for m in mods:
    try:
        __import__(m)
    except Exception:
        missing.append(m)
print("MISSING:"+",".join(missing))
PY
    } catch {
        $needsInstall = $true
    }
}

if ($needsInstall) {
    Write-Host "[INFO] Instalando dependencias (requirements.txt)..." -ForegroundColor Yellow
    pip install -r requirements.txt
}

# Ensure spaCy ES model
Write-Host "[INFO] Verificando modelo spaCy 'es_core_news_sm'..." -ForegroundColor Yellow
try {
    python - << 'PY'
import spacy
spacy.load("es_core_news_sm")
print("OK")
PY
} catch {
    Write-Host "[INFO] Descargando modelo spaCy ES..." -ForegroundColor Yellow
    python -m spacy download es_core_news_sm
}

# Check Ollama availability (optional)
Write-Host "[INFO] Comprobando Ollama..." -ForegroundColor Yellow
try {
    & ollama --version | Out-Null
    Write-Host "[OK] Ollama detectado" -ForegroundColor Green
} catch {
    Write-Host "[WARN] Ollama no detectado en PATH. Abra la app de Ollama o ejecute 'ollama serve'." -ForegroundColor Yellow
}

# Launch Streamlit
Write-Host "`n[OK] Iniciando aplicación Streamlit...`n" -ForegroundColor Green
python -m streamlit run app.py
