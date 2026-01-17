#!/usr/bin/env powershell
# Script para descargar y configurar Aloe-Beta-8B automáticamente

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Descargando y configurando Aloe-Beta-8B" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Crear carpetas necesarias (logs y models)
Write-Host "[0/4] Creando estructura de carpetas..." -ForegroundColor Yellow
$dirs = @("logs", "models")

foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  ✅ Creada: $dir" -ForegroundColor Green
    } else {
        Write-Host "  ✅ Ya existe: $dir" -ForegroundColor Green
    }
}

# Verificar que Ollama está ejecutándose
Write-Host "`n[1/4] Verificando Ollama..." -ForegroundColor Yellow
try {
    $gh = Get-Command ollama -ErrorAction Stop
    Write-Host "✅ Ollama encontrado: $gh" -ForegroundColor Green
} catch {
    Write-Host "❌ Ollama no está instalado o no está en PATH" -ForegroundColor Red
    Write-Host "   Instala desde: https://ollama.com/download" -ForegroundColor Yellow
    Read-Host "Presiona Enter para salir"
    exit 1
}

# Descargar modelo base
Write-Host "`n[2/4] Descargando Aloe-Beta-8B (5-7 GB, puede tardar 10-15 min)..." -ForegroundColor Yellow
ollama pull hf.co/mradermacher/Llama3.1-Aloe-Beta-8B-GGUF:Q4_K_M

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Error descargando modelo" -ForegroundColor Red
    Read-Host "Presiona Enter para salir"
    exit 1
}

Write-Host "✅ Modelo base descargado" -ForegroundColor Green

# Crear modelo personalizado
Write-Host "`n[3/4] Creando modelo personalizado con prompt médico (escriba-aloe-v3)..." -ForegroundColor Yellow

if (Test-Path "Modelfile") {
    ollama create escriba-aloe-v3 -f Modelfile
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Modelo personalizado 'escriba-aloe-v3' creado exitosamente" -ForegroundColor Green
        Write-Host "`n🎉 ¡Listo! Ahora puedes ejecutar: streamlit run app.py" -ForegroundColor Green
    } else {
        Write-Host "⚠️ Error creando modelo personalizado" -ForegroundColor Yellow
        Write-Host "   Intenta manualmente: ollama create escriba-aloe-v3 -f Modelfile" -ForegroundColor Yellow
    }
} else {
    Write-Host "❌ Modelfile no encontrado en el directorio actual" -ForegroundColor Red
    Write-Host "   Asegúrate de estar en el directorio correcto" -ForegroundColor Yellow
}

# Verificar estructura final
Write-Host "`n[4/4] Verificando estructura..." -ForegroundColor Yellow
$checkDirs = @("logs", "models")

$allOk = $true
foreach ($dir in $checkDirs) {
    if (Test-Path $dir) {
        Write-Host "  ✅ Encontrada: $dir" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️ Falta: $dir" -ForegroundColor Yellow
        $allOk = $false
    }
}

Write-Host "`nArchivos de datos disponibles:" -ForegroundColor Cyan
$dataDirs = @("data/conversaciones", "data/mtsdialog_es", "data/mtsdialog_en")
foreach ($dir in $dataDirs) {
    if (Test-Path $dir) {
        $count = (Get-ChildItem $dir -Recurse -File | Measure-Object).Count
        Write-Host "  ✅ $dir ($count archivos)" -ForegroundColor Green
    } else {
        Write-Host "  ℹ️  $dir (no incluido en este clone)" -ForegroundColor Gray
    }
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Configuración completada" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan
