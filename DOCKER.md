# 🐳 Docker - Ejecutar con un comando

## Opción: Todo listo en un contenedor

Si tu profesor tiene Docker instalado, puede ejecutar todo con UN SOLO COMANDO:

### 1. Construir la imagen (primera vez)

```bash
docker build -t escriba-medico:latest .
```

Esto tardará ~20-30 minutos (descarga Python, dependencias, spaCy ES, Ollama y modelo Aloe-Beta).

### 2. Ejecutar el contenedor

```bash
docker run -p 8501:8501 escriba-medico:latest
```

Se abrirá automáticamente en http://localhost:8501 con TODO preconfigurado.

### 3. Parar el contenedor

```bash
docker stop <container_id>
```

---

## Ventajas de Docker

✅ Cero configuración manual  
✅ Funciona igual en Windows/Mac/Linux  
✅ Modelo preincluido y listo  
✅ Todo en un contenedor aislado  

## Desventajas

❌ Docker ocupa ~10-15 GB  
❌ Requiere Docker instalado  
❌ GPU NVIDIA requiere `nvidia-docker`  

---

## Alternativa: Pre-compilar en GitHub Actions

Se puede usar GitHub Actions para generar la imagen Docker automáticamente y alojarla en GitHub Container Registry (GHCR).

Así el profesor solo hace:

```bash
docker pull ghcr.io/Seiksssss/escriba-medico:latest
docker run -p 8501:8501 ghcr.io/Seiksssss/escriba-medico:latest
```

Sin necesidad de compilar nada.

---

## Recomendación

| Opción | Esfuerzo Prof. | Tiempo | Tamaño |
|--------|----------------|--------|--------|
| setup_model.ps1 (descarga automática) | ⭐ Mínimo | ~15 min | 0 GB (descarga) |
| Docker manual | ⭐⭐ Bajo | ~30 min | 10-15 GB |
| GitHub Actions + GHCR | ⭐⭐⭐ Profesional | ~5 min | 10-15 GB (en línea) |
