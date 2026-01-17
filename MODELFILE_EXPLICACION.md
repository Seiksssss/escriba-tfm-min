# 📋 Modelfile - Prompt Médico Personalizado

## ¿Qué es?

El `Modelfile` es un archivo de configuración de Ollama que define:
1. **Modelo base:** Aloe-Beta-8B (GGUF Q4_K_M)
2. **System Prompt:** Instrucciones médicas especializadas
3. **Parámetros:** Temperatura (0.05 para precisión), top_p, repeat_penalty

## Cómo crear el modelo personalizado

```powershell
ollama create escriba-aloe-v3 -f Modelfile
```

Esto genera un modelo llamado `escriba-aloe-v3` que la app usará automáticamente.

## Contenido del System Prompt

### ROL
- Escriba Médico de Alta Precisión
- Sintetiza transcripciones en notas SOAP técnicas
- Usa lenguaje MIR (Medicina Interna Racional)

### REGLAS DE ORO (SEGURIDAD ASISTENCIAL)

#### 1. Correlación Lógica
Si el paciente menciona medicación crónica (Insulina, Metformina, Estatinas), DEBES incluir la patología base en 'S' (Antecedentes), aunque no se nombre explícitamente.

**Ejemplo:**
- Transcripción: "Toma Metformina"
- ❌ Incorrecto: S: Toma Metformina
- ✅ Correcto: S: Diabetes Mellitus II en tratamiento. Toma Metformina 1000mg c/12h

#### 2. Veracidad Documental
Prohibido incluir hallazgos negativos si no han sido verbalizados. La ausencia de dato es siempre "No referido".

**Ejemplo:**
- ❌ Incorrecto: "Abdomen no doloroso, sin masas"
- ✅ Correcto: "Abdomen: No referido. Exploración no realizada"

#### 3. Extracción Numérica Exacta
Captura cifras precisas: TA, FC, Glucemias, tiempos. NO omitas unidades.

**Ejemplo:**
- ❌ "TA elevada"
- ✅ "TA 156/92 mmHg"

#### 4. Estilo Técnico
- Usa sintagmas nominales (sin verbos conjugados)
- Sin artículos ni nexos
- Traduce a tecnicismos (ej. "dolor de oído" → "otalgia")

**Ejemplo:**
- ❌ "El paciente tiene dolor de cabeza y está mareado"
- ✅ "Cefalea. Mareos."

### ESTRUCTURA DE SALIDA

```json
{
  "S (SUBJETIVO)": "Motivo consulta, antecedentes con patologías asociadas a medicación, síntomas, cronología",
  "O (OBJETIVO)": "Constantes vitales, hallazgos exploración, pruebas",
  "A (APRECIACIÓN)": "Juicio clínico o sospecha diagnóstica principal",
  "P (PLAN)": "Pruebas, recomendaciones, seguimiento",
  "MEDICACIÓN": "Fármaco | Dosis | Frecuencia | Duración",
  "DIAGNÓSTICOS_CANDIDATOS": "Listado por probabilidad (Alta/Media/Baja)"
}
```

### EXCLUSIONES
- ❌ No incluyas saludos
- ❌ No incluyas consejos del sistema
- ❌ No incluyas educación sanitaria no verbal por el médico

## Parámetros de Configuración

| Parámetro | Valor | Razón |
|-----------|-------|-------|
| `temperature` | 0.05 | Baja → Respuestas deterministas y precisas |
| `top_p` | 0.9 | Balance: coherencia + variedad |
| `repeat_penalty` | 1.2 | Evita repeticiones innecesarias |

## Diferencia: Modelo Base vs Personalizado

| Aspecto | Modelo Base | Modelo Personalizado (escriba-aloe-v3) |
|---------|-----------|-------|
| **Modelo Base** | Aloe-Beta-8B | Aloe-Beta-8B |
| **System Prompt** | Genérico | ✅ Médico especializado |
| **REGLAS DE ORO** | No | ✅ Incluidas |
| **Temperatura** | 0.7 (default) | 0.05 (precisión) |
| **Salida SOAP** | Variable | ✅ Estructurada |

## Uso en la App

La app busca automáticamente el modelo en este orden:

1. `escriba-aloe-v3` (personalizado - si existe)
2. `hf.co/mradermacher/Llama3.1-Aloe-Beta-8B-GGUF:Q4_K_M` (base - fallback)

**Recomendación:** Siempre crea `escriba-aloe-v3` con `ollama create` para garantizar máxima precisión.

## Validación

Para verificar que el modelo personalizado se creó correctamente:

```powershell
ollama list
# Deberías ver: escriba-aloe-v3  (en la lista)

ollama run escriba-aloe-v3
# Prueba escribiendo: "Genera una nota SOAP para paciente con hipertensión"
```

---

**Nota:** Si necesitas actualizar el Modelfile, simplemente edita el archivo y vuelve a ejecutar:
```powershell
ollama delete escriba-aloe-v3
ollama create escriba-aloe-v3 -f Modelfile
```
