# Auditor Mejorado - Versión 2
# Este archivo contiene las funciones propuestas para mejorar la auditoría

import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from typing import Dict, List, Tuple
import logging

# spaCy es opcional - solo para análisis lingüístico avanzado
# Nota: spaCy tiene problemas de compatibilidad con Python 3.14+ debido a pydantic
try:
    import spacy
    SPACY_AVAILABLE = True
except Exception:  # Catch all exceptions, including pydantic.v1.errors.ConfigError
    SPACY_AVAILABLE = False

logger = logging.getLogger(__name__)

class AuditorMejorado:
    """Sistema de auditoría avanzado con múltiples capas de validación"""
    
    def __init__(self, device: int = 0):
        self.device = device
        self.nli_pipeline = None
        self.embeddings_model = None
        self.nlp = None
        
    def cargar_modelos(self):
        """Cargar modelos necesarios para auditoría mejorada"""
        logger.info("Cargando modelos para auditoría mejorada...")
        
        # 1. Modelo para Natural Language Inference (mejor que zero-shot)
        self.nli_pipeline = pipeline(
            "zero-shot-classification",
            model="microsoft/deberta-large-mnli",
            device=self.device
        )
        logger.info("✅ MNLI cargado")
        
        # 2. Modelo para embeddings semánticos
        self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✅ Embeddings cargado")
        
        # 3. SpaCy para análisis lingüístico (OPCIONAL)
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("es_core_news_sm")
                logger.info("✅ spaCy ES cargado")
            except:
                logger.warning("⚠️ Modelo spaCy 'es_core_news_sm' no encontrado. Usando análisis básico.")
                self.nlp = None
        else:
            logger.warning("⚠️ spaCy no está instalado (solo para Python < 3.14). Usando análisis básico.")
            self.nlp = None
    
    def encontrar_fragmento_similar(self, texto_buscar: str, transcripcion: str, top_k: int = 1) -> List[Tuple[str, float, float, float]]:
        """Encontrar fragmentos con búsqueda híbrida: semántica + keywords.

        Retorna tuplas (fragmento, score_combinado, score_keywords, score_semantico).
        """
        # Dividir transcripción en oraciones
        doc = self.nlp(transcripcion) if self.nlp else None
        
        if doc:
            oraciones = [sent.text.strip() for sent in doc.sents]
        else:
            # Fallback: dividir por puntos
            oraciones = [s.strip() for s in transcripcion.split('.') if s.strip() and len(s.strip()) > 10]
        
        if not oraciones:
            return [("No encontrado", 0.0)]
        
        # Extraer keywords importantes del texto a buscar
        keywords = self._extraer_keywords(texto_buscar)
        
        # Crear ventanas con scores por keywords
        ventanas = []
        scores_keywords = []
        
        for i in range(len(oraciones)):
            inicio = max(0, i - 1)
            fin = min(len(oraciones), i + 3)  # ventana más amplia (hasta 5 oraciones)
            ventana = ' '.join(oraciones[inicio:fin])
            ventanas.append(ventana)
            
            # Score por keywords
            ventana_lower = ventana.lower()
            import re
            matches = 0
            for kw in keywords:
                try:
                    if re.search(r"\b" + re.escape(kw.lower()) + r"\b", ventana_lower):
                        matches += 1
                except re.error:
                    continue
            keyword_score = matches / max(len(keywords), 1)
            if matches > 0:
                keyword_score = min(1.0, keyword_score + 0.30)  # boost por match exacto
            scores_keywords.append(keyword_score)
        
        # Calcular embeddings semánticos
        try:
            import torch
            embedding_buscar = self.embeddings_model.encode(texto_buscar, convert_to_tensor=True)
            embeddings_ventanas = self.embeddings_model.encode(ventanas, convert_to_tensor=True)
            
            similarities = util.pytorch_cos_sim(embedding_buscar, embeddings_ventanas)[0]
            
            # Combinar score semántico + keywords (50/50 para priorizar coincidencias específicas)
            scores_keywords_tensor = torch.tensor(scores_keywords, device=similarities.device)
            scores_finales = 0.50 * similarities + 0.50 * scores_keywords_tensor
            
            # Top K basado en score combinado
            top_indices = scores_finales.argsort(descending=True)[:max(top_k * 2, 3)]
            
            resultados = []
            for idx in top_indices[:top_k]:
                idx_int = int(idx)
                if idx_int < len(ventanas):
                    fragmento = ventanas[idx_int][:200]
                    resultados.append((
                        fragmento,
                        float(scores_finales[idx]),
                        float(scores_keywords_tensor[idx]),
                        float(similarities[idx])
                    ))
            
            return resultados if resultados else [("No encontrado", 0.0, 0.0, 0.0)]
        except Exception as e:
            logger.error(f"Error en búsqueda de embeddings: {str(e)}")
            return [("Error en búsqueda", 0.0, 0.0, 0.0)]
    
    def _extraer_keywords(self, texto: str) -> list:
        """Extraer palabras clave importantes del texto."""
        import re
        
        # Números (edades, valores)
        numeros = re.findall(r'\b\d+\b', texto)
        
        # Palabras importantes
        palabras_importantes = []
        doc = self.nlp(texto) if self.nlp else None
        
        if doc:
            palabras_importantes = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN", "NUM"]]
        else:
            palabras_importantes = [w for w in texto.split() if len(w) > 4 and w.isalpha()]
        
        keywords = list(set(numeros + palabras_importantes))
        return keywords[:10]
    
    def validar_entailment(self, texto_soap: str, fragmento_transcripcion: str) -> Tuple[str, float]:
        """
        Validar si el fragmento de transcripción respalda el texto del SOAP usando MNLI.
        
        Usa clasificación entailment/contradiction/neutral para determinar si el fragmento
        confirma, contradice o es neutral respecto al SOAP.
        
        Returns:
            (estado, confianza) donde estado es VERIFICADO/NO_CONFIRMADO/DUDOSO
        """
        try:
            # Usar MNLI con premisa-hipótesis directo
            # Premisa: lo que dice la transcripción
            # Hipótesis: lo que dice el SOAP
            resultado = self.nli_pipeline(
                fragmento_transcripcion,  # premise
                [texto_soap],  # hypothesis
                hypothesis_template="{}",  # usar texto tal cual
                multi_label=False
            )
            
            # Resultado tiene labels: entailment (respalda), contradiction (contradice), neutral
            label_principal = resultado['labels'][0]
            confianza_principal = resultado['scores'][0]
            
            # Umbral más alto para VERIFICADO (evita falsos positivos)
            if label_principal == "entailment" and confianza_principal > 0.7:
                return ("VERIFICADO", confianza_principal)
            elif label_principal == "contradiction" or confianza_principal < 0.5:
                return ("NO_CONFIRMADO", confianza_principal)
            else:
                return ("DUDOSO", confianza_principal)
                
        except Exception as e:
            logger.error(f"Error en validación de entailment: {str(e)}")
            return ("ERROR", 0.0)
    
    def detectar_modalidad(self, texto: str) -> str:
        """
        Detectar la modalidad del texto (hecho, propuesta, duda, negación).
        
        Returns:
            "HECHO", "PROPUESTA", "DUDA", "NEGACION", "DESCONOCIDA"
        """
        if not self.nlp:
            return "DESCONOCIDA"
        
        doc = self.nlp(texto)
        
        # Patrones para detectar modalidad
        verbos = [token.lemma_ for token in doc if token.pos_ == "VERB"]
        
        # Verbos de hecho
        if any(v in ["ser", "estar", "tener", "haber"] for v in verbos):
            return "HECHO"
        
        # Verbos condicionales/propuesta
        condicionales = ["poder", "deber", "poner", "hacer"]
        if any(v in condicionales for v in verbos):
            # Revisar si está en condicional usando morph.get()
            if any(token.morph.get("Mood") == ["Cnd"] for token in doc if token.pos_ == "VERB"):
                return "PROPUESTA"
        
        # Negación
        if any(token.lemma_ == "no" for token in doc if token.pos_ == "ADV"):
            return "NEGACION"
        
        return "DESCONOCIDA"
    
    def analizar_cambio_modalidad(self, texto_soap: str, fragmento_trans: str) -> Dict:
        """
        Detectar si hay cambio de modalidad entre SOAP y transcripción.
        
        Ejemplo: 
        - SOAP (HECHO): "Se le prescribió antibiótico"
        - Transcripción (PROPUESTA): "Le podríamos prescribir antibiótico"
        
        Esto sería un problema: el SOAP presenta como hecho lo que fue propuesta.
        """
        modalidad_soap = self.detectar_modalidad(texto_soap)
        modalidad_trans = self.detectar_modalidad(fragmento_trans)
        
        cambio = modalidad_soap != modalidad_trans
        
        # Clasificar severidad del cambio
        severidad = "NINGUNA"
        if cambio:
            if modalidad_soap == "HECHO" and modalidad_trans in ["PROPUESTA", "DUDA"]:
                severidad = "ALTA"  # SOAP es más afirmativo que la realidad
            elif modalidad_soap == "PROPUESTA" and modalidad_trans == "HECHO":
                severidad = "MEDIA"  # SOAP es menos comprometido
            elif modalidad_soap == "NEGACION" and modalidad_trans == "HECHO":
                severidad = "CRÍTICA"  # Negación vs afirmación
        
        return {
            "cambio": cambio,
            "modalidad_soap": modalidad_soap,
            "modalidad_trans": modalidad_trans,
            "severidad": severidad
        }
    
    def auditar_informe_completo(self, transcripcion: str, informe_soap: str) -> Dict:
        """
        Auditoría completa con múltiples capas de validación.
        """
        if not self.nli_pipeline:
            self.cargar_modelos()
        
        logger.info("Iniciando auditoría mejorada...")
        
        # Dividir informe en oraciones
        doc = self.nlp(informe_soap) if self.nlp else None
        
        if doc:
            oraciones_soap = [sent.text for sent in doc.sents]
        else:
            oraciones_soap = [s.strip() for s in informe_soap.split('.') if s.strip() and len(s.strip()) > 15]
        
        verificaciones = []
        alucinaciones = []
        omisiones = []
        
        # --- VALIDACIÓN DE ORACIONES DEL SOAP ---
        for idx, oracion in enumerate(oraciones_soap[:15]):  # Limitar a 15 para tiempo
            logger.info(f"Validando oración {idx+1}/{len(oraciones_soap)}: {oracion[:60]}...")
            
            # 1. Búsqueda de fragmentos similares
            fragmentos = self.encontrar_fragmento_similar(oracion, transcripcion, top_k=4)
            if not fragmentos:
                fragmentos = [("No encontrado", 0.0, 0.0, 0.0)]
            
            # 2. Validar entailment sobre top-3 y elegir el mejor (confianza + semántica)
            mejor_idx = 0
            mejor_score_ent = -1.0
            estado_entailment = "DUDOSO"
            confianza_entailment = 0.0
            for j, frag in enumerate(fragmentos[:3]):
                frag_text, score_comb, score_kw, score_sem = frag
                est, conf = self.validar_entailment(oracion, frag_text)
                ranking_score = conf + 0.15 * score_sem  # prioriza entailment, ajusta por semántica
                if ranking_score > mejor_score_ent:
                    mejor_score_ent = ranking_score
                    mejor_idx = j
                    estado_entailment = est
                    confianza_entailment = conf
            
            mejor_fragmento, similitud, score_kw, score_sem = fragmentos[mejor_idx]
            
            # 3. Análisis de modalidad
            cambio_modalidad = self.analizar_cambio_modalidad(oracion, mejor_fragmento)
            
            # 4. Determinación final - balance precisión/recuperación
            if (score_kw >= 0.20) and estado_entailment in ["VERIFICADO", "DUDOSO"] and similitud > 0.55:
                estado_final = "VERIFICADO"
                bandera = None
            elif similitud > 0.72 and estado_entailment in ["VERIFICADO", "DUDOSO"]:
                estado_final = "VERIFICADO"
                bandera = None
            elif estado_entailment == "VERIFICADO" and similitud > 0.60:
                estado_final = "VERIFICADO"
                bandera = None
            elif estado_entailment == "NO_CONFIRMADO" and similitud < 0.25:
                estado_final = "ALUCINACION"
                alucinaciones.append({
                    "frase": oracion[:100],
                    "razon": f"No confirmada (similitud: {similitud:.2f}, entailment: {estado_entailment})"
                })
                bandera = "sin_evidencia"
            elif cambio_modalidad["severidad"] == "ALTA":
                estado_final = "ALUCINACION"
                alucinaciones.append({
                    "frase": oracion[:100],
                    "razon": f"Cambio de modalidad: SOAP={cambio_modalidad['modalidad_soap']}, Trans={cambio_modalidad['modalidad_trans']}"
                })
                bandera = "cambio_modalidad"
            elif similitud > 0.45:
                estado_final = "PARCIAL"
                bandera = None
            else:
                estado_final = "PARCIAL"
                bandera = "confianza_baja"
            
            verificaciones.append({
                "frase": oracion[:100],
                "estado": estado_final,
                "bandera": bandera,
                "confianza_entailment": round(confianza_entailment, 3),
                "similitud_semantica": round(similitud, 3),
                "similitud_pura": round(score_sem, 3),
                "score_keywords": round(score_kw, 3),
                "fragmento_referencia": mejor_fragmento[:80],
                "modalidad_soap": cambio_modalidad["modalidad_soap"],
                "modalidad_trans": cambio_modalidad["modalidad_trans"],
                "cambio_modalidad_detectado": cambio_modalidad["cambio"]
            })
        
        # --- DETECCIÓN DE OMISIONES ---
        # Palabras/síntomas clave a buscar
        palabras_clave = [
            "ardor", "dolor", "arde", "duele", "náusea", "vómito", "diarrea", "estreñimiento",
            "mareo", "somnolencia", "insomnio", "cansancio", "fatiga", "debilidad",
            "medicamento", "pastilla", "droga", "fármaco", "inyección",
            "alergia", "reacción", "efecto secundario"
        ]
        
        para_verificar = []
        for palabra in palabras_clave:
            if palabra.lower() in transcripcion.lower() and palabra.lower() not in informe_soap.lower():
                para_verificar.append(palabra)
        
        if para_verificar:
            omisiones.append({
                "palabras_no_incluidas": para_verificar[:5],
                "sugestion": "Revisar si estos síntomas/datos deberían estar en el SOAP"
            })
        
        # --- CÁLCULO DE MÉTRICAS ---
        verificadas = sum(1 for v in verificaciones if v["estado"] == "VERIFICADO")
        parciales = sum(1 for v in verificaciones if v["estado"] == "PARCIAL")
        alucinaciones_count = sum(1 for v in verificaciones if v["estado"] == "ALUCINACION")
        
        fidelidad = ((verificadas + parciales * 0.5) / len(verificaciones)) * 100 if verificaciones else 0
        
        resultado = {
            "verificaciones": verificaciones,
            "alucinaciones_detectadas": alucinaciones,
            "omisiones_detectadas": omisiones,
            "metricas": {
                "fidelidad": round(fidelidad, 1),
                "verificadas": verificadas,
                "parciales": parciales,
                "alucinaciones": alucinaciones_count,
                "total_oraciones": len(verificaciones),
                "omisiones": len(para_verificar) if para_verificar else 0
            },
            "recomendaciones": self._generar_recomendaciones(alucinaciones, omisiones, fidelidad)
        }
        
        logger.info(f"✅ Auditoría completada: Fidelidad {fidelidad:.1f}%")
        return resultado
    
    def _generar_recomendaciones(self, alucinaciones: List, omisiones: List, fidelidad: float) -> List[str]:
        """Generar recomendaciones basadas en los resultados"""
        recomendaciones = []
        
        if fidelidad < 70:
            recomendaciones.append("⚠️ BAJA FIDELIDAD: Revisar completamente el SOAP generado")
        
        if len(alucinaciones) > 2:
            recomendaciones.append(f"🚨 Múltiples alucinaciones detectadas ({len(alucinaciones)}): El modelo puede estar inventando información")
        
        if omisiones:
            recomendaciones.append("📝 Síntomas o datos relevantes no incluidos en el SOAP")
        
        if fidelidad > 85:
            recomendaciones.append("✅ Buena calidad del SOAP: Cambios menores recomendados")
        
        return recomendaciones


# ============== FUNCIÓN DE INTEGRACIÓN ==============

def auditar_informe_mejorado(transcripcion: str, informe_soap: str) -> Dict:
    """
    Función para reemplazar directamente auditar_informe() en app.py
    """
    try:
        auditor = AuditorMejorado(device=0 if torch.cuda.is_available() else -1)
        resultado = auditor.auditar_informe_completo(transcripcion, informe_soap)
        return resultado
    except Exception as e:
        logger.error(f"Error en auditoría mejorada: {str(e)}")
        return {
            "verificaciones": [],
            "alucinaciones_detectadas": [],
            "omisiones_detectadas": [],
            "metricas": {"fidelidad": 0, "error": str(e)}
        }
