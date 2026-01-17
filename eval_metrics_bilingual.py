import argparse
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import pandas as pd
import ollama
from evaluate import load

# --- Logging setup ---
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)
log_file = os.path.join(logs_dir, f"eval_aloe_v3_bilingual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Dataset paths ---
DATA_DIR_EN = os.path.join("data", "RAW", "CSV", "har1MTS_Dialogue-Clinical_Note")
VALID_PATH_EN = os.path.join(DATA_DIR_EN, "MTS-Dialog-Validation.csv")

DATA_DIR_ES = os.path.join("data", "RAW", "CSV", "mts_dialog_spanish_csvs")
VALID_PATH_ES = os.path.join(DATA_DIR_ES, "validation_es.csv")

# --- Helpers ---
def get_cols(row: pd.Series, input_col: str, ref_col: str) -> Optional[Tuple[str, str]]:
    if input_col not in row or ref_col not in row:
        return None
    d = row[input_col]
    r = row[ref_col]
    return ("" if pd.isna(d) else str(d), "" if pd.isna(r) else str(r))


def run_inference(model_name: str, prompt: str) -> str:
    resp = ollama.generate(model=model_name, prompt=prompt, stream=False)
    return resp.get('response', '') if isinstance(resp, dict) else str(resp)


def compute_metrics(preds: List[str], refs: List[str], lang: str) -> Dict[str, float]:
    rouge = load('rouge')
    bertscore = load('bertscore')
    bleu = load('bleu')
    meteor = load('meteor')
    r = rouge.compute(predictions=preds, references=refs)
    b = bertscore.compute(predictions=preds, references=refs, lang=lang)
    bl = bleu.compute(predictions=preds, references=[[r] for r in refs])
    me = meteor.compute(predictions=preds, references=refs)
    return {
        'rouge1': float(r['rouge1']),
        'rouge2': float(r['rouge2']),
        'rougeL': float(r['rougeL']),
        'bertscore_f1': float(sum(b['f1']) / len(b['f1'])) if b['f1'] else 0.0,
        'bleu': float(bl['bleu']) if 'bleu' in bl else 0.0,
        'meteor': float(me['meteor']) if 'meteor' in me else 0.0,
    }


def evaluate_dataset(model_name: str, df: pd.DataFrame, lang: str, input_col: str, ref_col: str) -> Tuple[Dict[str, float], List[Dict]]:
    results = []
    for i, row in df.iterrows():
        cols = get_cols(row, input_col, ref_col)
        if cols is None:
            logger.error(f"Missing required columns '{input_col}' or '{ref_col}' in row {i}")
            continue
        dialogue, reference = cols
        try:
            if lang == 'es':
                prompt = f"Genera una nota clínica SOAP para este diálogo (responde en español): {dialogue}"
            else:
                prompt = f"Generate a clinical SOAP note for this dialogue (respond in English): {dialogue}"
            if (i+1) % 25 == 0 or i == 0:
                print(f"    [{i+1}/{len(df)}] ...")
            pred = run_inference(model_name, prompt)
            results.append({
                'id': int(i),
                'prediction': pred,
                'reference': reference,
                'dialogue': dialogue
            })
        except Exception as e:
            logger.error(f"Error case {i}: {e}")
    if not results:
        return {}, []
    preds = [r['prediction'] for r in results]
    refs = [r['reference'] for r in results]
    metrics = compute_metrics(preds, refs, lang)
    # Extra: lengths
    avg_pred_len = float(sum(len(p.split()) for p in preds)) / len(preds)
    avg_ref_len = float(sum(len(r.split()) for r in refs)) / len(refs)
    metrics.update({'avg_pred_len': avg_pred_len, 'avg_ref_len': avg_ref_len, 'num_cases': len(results)})
    return metrics, results


def main():
    parser = argparse.ArgumentParser(description="Evaluate aloe-beta v3 on validation datasets (ES + EN)")
    parser.add_argument('--model', type=str, default='escriba-aloe-v3:latest', help='Ollama model name')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of rows per dataset')
    parser.add_argument('--languages', nargs='*', default=['es','en'], choices=['es','en'], help='Languages to evaluate')
    args = parser.parse_args()

    modelos_disponibles = []
    try:
        response = ollama.list()
        modelos_disponibles = [m.model for m in response.models]
    except Exception as e:
        logger.warning(f"Could not list Ollama models: {e}")

    if args.model not in modelos_disponibles:
        print(f"⚠️ Model {args.model} not found in Ollama list. Proceeding anyway...")
        logger.warning(f"Model {args.model} not found in Ollama list")

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    datasets = []
    if 'en' in args.languages and os.path.exists(VALID_PATH_EN):
        df_en = pd.read_csv(VALID_PATH_EN)
        if args.limit:
            df_en = df_en.head(args.limit)
        # EN: input 'dialogue' -> reference 'section_text'
        datasets.append(('en', 'validation_en', df_en, 'dialogue', 'section_text'))
        logger.info(f"Loaded English validation: {len(df_en)} rows")
    if 'es' in args.languages and os.path.exists(VALID_PATH_ES):
        df_es = pd.read_csv(VALID_PATH_ES)
        if args.limit:
            df_es = df_es.head(args.limit)
        # ES: input 'dialogue' -> reference 'summary'
        datasets.append(('es', 'validation_es', df_es, 'dialogue', 'summary'))
        logger.info(f"Loaded Spanish validation: {len(df_es)} rows")

    if not datasets:
        print("❌ No datasets found to evaluate")
        logger.error("No datasets found to evaluate")
        return

    resultados_por_idioma: Dict[str, List[Dict]] = {}
    metricas_por_idioma: Dict[str, Dict[str, float]] = {}

    print(f"Evaluating model: {args.model} on {', '.join([d for (_, d, _, _, _) in datasets])}")
    logger.info(f"Evaluating model {args.model}")

    for lang, dname, df, input_col, ref_col in datasets:
        print("\n" + "-"*60)
        print(f"▶ Dataset: {dname} ({len(df)} rows) [{lang}]")
        logger.info(f"Dataset {dname} [{lang}]: {len(df)} rows")
        metrics, results = evaluate_dataset(args.model, df, lang, input_col, ref_col)
        if results:
            resultados_por_idioma[dname] = results
            metricas_por_idioma[dname] = metrics
            print(f"    ✅ Metrics: ROUGE-1 {metrics['rouge1']:.4f}, ROUGE-L {metrics['rougeL']:.4f}, BERT {metrics['bertscore_f1']:.4f}, BLEU {metrics['bleu']:.4f}, METEOR {metrics['meteor']:.4f}")
        else:
            print("    ⚠️ No results to compute metrics")

    # Persist results
    payload = {
        'timestamp': datetime.now().isoformat(),
        'model': args.model,
        'datasets': [d for (_, d, _, _, _) in datasets],
        'metricas_por_idioma': metricas_por_idioma,
        'resultados_por_idioma': resultados_por_idioma,
    }
    out_json = os.path.join(logs_dir, f"eval_aloe_v3_bilingual_{ts}.json")
    out_csv = os.path.join(logs_dir, f"eval_aloe_v3_bilingual_{ts}.csv")
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # Build CSV summary (one row per dataset)
    summary_rows = []
    for dname, m in metricas_por_idioma.items():
        summary_rows.append({
            'Model': args.model,
            'Dataset': dname,
            'Cases': m.get('num_cases', 0),
            'ROUGE-1': f"{m.get('rouge1', 0.0):.4f}",
            'ROUGE-2': f"{m.get('rouge2', 0.0):.4f}",
            'ROUGE-L': f"{m.get('rougeL', 0.0):.4f}",
            'BERTScore': f"{m.get('bertscore_f1', 0.0):.4f}",
            'BLEU': f"{m.get('bleu', 0.0):.4f}",
            'METEOR': f"{m.get('meteor', 0.0):.4f}",
            'AvgPredLen': f"{m.get('avg_pred_len', 0.0):.2f}",
            'AvgRefLen': f"{m.get('avg_ref_len', 0.0):.2f}",
        })
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False, encoding='utf-8')

    print("\n" + "="*80)
    print(f"✅ Saved JSON: {out_json}")
    print(f"✅ Saved CSV:  {out_csv}")
    print("="*80)


if __name__ == '__main__':
    main()
