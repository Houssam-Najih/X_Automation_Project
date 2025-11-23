
import json, time
from pathlib import Path

import pandas as pd
import requests
from sklearn.metrics import classification_report

# chemins
DATA_DIR = Path("data")
TEST_CSV = DATA_DIR / "test_split_full.csv"

# import du system prompt + 3-shots fournis (inchangés)
from system_prompt_fewshot import SYSTEM_PROMPT, FEW_SHOTS 

# Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral:7b-instruct"
GEN_OPTIONS = {"temperature": 0.0, "seed": 1234}

def build_prompt_system_3shot(tweet_to_classify: str) -> str:
    parts = [SYSTEM_PROMPT.strip(), "\n# Exemples:"]
    for T, Y_json in FEW_SHOTS:
        parts.append(f'T: "{T.strip()}"\nY: {Y_json.strip()}\n')
    parts.append("# À classifier:")
    parts.append(f'T: <<<{tweet_to_classify}>>>\nY:')
    return "\n".join(parts)

def call_ollama_json(prompt: str, timeout=300):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",     # structured output JSON
        "options": GEN_OPTIONS,  # temperature=0, seed
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    obj = r.json()
    resp = obj.get("response", "").strip()
    try:
        parsed = json.loads(resp)
    except Exception:
        # rappel si le modèle n'a pas renvoyé un JSON propre
        payload["prompt"] = prompt + "\nRéponds UNIQUEMENT en JSON valide."
        r2 = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        r2.raise_for_status()
        obj = r2.json()
        parsed = json.loads(obj.get("response", "").strip())
    meta = {
        "prompt_eval_count": obj.get("prompt_eval_count"),
        "eval_count": obj.get("eval_count"),
        "total_duration_ns": obj.get("total_duration"),
        "eval_duration_ns": obj.get("eval_duration"),
        "prompt_eval_duration_ns": obj.get("prompt_eval_duration"),
    }
    return parsed, meta

def norm_sent(x: str) -> str:
    s = str(x).strip().lower()
    if s in {"negatif","negative","neg"}: return "neg"
    if s in {"neutre","neutral","neu"}:   return "neu"
    if s in {"positif","positive","pos"}: return "pos"
    return s

def main():
    # charge test
    test_df = pd.read_csv(TEST_CSV)
    assert "tweet" in test_df.columns and "sentiment" in test_df.columns

    y_true, y_pred, lat = [], [], []
    outputs = []
    prompt_tok, out_tok = [], []

    for _, row in test_df.iterrows():
        tweet = str(row["tweet"])
        gold = str(row["sentiment"]).strip()

        prompt = build_prompt_system_3shot(tweet)

        t0 = time.time()
        out, meta = call_ollama_json(prompt)
        dt = time.time() - t0

        pred = norm_sent(out.get("sentiment","neu"))

        y_true.append(gold)
        y_pred.append(pred)
        lat.append(dt)
        prompt_tok.append(meta.get("prompt_eval_count"))
        out_tok.append(meta.get("eval_count"))

        outputs.append({
            "tweet": tweet,
            "gold_sentiment": gold,
            "pred_sentiment": pred,
            "model_output": out,
            "latency_s": round(dt,3),
            "prompt_eval_count": meta.get("prompt_eval_count"),
            "eval_count": meta.get("eval_count"),
        })

    # métriques
    rep = classification_report(y_true, y_pred, digits=3, zero_division=0)
    s = pd.Series(lat)
    stats = {"p50_ms": round(s.quantile(0.50)*1000,1),
             "p95_ms": round(s.quantile(0.95)*1000,1),
             "n": len(y_true)}

    pt = pd.Series(prompt_tok, dtype="float").dropna()
    ot = pd.Series(out_tok, dtype="float").dropna()
    token_stats = {
        "prompt_tokens_avg": float(pt.mean()) if not pt.empty else None,
        "output_tokens_avg": float(ot.mean()) if not ot.empty else None,
    }

    print("\n=== SYSTEM-PROMPT + 3-shot (Ollama / Mistral 7B Instruct) ===")
    print(rep)
    print({**stats, **token_stats})

    # sauvegardes
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    out_csv = DATA_DIR / "preds_system3_ollama.csv"
    pd.DataFrame(outputs)[["tweet","gold_sentiment","pred_sentiment",
                           "latency_s","prompt_eval_count","eval_count"]].to_csv(out_csv, index=False)

    out_jsonl = DATA_DIR / "preds_system3_ollama.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in outputs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # CSV full (6 champs prédits)
    rows_full = []
    for r in outputs:
        mo = r["model_output"] if isinstance(r["model_output"], dict) else {}
        rows_full.append({
            "tweet": r["tweet"],
            "gold_sentiment": r["gold_sentiment"],
            "pred_sentiment": r["pred_sentiment"],
            "pred_is_claim": mo.get("is_claim", None),
            "pred_topics": json.dumps(mo.get("topics", []), ensure_ascii=False),
            "pred_urgence": mo.get("urgence", ""),
            "pred_incident": mo.get("incident", ""),
            "pred_confidence": mo.get("confidence", None),
            "latency_s": r["latency_s"],
            "prompt_eval_count": r.get("prompt_eval_count"),
            "eval_count": r.get("eval_count"),
        })
    out_csv_full = DATA_DIR / "preds_system3_ollama_full.csv"
    pd.DataFrame(rows_full).to_csv(out_csv_full, index=False)

    print("Fichiers écrits :")
    print(" -", out_csv)
    print(" -", out_jsonl)
    print(" -", out_csv_full, "(CSV full avec 6 champs prédits)")

if __name__ == "__main__":
    main()
