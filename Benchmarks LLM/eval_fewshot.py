
import os, json, time, random
from pathlib import Path
import pandas as pd
import requests
from sklearn.metrics import classification_report

# chemins
DATA_DIR = Path("data")
FEWSHOT_JSONL = DATA_DIR / "fewshot_balanced_full.jsonl"   # 24 ex équilibrés
TEST_CSV      = DATA_DIR / "test_split_full.csv"           # test complet

# Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral:7b-instruct"

# Réglages de génération (stabilité & déterminisme)
GEN_OPTIONS = {
    "temperature": 0.0,   # sorties plus déterministes
    "seed": 1234,         # même prompt -> même sortie
    # tu peux ajouter top_p, top_k, num_predict, etc. si besoin
    # cf. 'options' de l'API generate
}

# Prompt templates
SYSTEM_INSTRUCTIONS = (
    "Tu es un classificateur de tweets SAV Free.\n"
    "Renvoyer UNIQUEMENT un JSON valide au format EXACT:\n"
    '{ "is_claim": true|false, "topics": string[], '
    '"sentiment": "neg|neu|pos", "urgence": "string", '
    '"incident": "string", "confidence": number|null }\n'
    "Règles:\n"
    "- pas de texte hors JSON;\n"
    "- si ambigu, choisis la classe la plus probable;\n"
    "- 'topics' = 0..3 mots-clés français, courts.\n"
)

def build_fewshot_block(fewshot_records):
    parts = ["# Exemples:"]
    for rec in fewshot_records:
        t = rec["T"].strip()
        y = json.dumps(rec["Y"], ensure_ascii=False)
        parts.append(f'T: "{t}"\nY: {y}\n')
    return "\n".join(parts)

def build_prompt(fewshot_records, tweet_to_classify):
    block = build_fewshot_block(fewshot_records)
    return (
        SYSTEM_INSTRUCTIONS
        + "\n" + block
        + "\n# À classifier:\n"
        + f'T: <<<{tweet_to_classify}>>>\nY:'
    )

# Appel Ollama
def call_ollama_json(prompt, model=MODEL, timeout=300):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",   # structured outputs / JSON mode
        "options": GEN_OPTIONS,  # temperature=0, seed=1234, etc.
        # "system": SYSTEM_INSTRUCTIONS,  # déjà injecté dans le prompt
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    obj = r.json()  # contient aussi prompt_eval_count / eval_count
    resp = obj.get("response", "").strip()

    try:
        parsed = json.loads(resp)
    except Exception:
        # petit rappel si non-JSON
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

# Utilitaires
def normalize_sent(x: str) -> str:
    s = str(x).strip().lower()
    if s in {"negatif", "negative", "neg"}: return "neg"
    if s in {"neutre", "neutral", "neu"}:   return "neu"
    if s in {"positif", "positive", "pos"}: return "pos"
    return s

def pick_balanced_subset(records, num_shots):
    by_class = {"neg": [], "neu": [], "pos": []}
    for r in records:
        lab = normalize_sent(r["Y"].get("sentiment", "neu"))
        if lab in by_class:
            by_class[lab].append(r)
    # taille par classe
    n_classes = len(by_class)
    per_class = num_shots // n_classes
    out = []
    for lab in ["neg", "neu", "pos"]:
        pool = by_class[lab]
        if len(pool) == 0: continue
        if len(pool) < per_class:
            out.extend(pool)
        else:
            out.extend(pool[:per_class])
    return out

def main():
    import argparse, numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-shots", type=int, default=24, choices=[6,12,24],
                    help="Nombre d'exemples few-shot (équilibrés si possible)")
    ap.add_argument("--shuffle-order", action="store_true",
                    help="Mélanger l'ordre des exemples few-shot")
    ap.add_argument("--seed", type=int, default=1234,
                    help="Graine pour ordre/reproductibilité (prompt & options)")
    args = ap.parse_args()

    # mettre à jour la seed des options d'inférence
    GEN_OPTIONS["seed"] = int(args.seed)

    # charger few-shot complet (24)
    fewshot_all = []
    with FEWSHOT_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            fewshot_all.append(json.loads(line))

    # sélectionner k exemples équilibrés si besoin
    fewshot = pick_balanced_subset(fewshot_all, args.num_shots)
    # mélanger l'ordre (l'ICL y est sensible)
    if args.shuffle_order:
        random.Random(args.seed).shuffle(fewshot)

    # charger test (full)
    test_df = pd.read_csv(TEST_CSV)
    assert "tweet" in test_df.columns and "sentiment" in test_df.columns

    y_true, y_pred, lat = [], [], []
    outputs = []
    tok_prompt_list, tok_gen_list = [], []

    for _, row in test_df.iterrows():
        tweet = str(row["tweet"])
        gold_sent = str(row["sentiment"]).strip()

        prompt = build_prompt(fewshot, tweet)

        t0 = time.time()
        out, meta = call_ollama_json(prompt)
        dt = time.time() - t0

        pred_sent = normalize_sent(out.get("sentiment", "neu"))

        # garder métriques & sorties
        y_true.append(gold_sent)
        y_pred.append(pred_sent)
        lat.append(dt)

        tok_prompt_list.append(meta.get("prompt_eval_count"))
        tok_gen_list.append(meta.get("eval_count"))

        outputs.append({
            "tweet": tweet,
            "gold_sentiment": gold_sent,
            "pred_sentiment": pred_sent,
            "model_output": out,
            "latency_s": round(dt, 3),
            "prompt_eval_count": meta.get("prompt_eval_count"),
            "eval_count": meta.get("eval_count"),
        })

    # métriques
    rep = classification_report(y_true, y_pred, digits=3, zero_division=0)
    s = pd.Series(lat)
    stats = {"p50_ms": round(s.quantile(0.50)*1000,1),
             "p95_ms": round(s.quantile(0.95)*1000,1),
             "n": len(y_true)}

    tok_prompt = pd.Series(tok_prompt_list, dtype="float").dropna()
    tok_gen = pd.Series(tok_gen_list, dtype="float").dropna()
    token_stats = {
        "prompt_tokens_avg": float(tok_prompt.mean()) if not tok_prompt.empty else None,
        "output_tokens_avg": float(tok_gen.mean()) if not tok_gen.empty else None,
    }

    print(f"\n=== FEW-SHOT (Ollama / {MODEL}) — shots={args.num_shots} "
          f"{'(shuffled)' if args.shuffle_order else ''} ===")
    print(rep)
    print({**stats, **token_stats})

    # sauvegardes
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # CSV minimal
    out_csv = DATA_DIR / f"preds_fewshot_ollama_{args.num_shots}.csv"
    pd.DataFrame(outputs)[["tweet","gold_sentiment","pred_sentiment",
                           "latency_s","prompt_eval_count","eval_count"]].to_csv(out_csv, index=False)

    # JSONL complet
    out_jsonl = DATA_DIR / f"preds_fewshot_ollama_{args.num_shots}.jsonl"
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
    out_csv_full = DATA_DIR / f"preds_fewshot_ollama_full_{args.num_shots}.csv"
    pd.DataFrame(rows_full).to_csv(out_csv_full, index=False)

    print("Fichiers écrits :")
    print(" -", out_csv)
    print(" -", out_jsonl)
    print(" -", out_csv_full, "(CSV full avec 6 champs prédits)")

if __name__ == "__main__":
    main()
