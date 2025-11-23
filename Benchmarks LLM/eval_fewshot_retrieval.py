import os, json, time, math, random
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize

# chemins
DATA_DIR = Path("data")
TEST_CSV = DATA_DIR / "test_split_full.csv"              # test complet (tweet + 6 colonnes)
DEFAULT_POOL_JSONL = DATA_DIR / "fewshot_balanced_full.jsonl"  # pool par défaut (24 ex.)

# Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral:7b-instruct"
GEN_OPTIONS = {"temperature": 0.0, "seed": 1234}  

# Prompt
SYSTEM_INSTRUCTIONS = (
    "Tu es un classificateur de tweets SAV Free.\n"
    "Renvoyer UNIQUEMENT un JSON valide au format EXACT:\n"
    '{ "is_claim": true|false, "topics": string[], '
    '"sentiment": "neg|neu|pos", "urgence": "string", '
    '"incident": "string", "confidence": number|null }\n'
    "Règles:\n- pas de texte hors JSON;\n- si ambigu, choisis la classe la plus probable;\n"
    "- 'topics' = 0..3 mots-clés français, courts.\n"
)

def build_prompt(fewshot_records, tweet_to_classify):
    parts = [SYSTEM_INSTRUCTIONS, "\n# Exemples:"]
    for rec in fewshot_records:
        t = rec["T"].strip()
        y = json.dumps(rec["Y"], ensure_ascii=False)
        parts.append(f'T: "{t}"\nY: {y}\n')
    parts.append("# À classifier:")
    parts.append(f'T: <<<{tweet_to_classify}>>>\nY:')
    return "\n".join(parts)

def call_ollama_json(prompt, timeout=300):
    payload = {
        "model": MODEL, "prompt": prompt, "stream": False,
        "format": "json",  # Structured outputs (JSON)
        "options": GEN_OPTIONS
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    obj = r.json()
    resp = obj.get("response", "").strip()
    try:
        parsed = json.loads(resp)
    except Exception:
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

def norm_sent(x:str)->str:
    s=str(x).strip().lower()
    if s in {"negatif","negative","neg"}: return "neg"
    if s in {"neutre","neutral","neu"}:   return "neu"
    if s in {"positif","positive","pos"}: return "pos"
    return s

# Embeddings (petit modèle CPU)
def get_encoder(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)

def embed_texts(encoder, texts):
    vecs = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype="float32")  # L2-normalized

# Retrieval
def mmr_select(query_vec, pool_vecs, k, lambda_div=0.5):
    n = pool_vecs.shape[0]
    selected, candidates = [], list(range(n))
    if n <= k: return list(range(n))
    # précompute sim au query
    sim_to_q = (pool_vecs @ query_vec)
    # 1er: plus similaire
    first = int(sim_to_q.argmax())
    selected.append(first)
    candidates.remove(first)
    while len(selected) < k and candidates:
        best_c, best_score = None, -1e9
        for c in candidates:
            div = (pool_vecs[c:c+1] @ pool_vecs[selected].T).max() if selected else 0.0
            score = lambda_div * sim_to_q[c] - (1 - lambda_div) * div
            if score > best_score:
                best_score, best_c = score, c
        selected.append(best_c)
        candidates.remove(best_c)
    return selected

def load_pool_from_jsonl(path: Path):
    pool=[]
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            pool.append({"T": rec["T"], "Y": rec["Y"]})
    return pool

def pool_from_csv_excluding_test(csv_path: Path, test_df: pd.DataFrame):
    df = pd.read_csv(csv_path)
    assert "full_text" in df.columns
    test_texts = set(map(str, test_df["tweet"].tolist()))
    df = df[~df["full_text"].astype(str).isin(test_texts)].copy()
    cols = ["is_claim","topics","sentiment","urgence","incident","confidence"]
    cols_present = [c for c in cols if c in df.columns]
    pool=[]
    for _,r in df.iterrows():
        Y={}
        for c in cols_present:
            if c=="sentiment": Y[c]=str(r[c]).strip()
            elif c=="is_claim": Y[c]=bool(r[c])
            elif c=="topics":
                val=r[c]
                try: Y[c]=json.loads(val) if isinstance(val,str) else (val if isinstance(val,list) else [])
                except: Y[c]=[]
            elif c=="confidence":
                try: Y[c]=float(r[c])
                except: Y[c]=None
            else:
                Y[c]=str(r[c]) if not pd.isna(r[c]) else ""
        pool.append({"T": str(r["full_text"]), "Y": Y})
    return pool

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=12, choices=[7,12,24],
                    help="Nombre d'exemples récupérés par tweet (retrieval few-shot)")
    ap.add_argument("--lambda-div", type=float, default=0.5,
                    help="MMR lambda (0=diversité max, 1=similarité max)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--pool-jsonl", type=str, default=str(DEFAULT_POOL_JSONL),
                    help="Ban d'exemples (JSONL T/Y). Par défaut: 24 équilibrés")
    ap.add_argument("--pool-from-csv", type=str, default="",
                    help="Optionnel: construire la banque depuis un CSV annoté (exclut le test)")
    ap.add_argument("--csv-source", type=str, default="data/free_tweet_export.llm_labeled_300_2.csv",
                    help="CSV annoté si --pool-from-csv est utilisé")
    ap.add_argument("--encoder", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                    help="Encodeur d'embeddings (CPU-friendly)")
    args = ap.parse_args()
    random.seed(args.seed)

    # Charger test
    test_df = pd.read_csv(TEST_CSV)
    assert "tweet" in test_df.columns and "sentiment" in test_df.columns

    # Construire la banque d'exemples
    if args.pool_from_csv:
        pool = pool_from_csv_excluding_test(Path(args.csv_source), test_df)
    else:
        pool = load_pool_from_jsonl(Path(args.pool_jsonl))
    if len(pool) == 0:
        raise RuntimeError("Ban d'exemples vide.")

    # Embeddings (ban + test)
    encoder = get_encoder(args.encoder)
    pool_texts = [p["T"] for p in pool]
    pool_vecs = embed_texts(encoder, pool_texts)

    # k réel utilisé pour l'étiquette des fichiers (au cas où banque < k demandé)
    k_for_tag = min(args.k, len(pool))

    # Boucle d'éval
    y_true, y_pred, lat = [], [], []
    outputs = []
    prompt_tok, out_tok = [], []

    for _, row in test_df.iterrows():
        tweet = str(row["tweet"])
        gold  = str(row["sentiment"]).strip()

        q_vec = embed_texts(encoder, [tweet])[0]
        idxs = mmr_select(q_vec, pool_vecs, k=min(args.k, len(pool)), lambda_div=args.lambda_div)
        fewshot = [pool[i] for i in idxs]

        prompt = build_prompt(fewshot, tweet)

        t0 = time.time()
        out, meta = call_ollama_json(prompt)
        dt = time.time() - t0

        pred = norm_sent(out.get("sentiment","neu"))
        y_true.append(gold); y_pred.append(pred); lat.append(dt)
        prompt_tok.append(meta.get("prompt_eval_count")); out_tok.append(meta.get("eval_count"))

        outputs.append({
            "tweet": tweet,
            "gold_sentiment": gold,
            "pred_sentiment": pred,
            "model_output": out,
            "latency_s": round(dt,3),
            "prompt_eval_count": meta.get("prompt_eval_count"),
            "eval_count": meta.get("eval_count"),
            "retrieved_idx": idxs
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

    print(f"\n=== RETRIEVAL FEW-SHOT (k={args.k}, lambda_div={args.lambda_div}) ===")
    print(rep)
    print({**stats, **token_stats})

    # exports
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"ret{k_for_tag}_{str(args.lambda_div).replace('.','_')}".replace('__','_')

    out_csv = DATA_DIR / f"preds_fewshot_{tag}.csv"
    pd.DataFrame(outputs)[["tweet","gold_sentiment","pred_sentiment",
                           "latency_s","prompt_eval_count","eval_count","retrieved_idx"]].to_csv(out_csv, index=False)

    out_jsonl = DATA_DIR / f"preds_fewshot_{tag}.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in outputs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    rows_full=[]
    for r in outputs:
        mo = r["model_output"] if isinstance(r["model_output"], dict) else {}
        rows_full.append({
            "tweet": r["tweet"], "gold_sentiment": r["gold_sentiment"], "pred_sentiment": r["pred_sentiment"],
            "pred_is_claim": mo.get("is_claim", None),
            "pred_topics": json.dumps(mo.get("topics", []), ensure_ascii=False),
            "pred_urgence": mo.get("urgence",""), "pred_incident": mo.get("incident",""),
            "pred_confidence": mo.get("confidence", None),
            "latency_s": r["latency_s"],
            "prompt_eval_count": r.get("prompt_eval_count"),
            "eval_count": r.get("eval_count"),
        })
    out_csv_full = DATA_DIR / f"preds_fewshot_full_{tag}.csv"
    pd.DataFrame(rows_full).to_csv(out_csv_full, index=False)

    print("Fichiers écrits :")
    print(" -", out_csv)
    print(" -", out_jsonl)
    print(" -", out_csv_full)

if __name__ == "__main__":
    main()
