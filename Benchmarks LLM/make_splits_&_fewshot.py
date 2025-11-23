
import argparse, json, math, sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RNG = 42 

def parse_topics(val):
    """Convertit la colonne 'topics' en liste Python propre."""
    if isinstance(val, list):
        return val
    if pd.isna(val):
        return []
    s = str(val).strip()
    # essaie JSON d'abord
    try:
        x = json.loads(s)
        if isinstance(x, list):
            return x
        return [str(x)]
    except Exception:
        pass
    # fallback: découpe simple sur virgule / point-virgule
    s = s.strip("[]")
    toks = [t.strip() for t in s.replace(";", ",").split(",") if t.strip()]
    return toks

def to_bool(x):
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in {"1", "true", "vrai", "yes", "oui"}

def to_float_or_none(x):
    try:
        f = float(x)
        if math.isnan(f):
            return None
        return f
    except Exception:
        return None

# programme principal

def main(args):
    ROOT = Path(__file__).resolve().parent
    inp = (ROOT / args.input_csv).resolve()
    outdir = (ROOT / args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)

    # Colonnes minimales
    needed = {"full_text", "sentiment"}
    missing = needed - set(df.columns)
    if missing:
        print(f"[ERREUR] Colonnes manquantes: {missing}", file=sys.stderr)
        sys.exit(1)

    # Nettoyages légers
    df = df.dropna(subset=["full_text", "sentiment"]).copy()
    df["sentiment"] = df["sentiment"].astype(str).str.strip()

    # Split stratifié 80/10/10 sur 'sentiment'
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=RNG, stratify=df["sentiment"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=RNG, stratify=temp_df["sentiment"]
    )

    # TEST minimal (sentiment uniquement)
    test_min = test_df[["full_text", "sentiment"]].rename(columns={"full_text": "tweet"})
    p_test_min = outdir / "test_split_stratified.csv"
    test_min.to_csv(p_test_min, index=False)

    # TEST "full"
    full_cols = [
        "full_text", "is_claim", "topics", "sentiment", "urgence", "incident", "confidence"
    ]
    present = [c for c in full_cols if c in test_df.columns]
    test_full = test_df[present].rename(columns={"full_text": "tweet"})
    p_test_full = outdir / "test_split_full.csv"
    test_full.to_csv(p_test_full, index=False)

    # FEW-SHOT équilibré depuis TRAIN+VAL
    pool = pd.concat([train_df, val_df], ignore_index=True)

    # borne équilibrée = taille de la classe minoritaire dans le pool
    counts = pool["sentiment"].value_counts()
    n_classes = counts.shape[0]
    per_class = counts.min()

    # si --cap30, plafonner ~30 exemples au total
    cap_per_class = 30 // n_classes if args.cap30 else per_class
    k = int(min(per_class, cap_per_class))
    if k == 0:
        print("[ERREUR] Impossible d'équilibrer: une classe a 0 exemple dans train+val.", file=sys.stderr)
        sys.exit(2)

    fewshot_df = (
        pool.groupby("sentiment", group_keys=False)
            .apply(lambda g: g.sample(n=k, random_state=RNG))
            .reset_index(drop=True)
    )

    # JSONL few-shot COMPLET
    def row_to_full_Y(r):
        return {
            "sentiment": r["sentiment"],
            "is_claim": to_bool(r["is_claim"]) if "is_claim" in r else False,
            "topics": parse_topics(r["topics"]) if "topics" in r else [],
            "urgence": str(r["urgence"]) if "urgence" in r and pd.notna(r["urgence"]) else "",
            "incident": str(r["incident"]) if "incident" in r and pd.notna(r["incident"]) else "",
            "confidence": to_float_or_none(r["confidence"]) if "confidence" in r else None,
        }

    p_fewshot_full = outdir / "fewshot_balanced_full.jsonl"
    with p_fewshot_full.open("w", encoding="utf-8") as f:
        for _, r in fewshot_df.iterrows():
            rec = {"T": r["full_text"], "Y": row_to_full_Y(r)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    
    def dist(frame, name):
        return name, frame["sentiment"].value_counts().to_dict()

    print("== RÉSUMÉ ==")
    print("Taille totale:", len(df))
    print("Répartition globale:", df["sentiment"].value_counts().to_dict())
    print(dist(train_df, "train"))
    print(dist(val_df,   "val"))
    print(dist(test_df,  "test"))
    print(f"Few-shot équilibré: {k} par classe → total = {k*n_classes}")
    print("Fichiers écrits :")
    print(" -", p_test_min,   "(test minimal sentiment)")
    print(" -", p_test_full,  "(test complet)")
    print(" -", p_fewshot_full, "(exemples few-shot complets)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", required=False,
                    default="data/tweet_labeled_300.csv",
                    help="Chemin du CSV des 300 tweets annotés")
    ap.add_argument("--output-dir", default="data",
                    help="Dossier de sortie")
    ap.add_argument("--cap30", action="store_true",
                    help="Limiter ~30 exemples au total (équilibrés) si possible")
    main(ap.parse_args())
