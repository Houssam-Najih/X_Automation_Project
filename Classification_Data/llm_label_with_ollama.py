# -*- coding: utf-8 -*-
"""CLI utilitaire pour lancer la classification des tweets Free."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.config.settings import get_settings
from src.models import TweetPayload
from src.services import OllamaClassifier
from src.services.kpi import compute_kpis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classification LLM des tweets Free.")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Taille d'échantillon (None = totalité).",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Chemin CSV d'entrée (override).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Chemin CSV de sortie (override).",
    )
    return parser.parse_args()


def build_payloads(df: pd.DataFrame) -> list[TweetPayload]:
    payloads: list[TweetPayload] = []
    for _, row in df.iterrows():
        tweet_id = row.get("id")
        if pd.isna(tweet_id):
            continue
        created_at = row.get("created_at")
        text_raw = row.get("text_raw") or row.get("full_text") or row.get("text_clean_llm")
        payloads.append(
            TweetPayload(
                id=str(tweet_id),
                created_at=pd.to_datetime(created_at, errors="coerce").isoformat()
                if pd.notna(created_at)
                else None,
                text_raw=text_raw,
            )
        )
    return payloads


def main() -> None:
    args = parse_args()
    settings = get_settings()
    dataset_cfg = settings.dataset

    source_path = args.source or (dataset_cfg.base_dir / dataset_cfg.source_csv)
    output_path = args.output or (dataset_cfg.base_dir / dataset_cfg.output_csv)

    if not source_path.exists():
        print(f" Fichier introuvable: {source_path}")
        sys.exit(1)

    print(f" Lecture: {source_path}")
    df_full = pd.read_csv(source_path, engine="python", sep=dataset_cfg.csv_separator)

    if args.sample:
        df = df_full.head(args.sample).copy()
        print(f" Échantillon: {len(df)} lignes")
    else:
        df = df_full.copy()
        print(f" Traitement complet: {len(df)} lignes")

    payloads = build_payloads(df)
    if not payloads:
        print("Aucun tweet valide dans le dataset fourni.")
        sys.exit(0)

    classifier = OllamaClassifier(settings=settings)
    out_df = classifier.classify_tweets(payloads)

    cols_to_drop = [c for c in dataset_cfg.columns_to_drop if c in out_df.columns]
    if cols_to_drop:
        out_df = out_df.drop(columns=cols_to_drop)

    priority_columns = [
        "id",
        "created_at",
        "text_clean",
        "is_claim",
        "topics",
        "sentiment",
        "urgence",
        "incident",
        "confidence",
    ]
    final_columns = priority_columns + [
        c for c in out_df.columns if c not in priority_columns
    ]
    out_df = out_df[final_columns]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False, sep=",", encoding="utf-8")
    print(f" Export: {output_path}")

    kpi = compute_kpis(out_df)
    print(f" Réclamations détectées: {kpi.claim_rate:.1f}% ({kpi.total_claims})")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Script CLI pour lancer la classification batch côté data-engineering."""

import requests, pandas as pd, orjson, time, sys, csv, re
from pathlib import Path
import unicodedata

# ====== CONFIG ======
MODEL = "mistral"                           # ou "llama3.1:8b-instruct", "qwen2.5:7b-instruct"
SOURCE_CSV = "Data/free_tweet_export.cleaned.llm.csv"  # fichier complet avec 3500 tweets
SAMPLE_N = None                            # None pour traiter tout le fichier
SAVE_SAMPLE_AS = None # None pour ne pas enregistrer l'échantillon
OUTPUT_CSV = "Data/free_tweet_classified_clean.csv"  # Fichier de sortie nettoyé et structuré
TEXT_COL = "text_clean_llm"                 # colonne source pour le texte
CSV_SEPARATOR = ","                         # séparateur utilisé dans les CSV source (virgule pour free_tweet_export)
BATCH_SIZE = 50                             # taille des lots pour traitement (échantillonnage de 50)
HTTP_TIMEOUT = 300                          # secondes (au lieu de 120)
MAX_RETRIES = 3                             # nombre d'essais par tweet
BACKOFFS = [1, 3, 7]                        # secondes entre essais
API_URL = "http://localhost:11434/api/chat"

# Colonnes à supprimer du fichier de sortie
COLUMNS_TO_DROP = ["hashtags_list", "retweeted", "favorited", "views_count", "reply_count", "media_tags"]

# Session HTTP réutilisée (perf + keep-alive)
SESSION = requests.Session()

# Répertoire du script pour gérer les chemins relatifs même si on lance python depuis ailleurs
BASE_DIR = Path(__file__).resolve().parent

# 1) prompt 
SYSTEM_PROMPT = """

Tu es un classifieur francophone pour des tweets adressés à un opérateur télécom (Free).
Ta tâche: décider si le message est une RÉCLAMATION et attribuer des étiquettes.
Réponds STRICTEMENT en JSON compact, UNE SEULE LIGNE, sans aucun texte en dehors du JSON.

Champs JSON attendus:
- is_claim: 0 ou 1
- topics: liste parmi ["fibre","dsl","wifi","tv","mobile","facture","activation","resiliation","autre"] (multi-étiquettes possible)
- sentiment: "neg" | "neu" | "pos"
- urgence: "haute" | "moyenne" | "basse"
- incident: l'une de ["facturation","incident_reseau","livraison","information","processus_sav","autre"]
- confidence: float entre 0 et 1 (confiance globale de ta classification)

Définition de 'incident':
- "facturation" : factures, prélèvements, remboursements, erreurs de montant.
- "incident_reseau" : panne, coupure, débit, TV KO, fibre/DSL/WiFi indisponible, réseau mobile HS.
- "livraison" : livraison/expédition de box/SIM/équipement, point relais, suivi colis.
- "information" : annonces, promos, news, partages de liens, questions générales sans demande d'aide concrète.
- "processus_sav" : lenteur de réponse, absence de retour, mauvaise prise en charge, qualité d’échange avec le SAV.
- "autre" : tout ce qui ne rentre pas clairement dans les catégories précédentes.

Règles:
- "Réclamation" = demande d'aide/dysfonctionnement/problème (panne, débit, facture, activation...).
- Les retweets/annonces/promo sans demande explicite -> is_claim = 0 (incident="information").
- Si le sujet n'entre pas clairement dans une catégorie, ajoute "autre".
- Si is_claim=1, ne laisse jamais topics vide (ajoute "autre" au besoin).
- Hashtags et emojis aident le contexte mais ne doivent pas polluer le JSON.
- Si ambigu, sois conservateur: is_claim=0, topics=["autre"], incident="autre", confidence plus faible.

Règle spécifique : si un tweet indique qu’un plan/événement (ex: stream, live, visite) a été annulé ou fortement dégradé à cause de la connexion Free (ex: “connexion instable”, “ça coupe”, “le stream tient pas”, "je voulais faire le Panthéon, j'ai changé d'avis.."), alors:
- is_claim = 1
- incident = "incident_reseau"
- topics = ["wifi"] si usage domestique/Freebox, sinon ["mobile"] s’il s’agit d’un stream en mobilité ; si doute, ["wifi"]
- sentiment = "neg"
- urgence = "haute" si l’activité était prévue en direct (stream/live) ; sinon "moyenne"
- confidence ≥ 0.8

"""

SYSTEM_PROMPT += """

EXEMPLES (OBLIGATOIRE) :
Tweet: "@redhairsan @freebox @free ca beug aussi chez toi ? t’habites où ?"
Réponse: {"is_claim":1,"topics":["mobile"],"sentiment":"neg","urgence":"moyenne","incident":"incident_reseau","confidence":0.9}

Tweet: "coucou @free, ça serait sympa un peu de couverture réseau en côte d'armor dans le futur, on enchaine les zones blanches ici 😅"
Réponse: {"is_claim":1,"topics":["mobile"],"sentiment":"neg","urgence":"moyenne","incident":"incident_reseau","confidence":0.85}

Tweet: "dites @free, c’est bientôt fini votre cinéma avec les pages perso ?"
Réponse: {"is_claim":1,"topics":["autre"],"sentiment":"neg","urgence":"moyenne","incident":"incident_reseau","confidence":0.85}

Tweet: "@free à la rentrée je passe chez orange après 17 ans chez vous"
Réponse: {"is_claim":1,"topics":["resiliation"],"sentiment":"neg","urgence":"moyenne","incident":"autre","confidence":0.9}

Tweet: "@freenewsactu je suis depuis plusieurs années free pour delta ont peux aussi aller voir ailleurs si vous préférez @free"
Réponse: {"is_claim":1,"topics":["resiliation"],"sentiment":"neg","urgence":"moyenne","incident":"autre","confidence":0.9}

Tweet: "@free bonjour, à qui faut il s’adresser pour demandé la freebox pop avec le wifi 7 ?"
Réponse: {"is_claim":0,"topics":["autre"],"sentiment":"neu","urgence":"basse","incident":"information","confidence":0.9}

Tweet: "@freebox bonjour, est-ce qu'on peut annuler l'option TV by canal sur une revolution ?"
Réponse: {"is_claim":0,"topics":["tv"],"sentiment":"neu","urgence":"basse","incident":"information","confidence":0.9}

Tweet: "@free_1337 @freebox bonjour, les rdv pour installer la fibre c'est en combien de temps ? j'ai la fibre disponible dans le quartier et j'ai besoin de m'organiser ..."
Réponse: {"is_claim":0,"topics":["fibre"],"sentiment":"neu","urgence":"basse","incident":"information","confidence":0.9}

"""

SYSTEM_PROMPT += """

PRIORITÉ DES RÈGLES (OBLIGATOIRE) :
1) Applique d’abord les règles spécifiques ci-dessous.
2) Si plusieurs règles s’appliquent, choisis la plus spécifique.
3) Tu dois toujours renvoyer un JSON valide, une seule ligne.

DÉTECTION EXPLICITE DES INTERPELLATIONS SAV (OBLIGATOIRE) :
Si le tweet contient au moins une @mention de @free ou @freebox ET au moins un des motifs suivants
(variantes/accents tolérés, casse indifférente) :
- "allo", "allô", "répondez", "repondez", "réponse svp", "reponse svp"
- "toujours pas de réponse", "vous jouez à quoi", "réagissez", "reagissez"
- "quelqu'un ?", "quelqu’un ?", "vous répondez ?"
ALORS classifier :
- is_claim = 1
- incident = "processus_sav"
- sentiment = "neg" (sauf si le ton est explicitement neutre/positif)
- topics = ["autre"] (sauf si un sujet technique clair est mentionné)
- urgence = "moyenne" par défaut (mettre "haute" si "urgent", "immédiat", "maintenant")
- confidence ≥ 0.8

SARCASME/IRONIE (OBLIGATOIRE) :
Si une formule positive apparente est contredite par des marques négatives ou une indisponibilité (ex. « ça fonctionne super bien… c’est abusé », « génial… indisponible »),
ET que le contenu décrit une panne/lenteur/indisponibilité :
- is_claim = 1
- incident = "incident_reseau"
- sentiment = "neg"
- topics = au moins l’un de ["fibre","dsl","wifi","tv","mobile"] selon le contexte ; à défaut ["fibre"]
- urgence = "moyenne"
- confidence ≥ 0.75

RÉCLAMATION IMPLICITE (OBLIGATOIRE) :
Si le message décrit ou suggère un dysfonctionnement réseau/service, même sous forme de question, comparaison ou ironie, ex. :
- "ça beug", "bug", "panne", "plus de [tv/internet]", "ça coupe", "zones blanches", "pas de réseau", "débit pourri", "HS"
ALORS is_claim=1, incident="incident_reseau", sentiment="neg", urgence="moyenne" (ou "haute" si impact immédiat), topics selon le contexte :
- "tv" si TV mentionnée ; "mobile" si couverture/réseau/antenne ; "wifi" si usage domestique non précisé ;
- "fibre"/"dsl" si la techno est mentionnée.

MENACE DE RÉSILIATION (OBLIGATOIRE) :
Si le message indique vouloir/quitter/changer d’opérateur ("je passe chez [X]", "on va ailleurs", "résilier"),
ALORS is_claim=1, sentiment="neg", topics=["resiliation"], incident="autre" (ou "processus_sav" si le SAV est mis en cause), urgence="moyenne", confidence ≥ 0.8.

DEMANDE D’INFO NEUTRE (OBLIGATOIRE) :
Si le message demande une information commerciale/administrative/questions (ex. "bonjour, à qui faut il s’adresser ?", "peut-on...?", "bonjour, est-ce qu'on peut annuler...?", " quel est le délai actuel pour revenir... ?"," c'est en combien de temps ? " ,"en combien de temps ?") SANS signaler de problème,
ALORS is_claim=0, incident="information", topics=["autre"], sentiment="neu", urgence="basse", confidence ≥ 0.7.

CAS NON RÉCLAMATIONS (OBLIGATOIRE) :
- Retweet d’annonce, partage d’article, promo, vœux, félicitations, communiqué ⇒ is_claim=0, incident="information".
- Un tweet constitué uniquement de hashtags/émoticônes sans demande ⇒ is_claim=0 (sauf si un hashtag indique explicitement une panne, ex. #outage avec un lieu/verbe d’indisponibilité).

ANTI-FAUX POSITIFS (OBLIGATOIRE) :
Si le message contient des patrons purement informationnels ("à qui s’adresser", "peut-on", "est-ce qu’on peut", "quel est le délai", "en combien de temps") et n’exprime ni mécontentement ni dysfonctionnement, classifier is_claim=0.

LEXIQUE (AIDE INTERNE, NON BLOQUANT) :
mots_problème ≈ {"beug","bug","bugue","panne","coupure","ça coupe","plus de tv","pas de réseau","zone blanche","hors service","HS","débit pourri","instable","down"}

FORMAT DE SORTIE (OBLIGATOIRE) :
Une seule ligne JSON, strictement conforme :
{"is_claim":0|1,"topics":[...],"sentiment":"neg|neu|pos","urgence":"haute|moyenne|basse","incident":"facturation|incident_reseau|livraison|information|processus_sav|autre","confidence":0.x}

""" 

USER_TEMPLATE = """Tweet:
{tweet}

Réponds en JSON (une seule ligne), exemple:
{{"is_claim":1,"topics":["fibre"],"sentiment":"neg","urgence":"haute","incident":"incident_reseau","confidence":0.82}}"""

# ====== FEW-SHOT (exemples) ======
FEW_SHOTS = [ 

]
# ==================================

 
def chat_ollama_batch(model: str, prompts: list[str]) -> list[str]:
    """Appelle Ollama (chat) séquentiellement avec retries/backoff. Retourne les réponses brutes (strings)."""
    out = []
    for p in prompts:
        # messages = system + few-shots + user (inchangé)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for u_text, y_json in FEW_SHOTS:
            messages.append({"role": "user", "content": u_text})
            messages.append({"role": "assistant", "content": y_json})
        messages.append({"role": "user", "content": p})

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.1}
        }

        last_err = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                r = SESSION.post(API_URL, json=payload, timeout=HTTP_TIMEOUT)
                r.raise_for_status()
                data = r.json()
                content = data.get("message", {}).get("content", "").strip()
                if content:
                    out.append(content)
                    break
                else:
                    last_err = "empty_content"
                    raise RuntimeError("Empty content from model")
            except Exception as e:
                last_err = str(e)
                # backoff si on n'a pas épuisé les essais
                if attempt < MAX_RETRIES:
                    time.sleep(BACKOFFS[min(attempt-1, len(BACKOFFS)-1)])
                else:
                    # on renvoie un JSON d'erreur minimal pour ne pas bloquer le pipeline
                    safe_err = last_err.replace('"', "'")
                    out.append(
                        '{"is_claim":0,"topics":["autre"],"sentiment":"neu","urgence":"basse","incident":"autre","confidence":0.0,"_error":"%s"}'
                        % safe_err
                    )
        # fin for attempt
    return out


def clean_text(text: str) -> str:
    """
    Nettoie le texte en supprimant :
    - Les emojis
    - Les hashtags (#mot)
    - Les caractères spéciaux (garde uniquement lettres, chiffres, espaces et ponctuation de base)
    - Normalise les espaces multiples
    """
    if not isinstance(text, str):
        return ""
    
    # Supprimer les emojis
    # Méthode 1: Supprimer les caractères Unicode emoji
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    text = emoji_pattern.sub('', text)
    
    # Supprimer les hashtags (#mot)
    text = re.sub(r'#\w+', '', text)
    
    # Supprimer les mentions @ (optionnel, mais on les garde pour le contexte)
    # text = re.sub(r'@\w+', '', text)
    
    # Supprimer les URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Garder uniquement lettres, chiffres, espaces, apostrophes, tirets et ponctuation de base
    # Permet les caractères accentués français
    text = re.sub(r'[^\w\s\'-.,!?;:àâäéèêëïîôöùûüÿçÀÂÄÉÈÊËÏÎÔÖÙÛÜŸÇ]', '', text)
    
    # Normaliser les espaces multiples
    text = re.sub(r'\s+', ' ', text)
    
    # Supprimer les espaces en début et fin
    text = text.strip()
    
    return text

def _norm_incident(x: str) -> str:
    """Normalise la valeur incident vers le set autorisé (avec alias fréquents)."""
    allowed = {"facturation","incident_reseau","livraison","information","processus_sav","autre"}
    s = str(x).strip().lower().replace(" ", "_")
    aliases = {
        "reseau": "incident_reseau",
        "réseau": "incident_reseau",
        "incident": "incident_reseau",
        "sav": "processus_sav",
        "service_apres_vente": "processus_sav",
        "service_après_vente": "processus_sav",
        "support": "processus_sav",
        "facture": "facturation",
        "info": "information",
        "livraisons": "livraison"
    }
    s = aliases.get(s, s)
    return s if s in allowed else "autre"

def parse_json_line(s: str) -> dict:
    """Parse robuste de la ligne JSON renvoyée par le LLM."""
    try:
        return orjson.loads(s)
    except Exception:
        try:
            start = s.find("{"); end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                return orjson.loads(s[start:end+1])
        except Exception:
            pass
    return {
        "is_claim": 0,
        "topics": ["autre"],
        "sentiment": "neu",
        "urgence": "basse",
        "incident": "autre",
        "confidence": 0.0,
        "_parse_error": True
    }

def main():
    # 0) Lecture du fichier source complet
    source_path = BASE_DIR / SOURCE_CSV
    if not source_path.exists():
        print(f"Fichier introuvable: {source_path}")
        sys.exit(1)

    # Détection automatique du séparateur
    try:
    df_full = pd.read_csv(source_path, engine="python", sep=CSV_SEPARATOR)
    except Exception:
        # Essayer avec point-virgule si la virgule échoue
        try:
            df_full = pd.read_csv(source_path, engine="python", sep=";")
            print("  Séparateur détecté: point-virgule (;)")
        except Exception:
            # Essayer avec virgule
            df_full = pd.read_csv(source_path, engine="python", sep=",")
            print("  Séparateur détecté: virgule (,)")
    
    if TEXT_COL not in df_full.columns:
        raise SystemExit(f"Colonne {TEXT_COL} introuvable dans {SOURCE_CSV}. Colonnes disponibles: {list(df_full.columns)}")

    # 0bis) Échantillonnage (dans ce même script)
    if SAMPLE_N is not None:
        df = df_full.head(int(SAMPLE_N)).copy()
        if SAVE_SAMPLE_AS:
            df.to_csv(BASE_DIR / SAVE_SAMPLE_AS, index=False, encoding="utf-8", sep=CSV_SEPARATOR)
            print(f" Échantillon sauvegardé: {SAVE_SAMPLE_AS} ({len(df)} lignes)")
        else:
            print(f" Échantillon en mémoire: {len(df)} lignes (non sauvegardé)")
    else:
        df = df_full.copy()
        print(f" Traitement de tout le fichier: {len(df)} lignes")

    # 0ter) Nettoyage du texte - création de la colonne text_clean
    print("🧹 Nettoyage des textes...")
    df['text_clean'] = df[TEXT_COL].astype(str).apply(clean_text)
    print(f"✅ Colonne text_clean créée")

    # 1) Préparation des textes et inférences (utiliser text_clean pour la classification)
    texts = df['text_clean'].astype(str).tolist()
    results = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        prompts = [USER_TEMPLATE.format(tweet=t) for t in batch]
        responses = chat_ollama_batch(MODEL, prompts)

        for raw in responses:
            obj = parse_json_line(raw)
            # normalisation minimale et robuste
            obj["topics"] = [str(t).lower() for t in obj.get("topics", [])]
            obj["sentiment"] = str(obj.get("sentiment", "neu")).lower()
            obj["urgence"]   = str(obj.get("urgence", "basse")).lower()
            obj["incident"]  = _norm_incident(obj.get("incident", "autre"))
            try:
                obj["is_claim"] = int(obj.get("is_claim", 0))
            except Exception:
                obj["is_claim"] = 0
            try:
                obj["confidence"] = float(obj.get("confidence", 0.0))
            except Exception:
                obj["confidence"] = 0.0
            results.append(obj)

        print(f"Traités: {min(i+BATCH_SIZE, len(texts))}/{len(texts)}")

    # 2) Fusion des résultats de classification
    classification_df = pd.DataFrame(results)
    out_df = pd.concat([df, classification_df], axis=1)
    
    # 3) Nettoyage et restructuration du DataFrame de sortie
    print("🔧 Restructuration des données...")
    
    # Supprimer les colonnes indésirables (si elles existent)
    columns_to_drop = [col for col in COLUMNS_TO_DROP if col in out_df.columns]
    if columns_to_drop:
        out_df = out_df.drop(columns=columns_to_drop)
        print(f" Colonnes supprimées: {columns_to_drop}")
    
    # Réorganiser les colonnes pour avoir un ordre logique
    # Colonnes principales en premier, puis les métadonnées, puis les classifications
    priority_columns = ['id', 'created_at', 'text_clean', 'is_claim', 'topics', 
                       'sentiment', 'urgence', 'incident', 'confidence']
    
    # Créer la liste finale des colonnes
    final_columns = []
    for col in priority_columns:
        if col in out_df.columns:
            final_columns.append(col)
    
    # Ajouter les autres colonnes (sauf celles déjà ajoutées)
    for col in out_df.columns:
        if col not in final_columns:
            final_columns.append(col)
    
    out_df = out_df[final_columns]
    
    # 4) Export du fichier nettoyé
    output_path = BASE_DIR / OUTPUT_CSV
    
    # S'assurer que le répertoire parent existe
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Si on ne peut pas créer le répertoire, utiliser le répertoire courant
        output_path = BASE_DIR / Path(OUTPUT_CSV).name
    
    # Si le fichier existe et est verrouillé, utiliser un nom alternatif
    import time
    if output_path.exists():
        try:
            # Tester si on peut ouvrir le fichier en mode append (test de verrouillage)
            test_file = open(output_path, 'a')
            test_file.close()
            # Si on arrive ici, le fichier n'est pas verrouillé, on peut le supprimer
            output_path.unlink()
        except (OSError, PermissionError, IOError):
            # Fichier verrouillé, utiliser un nom alternatif
            print(f"  Le fichier {output_path.name} est verrouillé, utilisation d'un nom alternatif")
            timestamp = int(time.time())
            output_path = BASE_DIR / f"{output_path.stem}_{timestamp}.csv"
    
    # Utiliser pandas.to_csv qui gère mieux les chemins
    try:
        # Convertir en string pour pandas
        output_str = str(output_path)
        out_df.to_csv(output_str, index=False, sep=',', encoding='utf-8')
        print(f" Fichier nettoyé et structuré écrit: {output_path}")
        print(f"   Nombre de lignes: {len(out_df)}")
        print(f"   Nombre de colonnes: {len(out_df.columns)}")
    except Exception as e:
        # Si pandas échoue, essayer avec le répertoire temp
        import tempfile
        try:
            temp_dir = Path(tempfile.gettempdir())
            temp_output = temp_dir / output_path.name
            out_df.to_csv(str(temp_output), index=False, sep=',', encoding='utf-8')
            print(f" Écrit dans le répertoire temporaire: {temp_output}")
            print(f"   Copiez le fichier vers: {output_path}")
        except Exception as e2:
            print(f" Erreur lors de l'écriture du fichier: {e}")
            print(f"   Tentative dans le répertoire temporaire a aussi échoué: {e2}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    if "is_claim" in out_df.columns:
        rate = float((out_df["is_claim"] == 1).mean())
        print(f"Réclamations détectées: {rate:.1%}")

if __name__ == "__main__":
    main()
