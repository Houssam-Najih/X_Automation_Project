import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# ==========================
# PALETTE FREE
# ==========================
FREE_PRIMARY = "#E60000"   
FREE_SECONDARY = "#FF7043"
FREE_DARK = "#B71C1C"

FREE_SENTIMENT_MAP = {
    "neg": FREE_PRIMARY,
    "neu": "#FFB300",
    "pos": "#4CAF50",
}

FREE_URGENCE_MAP = {
    "basse": "#4CAF50",
    "moyenne": "#FFB300",
    "haute": FREE_PRIMARY,
}

FREE_HEATMAP = ["#FFE5E5", "#FF9999", FREE_PRIMARY, "#8B0000"]


def apply_free_layout(fig):
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def minutes_to_dhm(m):
    """
    Convertit un nombre de minutes en 'X jours Y h Z min'
    """
    if m is None:
        return "N/A"

    total_minutes = int(round(m))

    jours = total_minutes // (24 * 60)
    reste = total_minutes % (24 * 60)
    heures = reste // 60
    minutes = reste % 60

    morceaux = []
    if jours > 0:
        morceaux.append(f"{jours} jour{'s' if jours > 1 else ''}")
    if heures > 0:
        morceaux.append(f"{heures} h")
    if minutes > 0 or not morceaux:
        morceaux.append(f"{minutes} min")

    return " ".join(morceaux)


# --------------------------------------------------
# CONFIG PAGE
# --------------------------------------------------
st.set_page_config(
    page_title="Free - Analyse SAV Twitter",
    page_icon="📡",
    layout="wide"
)

# --------------------------------------------------
# CHARGEMENT DES DONNÉES CLIENT
# --------------------------------------------------
@st.cache_data
def load_data(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()

    # 1) Lecture du CSV
    df = pd.read_csv(file, encoding="utf-8")

    # 2) Nettoyage des noms de colonnes
    df.columns = df.columns.str.strip()

    # 3) Vérification de la colonne created_at
    if "created_at" not in df.columns:
        st.error("La colonne 'created_at' est introuvable dans le CSV.")
        st.stop()

    # 4) Conversion en datetime
    created = pd.to_datetime(
        df["created_at"].astype(str),
        errors="coerce",
        utc=True
    )

    # Si tout est NaT → problème de format de date
    if created.isna().all():
        st.error("Impossible de parser les dates de 'created_at'. Vérifie le format dans le CSV.")
        st.stop()

    # 5) Colonnes dérivées de la date
    df["created_at"] = created
    df["date"] = created.dt.date
    df["week"] = created.dt.to_period("W").apply(lambda r: r.start_time.date())
    df["day_of_week"] = created.dt.day_name()
    df["hour"] = created.dt.hour

    # 6) Topic principal
    df["topic_main"] = df["topics"].str.extract(r"\['?(.*?)'?\]")[0]

    # 7) Colonnes d’engagement
    for col in ["favorite_count", "retweet_count", "reply_count", "quote_count"]:
        if col not in df.columns:
            df[col] = 0

    df["engagement"] = (
        df["favorite_count"]
        + df["retweet_count"]
        + df["reply_count"]
        + df["quote_count"]
    )

    # 8) ID en string
    df["id"] = df["id"].astype(str)

    return df


# --------------------------------------------------
# CHARGEMENT DES RÉPONSES FREE (2e fichier, sans upload)
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # dossier du projet
REPLIES_CSV_PATH = BASE_DIR / "data" / "reponses_free.csv"


@st.cache_data
def load_replies(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

    # cast en str pour être sûr que ça matche avec df_clients["id"]
    df["id"] = df["id"].astype(str)
    if "in_reply_to" in df.columns:
        df["in_reply_to"] = df["in_reply_to"].astype(str)

    # on garde seulement les vraies réponses
    if "in_reply_to" in df.columns:
        df = df[~df["in_reply_to"].isna()]

    return df


def compute_response_time(df_clients: pd.DataFrame, df_replies: pd.DataFrame):
    """
    df_clients : tweets clients (id, created_at, ...), idéalement df_sav (réclamations filtrées)
    df_replies : tweets de réponse de Free (id, in_reply_to, created_at)
    """

    if df_clients.empty or df_replies.empty:
        return None, None, None

    # On travaille sur des copies pour ne pas modifier les df d'origine
    df_clients = df_clients.copy()
    df_replies = df_replies.copy()

    # Alignement des types d'ID
    df_clients["id"] = df_clients["id"].astype(str)
    df_replies["in_reply_to"] = df_replies["in_reply_to"].astype(str)

    # On ne garde que les réponses qui pointent vers un tweet client présent
    client_ids = df_clients["id"].unique()
    rep = df_replies[df_replies["in_reply_to"].isin(client_ids)].copy()

    if rep.empty:
        return None, None, None

    # Fusion client <-> réponse
    merged = rep.merge(
        df_clients[["id", "created_at"]],
        left_on="in_reply_to",
        right_on="id",
        suffixes=("_reply", "_client")
    )

    # Sécuriser les types de dates AVANT la soustraction
    merged["created_at_reply"] = pd.to_datetime(merged["created_at_reply"], errors="coerce")
    merged["created_at_client"] = pd.to_datetime(merged["created_at_client"], errors="coerce")

    # On enlève les lignes où une des deux dates est NaT
    merged = merged.dropna(subset=["created_at_reply", "created_at_client"])

    if merged.empty:
        return None, None, None

    # Calcul du délai en minutes (maintenant les deux sont bien des datetime)
    merged["delay_minutes"] = (
        merged["created_at_reply"] - merged["created_at_client"]
    ).dt.total_seconds() / 60

    # On enlève les délais négatifs (au cas où)
    merged = merged[merged["delay_minutes"] >= 0]

    if merged.empty:
        return None, None, None

    # Pour chaque tweet client : première réponse de Free (délai minimal)
    first_reply = (
        merged.sort_values("delay_minutes")
        .groupby("in_reply_to")
        .first()
        .reset_index()
    )

    mean_delay = first_reply["delay_minutes"].mean()
    median_delay = first_reply["delay_minutes"].median()

    return first_reply, mean_delay, median_delay


# --------------------------------------------------
# UPLOAD DU FICHIER CLIENT
# --------------------------------------------------
st.sidebar.title("📁 Import du fichier")
uploaded_file = st.sidebar.file_uploader(
    "Importe ton fichier CSV de tweets clients",
    type=["csv"]
)

if uploaded_file is None:
    st.title("📡 Analyse des tweets SAV Free")
    st.info("👈 Merci d'importer un fichier CSV dans la barre latérale pour afficher le tableau de bord.")
    st.stop()

df = load_data(uploaded_file)

if "lang" in df.columns:
    df = df[df["lang"] == "fr"]

# Chargement du fichier des réponses Free
try:
    df_replies = load_replies(REPLIES_CSV_PATH)
except FileNotFoundError:
    df_replies = pd.DataFrame()
    st.sidebar.error(f"⚠️ Fichier réponses introuvable : {REPLIES_CSV_PATH}")

# --------------------------------------------------
# SIDEBAR : PROFIL
# --------------------------------------------------
st.sidebar.markdown("---")
profil = st.sidebar.radio(
    "👤 Profil de vue",
    ["Manager", "Data analyst", "Agent SAV"],
    index=0
)

# --------------------------------------------------
# SIDEBAR : FILTRES
# --------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.title("Filtres")

min_date = df["date"].min()
max_date = df["date"].max()

date_range = st.sidebar.date_input(
    "Période",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(date_range, tuple):
    start_date, end_date = date_range
else:
    start_date = date_range
    end_date = date_range

df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

topics = sorted(df["topic_main"].dropna().unique().tolist())
sentiments = sorted(df["sentiment"].dropna().unique().tolist())
incidents = sorted(df["incident"].dropna().unique().tolist())
urgences = sorted(df["urgence"].dropna().unique().tolist())

topic_options = ["Tous"] + topics
sentiment_options = ["Tous"] + sentiments
incident_options = ["Tous"] + incidents
urgence_options = ["Tous"] + urgences

selected_topic = st.sidebar.selectbox("Thème (topic)", topic_options)
selected_sentiment = st.sidebar.selectbox("Sentiment", sentiment_options)
selected_incident = st.sidebar.selectbox("Type d'incident", incident_options)
selected_urgence = st.sidebar.selectbox("Niveau d'urgence", urgence_options)

if selected_topic != "Tous":
    df = df[df["topic_main"] == selected_topic]

if selected_sentiment != "Tous":
    df = df[df["sentiment"] == selected_sentiment]

if selected_incident != "Tous":
    df = df[df["incident"] == selected_incident]

if selected_urgence != "Tous":
    df = df[df["urgence"] == selected_urgence]

df_sav = df[df["is_claim"] == 1]

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("Analyse des tweets SAV Free")
st.markdown(
    f"Vue actuelle : **{profil}** — tableau de bord filtré selon les options ci-dessus."
)

# --------------------------------------------------
# KPI GLOBAUX (Manager + Data Analyst uniquement)
# --------------------------------------------------
if profil in ["Manager", "Data analyst"]:

    st.subheader("Indicateurs généraux")

    col1, col2, col3 = st.columns(3)

    total_tweets = len(df)
    total_sav = int(df_sav["is_claim"].sum())
    taux_sav = (total_sav / total_tweets * 100) if total_tweets > 0 else 0

    with col1:
        st.metric("Tweets analysés", f"{total_tweets:,}".replace(",", " "))

    with col2:
        st.metric("Tweets SAV (réclamations)", f"{total_sav:,}".replace(",", " "))

    with col3:
        st.metric("Part de tweets SAV (réclamations)", f"{taux_sav:.1f} %")

    st.markdown("---")

# --------------------------------------------------
# CAMEMBERT SENTIMENT (Manager + Data Analyst uniquement)
# --------------------------------------------------
if profil in ["Manager", "Data analyst"]:

    st.subheader("Répartition des sentiments (tweets SAV)")

    if len(df_sav) > 0:
        sent_counts = (
            df_sav.groupby("sentiment")["id"]
            .count()
            .reset_index(name="nb")
        )

        fig_pie_sent = px.pie(
            sent_counts,
            names="sentiment",
            values="nb",
            title="Sentiments des tweets SAV",
            color="sentiment",
            color_discrete_map=FREE_SENTIMENT_MAP,
        )
        fig_pie_sent = apply_free_layout(fig_pie_sent)
        st.plotly_chart(fig_pie_sent, use_container_width=True)
    else:
        st.info("Aucun tweet SAV pour calculer la répartition des sentiments.")

    st.markdown("---")

# --------------------------------------------------
# URGENCES (Tout le monde)
# --------------------------------------------------
st.subheader("Répartition des niveaux d'urgence (tweets SAV)")

if len(df_sav) > 0:
    urg_counts = (
        df_sav.groupby("urgence")["id"]
        .count()
        .reset_index(name="nb")
    )

    fig_urg = px.bar(
        urg_counts,
        x="urgence",
        y="nb",
        color="urgence",
        color_discrete_map=FREE_URGENCE_MAP,
        title="Tweets SAV par niveau d'urgence",
        labels={"urgence": "Niveau d'urgence", "nb": "Tweets SAV"}
    )
    fig_urg = apply_free_layout(fig_urg)
    st.plotly_chart(fig_urg, use_container_width=True)
else:
    st.info("Aucune réclamation pour afficher l'urgence.")

st.markdown("---")

# --------------------------------------------------
# TOP 5 INCIDENTS (Tout le monde)
# --------------------------------------------------
st.subheader("Top 5 des incidents")

if len(df_sav) > 0:
    inc_counts = (
        df_sav.groupby("incident")["id"]
        .count()
        .reset_index(name="nb")
        .sort_values("nb", ascending=False)
        .head(5)
    )

    fig_inc_top5 = px.bar(
        inc_counts,
        x="incident",
        y="nb",
        title="Top 5 incidents les plus fréquents",
        labels={"incident": "Incident", "nb": "Tweets SAV"},
        color_discrete_sequence=[FREE_PRIMARY]
    )
    fig_inc_top5 = apply_free_layout(fig_inc_top5)
    st.plotly_chart(fig_inc_top5, use_container_width=True)
else:
    st.info("Aucune réclamation pour afficher les incidents.")

st.markdown("---")

# --------------------------------------------------
# VOLUME PAR JOUR ET PAR SEMAINE (Tout le monde)
# --------------------------------------------------
st.subheader("Volume de tweets SAV par jour et par semaine")

if len(df_sav) > 0:
    # Volume par jour
    volume_jour = (
        df_sav.groupby("date")["id"]
        .count()
        .reset_index(name="nb_tweets")
        .sort_values("date")
    )

    # Volume par semaine
    volume_semaine = (
        df_sav.groupby("week")["id"]
        .count()
        .reset_index(name="nb_tweets")
        .sort_values("week")
    )

    col_v1, col_v2 = st.columns(2)

    with col_v1:
        fig_vol_jour = px.line(
            volume_jour,
            x="date",
            y="nb_tweets",
            title="Volume quotidien de tweets SAV",
            labels={"date": "Date", "nb_tweets": "Tweets SAV"},
            markers=True,
            color_discrete_sequence=[FREE_PRIMARY]
        )
        fig_vol_jour = apply_free_layout(fig_vol_jour)
        st.plotly_chart(fig_vol_jour, use_container_width=True)

    with col_v2:
        fig_vol_semaine = px.bar(
            volume_semaine,
            x="week",
            y="nb_tweets",
            title="Volume hebdomadaire de tweets SAV",
            labels={"week": "Semaine (lundi)", "nb_tweets": "Tweets SAV"},
            color_discrete_sequence=[FREE_PRIMARY]
        )
        fig_vol_semaine = apply_free_layout(fig_vol_semaine)
        st.plotly_chart(fig_vol_semaine, use_container_width=True)
else:
    st.info("Aucun tweet SAV sur la période sélectionnée.")

st.markdown("---")

# --------------------------------------------------
# HISTOGRAMMES DE VOLUME (heure, jour de semaine)
# --------------------------------------------------
st.subheader("Histogrammes de volume des tweets SAV")

if len(df_sav) > 0:
    col_h1, col_h2 = st.columns(2)

    # Volume par heure de la journée
    with col_h1:
        vol_hour = (
            df_sav.groupby("hour")["id"]
            .count()
            .reset_index(name="nb_tweets")
            .sort_values("hour")
        )
        fig_h_hour = px.bar(
            vol_hour,
            x="hour",
            y="nb_tweets",
            title="Volume de tweets SAV par heure",
            labels={"hour": "Heure", "nb_tweets": "Tweets SAV"},
            color_discrete_sequence=[FREE_PRIMARY]
        )
        fig_h_hour = apply_free_layout(fig_h_hour)
        st.plotly_chart(fig_h_hour, use_container_width=True)

    # Volume par jour de la semaine
    with col_h2:
        vol_dow = (
            df_sav.groupby("day_of_week")["id"]
            .count()
            .reset_index(name="nb_tweets")
        )

        order_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        vol_dow["day_of_week"] = pd.Categorical(
            vol_dow["day_of_week"],
            categories=order_days,
            ordered=True
        )

        vol_dow = vol_dow.sort_values("day_of_week")

        fig_h_dow = px.bar(
            vol_dow,
            x="day_of_week",
            y="nb_tweets",
            title="Volume de tweets SAV par jour de la semaine",
            labels={"day_of_week": "Jour", "nb_tweets": "Tweets SAV"},
            color_discrete_sequence=[FREE_PRIMARY]
        )
        fig_h_dow = apply_free_layout(fig_h_dow)
        st.plotly_chart(fig_h_dow, use_container_width=True)
else:
    st.info("Aucun tweet SAV pour afficher les histogrammes de volume.")

st.markdown("---")
