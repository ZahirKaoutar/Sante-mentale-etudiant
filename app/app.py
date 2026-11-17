"""Application Streamlit simple pour la prédiction du risque de santé mentale des étudiants.

Cette interface permet :
1. De saisir les caractéristiques d'un étudiant (académiques, hygiène de vie, stress).
2. D'obtenir une prédiction instantanée du niveau de risque (Faible, Modéré, Élevé).
3. De visualiser les probabilités associées et des conseils personnalisés.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuration de la page
st.set_page_config(
    page_title="Santé Mentale Étudiant - Prédiction",
    page_icon="🧠",
    layout="wide"
)

# Chemins
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student_mental_health.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"

RISK_LABELS = {
    0: ("Faible (Low)", "🟢", "success"),
    1: ("Modéré (Medium)", "🟡", "warning"),
    2: ("Élevé (High)", "🔴", "error")
}


@st.cache_resource
def load_or_train_model():
    """Charge le modèle sérialisé ou lance un entraînement s'il n'existe pas encore."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    # Entraînement automatique de secours
    from src.tune import evaluate_final_model
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        best_model, _ = evaluate_final_model(df)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)
        return best_model
    return None


# Titre de l'application
st.title("🧠 Prédiction du Risque de Santé Mentale chez les Étudiants")
st.markdown(
    "Cet outil interactif utilise le Machine Learning pour estimer le niveau de vulnérabilité "
    "psychologique d'un étudiant à partir de son mode de vie, de son stress et de sa situation académique."
)

st.subheader("📋 Informations de l'Étudiant")

with st.form("student_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### 📚 Profil Académique")
        gender = st.selectbox("Genre", ["Female", "Male", "Non-binary"])
        year_of_study = st.slider("Année d'étude (1 à 5)", min_value=1, max_value=5, value=2)
        study_hours = st.slider("Heures d'étude par semaine", min_value=1.0, max_value=60.0, value=25.0, step=0.5)
        gpa = st.slider("Moyenne générale (GPA / 4.0)", min_value=1.0, max_value=4.0, value=3.2, step=0.05)
        attendance = st.slider("Taux de présence aux cours (%)", min_value=20.0, max_value=100.0, value=85.0, step=1.0)

    with col2:
        st.markdown("##### 🛌 Hygiène de Vie & Écran")
        sleep_hours = st.slider("Heures de sommeil par nuit", min_value=3.0, max_value=11.0, value=7.0, step=0.5)
        screen_time = st.slider("Temps d'écran total par jour (h)", min_value=1.0, max_value=15.0, value=6.0, step=0.5)
        social_media = st.slider("Temps sur réseaux sociaux (h)", min_value=0.0, max_value=12.0, value=3.0, step=0.5)
        physical_activity = st.selectbox("Activité physique", ["Moderate", "Low", "High"])
        diet_quality = st.selectbox("Qualité de l'alimentation", ["Average", "Poor", "Good"])

    with col3:
        st.markdown("##### 💼 Environnement & Soutien")
        financial_stress = st.slider("Stress financier (1 = Faible, 10 = Extrême)", min_value=1.0, max_value=10.0, value=5.0, step=0.5)
        family_support = st.slider("Soutien familial (1 = Faible, 10 = Très fort)", min_value=1.0, max_value=10.0, value=7.0, step=0.5)
        part_time_job = st.selectbox("Travail à temps partiel", ["No", "Yes"])
        mental_history = st.selectbox("Antécédents de santé mentale", ["No", "Yes"])

    # Cohérence métier
    if social_media > screen_time:
        st.warning("⚠️ Attention : Le temps passé sur les réseaux sociaux dépasse le temps d'écran total. Une correction sera appliquée.")

    submit = st.form_submit_button("🔍 Évaluer le Risque de Santé Mentale", use_container_width=True)

if submit:
    # Construction du DataFrame d'entrée
    input_data = pd.DataFrame([{
        "Gender": gender,
        "Year_of_Study": year_of_study,
        "Sleep_Hours": sleep_hours,
        "Study_Hours_Per_Week": study_hours,
        "GPA": gpa,
        "Social_Media_Hours": social_media if social_media <= screen_time else np.nan,
        "Screen_Time_Hours": screen_time,
        "Physical_Activity": physical_activity,
        "Financial_Stress": financial_stress,
        "Family_Support": family_support,
        "Attendance_Rate": attendance,
        "Part_Time_Job": part_time_job,
        "Diet_Quality": diet_quality,
        "Previous_Mental_Health_History": mental_history
    }])

    model = load_or_train_model()

    if model is None:
        st.error("Impossible de charger ou d'entraîner le modèle.")
    else:
        prediction = model.predict(input_data)[0]
        label, icon, alert_type = RISK_LABELS.get(prediction, ("Inconnu", "❓", "info"))

        # Affichage du résultat
        st.markdown("---")
        st.subheader("🎯 Résultat de l'Évaluation")

        if alert_type == "success":
            st.success(f"### {icon} Niveau de Risque Prédit : **{label}**")
        elif alert_type == "warning":
            st.warning(f"### {icon} Niveau de Risque Prédit : **{label}**")
        else:
            st.error(f"### {icon} Niveau de Risque Prédit : **{label}**")

        # Probabilités si disponibles
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(input_data)[0]
            p_col1, p_col2, p_col3 = st.columns(3)
            p_col1.metric("Risque Faible (Low)", f"{probas[0]:.1%}")
            p_col2.metric("Risque Modéré (Medium)", f"{probas[1]:.1%}")
            p_col3.metric("Risque Élevé (High)", f"{probas[2]:.1%}")

        # Recommandations personnalisées
        st.markdown("#### 💡 Conseils et Recommandations :")
        if prediction == 0:
            st.info(
                "- **Maintenir l'équilibre actuel** : Votre hygiène de vie et votre soutien sont de très bons remparts.\n"
                "- Continuer à pratiquer une activité physique régulière et veiller à des nuits de sommeil complètes."
            )
        elif prediction == 1:
            st.warning(
                "- **Surveillance active** : Des signes de tension ou de fatigue sont perceptibles.\n"
                "- Prioriser au moins 7 heures de sommeil par nuit.\n"
                "- Réduire l'exposition aux écrans en soirée et solliciter du soutien auprès de vos proches ou tuteurs."
            )
        else:
            st.error(
                "- **Accompagnement recommandé** : Plusieurs facteurs de stress importants sont combinés.\n"
                "- Nous vous recommandons de prendre contact avec le service médical ou psychologique universitaire.\n"
                "- Se renseigner sur les aides financières aux étudiants si le stress financier est prépondérant."
            )
