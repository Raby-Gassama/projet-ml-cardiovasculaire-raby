import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

@st.cache_resource
def train_model():
    # Charger les données (heart.csv doit être dans le même dossier)
    df = pd.read_csv("heart.csv")
    target = "HeartDisease"
    y = df[target]
    X = df.drop(columns=[target])

    # Séparer numériques / catégorielles
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    preprocess = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    # Baseline modèles
    models = {
        "LogReg": LogisticRegression(max_iter=2000),
        "KNN": KNeighborsClassifier(7),
        "SVM": SVC(probability=True),
        "RF": RandomForestClassifier(n_estimators=300, random_state=42)
    }

    best_name = None
    best_auc = -1
    best_pipe = None

    for name, clf in models.items():
        pipe = Pipeline([("preprocess", preprocess), ("model", clf)])
        pipe.fit(X_train, y_train)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_pipe = pipe

    # Optionnel : GridSearch sur le meilleur modèle
    if best_name == "SVM":
        param_grid = {
            "model__C": [0.5, 1, 5],
            "model__gamma": ["scale", 0.05, 0.1],
            "model__kernel": ["rbf"],
        }
        base_est = SVC(probability=True)
    elif best_name == "RF":
        param_grid = {
            "model__n_estimators": [300, 600],
            "model__max_depth": [None, 8, 12],
            "model__min_samples_split": [2, 5],
        }
        base_est = RandomForestClassifier(random_state=42)
    elif best_name == "LogReg":
        param_grid = {
            "model__C": [0.1, 1, 10]
        }
        base_est = LogisticRegression(max_iter=2000)
    else:  # KNN
        param_grid = {"model__n_neighbors": [5, 7, 9]}
        base_est = KNeighborsClassifier()

    pipe = Pipeline([("preprocess", preprocess), ("model", base_est)])
    gs = GridSearchCV(pipe, param_grid, scoring="roc_auc", cv=3, n_jobs=1)
    gs.fit(X_train, y_train)
    best_pipe = gs.best_estimator_

    # On renvoie aussi la liste des colonnes pour le formulaire
    return best_pipe, X

# ================== INTERFACE STREAMLIT ==================

st.title("🫀 Prédiction du risque de maladie cardiovasculaire")
st.write("Application déployée par **Fatoumata Raby Gassama** (M1 Finance Digitale – IA).")
st.markdown("---")

with st.spinner("Entraînement du modèle en cours..."):
    model, X_example = train_model()

st.success("Modèle entraîné et prêt à l'emploi ✅")

st.subheader("➡️ Saisir les informations du patient")

# On construit dynamiquement les widgets en fonction des colonnes du dataset
user_input = {}

for col in X_example.columns:
    if pd.api.types.is_numeric_dtype(X_example[col]):
        # numérique : slider ou number_input
        min_val = float(X_example[col].min())
        max_val = float(X_example[col].max())
        default_val = float(X_example[col].median())
        user_input[col] = st.slider(
            col,
            min_value=min_val,
            max_value=max_val,
            value=default_val,
        )
    else:
        # catégorielle : selectbox
        options = sorted(X_example[col].unique())
        default_val = options[0]
        user_input[col] = st.selectbox(col, options, index=0)

if st.button("Prédire le risque"):
    # Transformer le dictionnaire en DataFrame (une seule ligne)
    input_df = pd.DataFrame([user_input])

    # Probabilité que HeartDisease = 1
    proba = model.predict_proba(input_df)[:, 1][0]
    pred = model.predict(input_df)[0]

    st.write("### Résultat du modèle :")
    st.metric(
        "Probabilité estimée de maladie cardiovasculaire",
        f"{proba*100:.1f} %"
    )

    if proba >= 0.7:
        st.error("⚠️ Risque élevé – dépistage recommandé.")
    elif proba >= 0.4:
        st.warning("⚠️ Risque modéré – à surveiller.")
    else:
        st.success("✅ Risque estimé faible.")

    st.caption("Ce modèle est un outil d'aide à la décision et ne remplace pas un avis médical.")
