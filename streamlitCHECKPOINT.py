import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import os
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Charger le dataset
df = pd.read_csv("D:/DOWNNLOAD/Financial_inclusion_dataset.csv")

# Initialiser le dataframe propre (df_clean) en clonant le dataframe original
df_clean = df.copy()

st.sidebar.title("Sommaire")
pages = ["Contexte du projet", "Exploration des données", "Analyse de données", "Modélisation", "Prédiction"]
page = st.sidebar.radio("Aller vers la page :", pages)

if page == pages[0]:
    st.write("### Accueil")
    st.image("D:/DOWNNLOAD/bank.jpg")
    st.write("### Contexte du projet")
    st.write("Ce projet s'inscrit dans le cadre de l'Inclusion financière en Afrique.")
    st.write("L'objectif est de prédire quelles personnes sont les plus susceptibles d'avoir ou d'utiliser un compte bancaire.")
    st.write("Nous avons à notre disposition le fichier Financial_inclusion_dataset.csv...")
    st.write("qui contient des informations démographiques et les services financiers")
    st.write("utilisés par environ 33 600 personnes en Afrique de l'OUEST")
    st.write("Nous explorerons ce dataset, puis l'analyserons visuellement,")
    st.write("et enfin nous implémenterons des modèles de Machine Learning.")
    st.write("pour pouvoir faire des prédictions")
if page == pages[1]:
    st.write("### Exploration des données")
    st.dataframe(df.head())
    st.write("Dimensions du dataframe :")
    st.write(df.shape)
    if st.checkbox("Afficher les valeurs manquantes"):
        st.write(df.isna().sum())
    if st.checkbox("Gérer les valeurs aberrantes : remplacer la valeur aberrante par la médiane de la variable"):
        for col in df_clean.select_dtypes(include=['float64', 'int64']).columns:
            median = df_clean[col].median()
            std = df_clean[col].std()
            outliers = (df_clean[col] - median).abs() > 3 * std
            df_clean.loc[outliers, col] = median
        st.write("Valeurs aberrantes remplacées par la médiane.")
        st.dataframe(df_clean.head())
    if st.checkbox("Afficher les doublons"):
        st.write(f"Nombre de doublons : {df_clean.duplicated().sum()}")
        if st.button("Supprimer les doublons"):
            df_clean = df_clean.drop_duplicates()
            st.write("Doublons supprimés.")
            st.write(f"Nombre de doublons après suppression : {df_clean.duplicated().sum()}")
            st.dataframe(df_clean.head())

elif page == pages[2]:
    st.write("### Analyse de données")
    st.write("#### Statistiques descriptives")
    st.write(df_clean.describe())
    st.write("#### Visualisations")
    numeric_columns = df_clean.select_dtypes(include=['float64', 'int64']).columns
    st.write("Distribution des variables numériques")
    for col in numeric_columns:
        st.write(f"Distribution de {col}")
        fig, ax = plt.subplots()
        sns.histplot(df_clean[col], kde=True, ax=ax)
        st.pyplot(fig)
    st.write("Matrice de corrélation")
    numeric_df = df_clean.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    st.write("Relations entre les variables")
    selected_columns = st.multiselect("Choisissez des colonnes pour la visualisation des relations", numeric_columns)
    if len(selected_columns) > 1:
        st.write("Pairplot des colonnes sélectionnées")
        fig = sns.pairplot(df_clean[selected_columns])
        st.pyplot(fig)

elif page == pages[3]:
    st.write("### Modélisation")

    # Préparer les données
    X = df_clean.drop(['uniqueid', 'bank_account'],
                      axis=1)  # Remplacez 'uniqueid' et 'bank_account' par les colonnes appropriées
    y = df_clean['bank_account']  # Variable cible

    # Convertir les colonnes non numériques en numériques (par exemple, en utilisant one-hot encoding)
    X = pd.get_dummies(X)

    # Diviser les données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardiser les caractéristiques
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Sauvegarder le scaler
    joblib.dump(scaler, 'scaler.pkl')
    st.write("Scaler sauvegardé sous 'scaler.pkl'.")

    # Choisir le modèle
    model_name = st.selectbox("Choisissez le modèle", ["Régression Logistique", "KNN", "Arbre de Décision"])

    if model_name == "Régression Logistique":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "KNN":
        k = st.slider("Nombre de voisins (k)", min_value=1, max_value=20, value=5)
        model = KNeighborsClassifier(n_neighbors=k)
    elif model_name == "Arbre de Décision":
        model = DecisionTreeClassifier()

    # Entraîner le modèle
    model.fit(X_train, y_train)

    # Sauvegarder le modèle
    joblib.dump(model, 'model.pkl')
    st.write("Modèle sauvegardé sous 'model.pkl'.")

    # Sauvegarder les colonnes
    columns = X.columns
    joblib.dump(columns, 'columns.pkl')
    st.write("Colonnes sauvegardées sous 'columns.pkl'.")

    # Prédictions
    y_pred = model.predict(X_test)

    # Évaluer le modèle
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    st.write(f"### Évaluation du modèle ({model_name})")
    st.write(f"Précision : {accuracy:.2f}")
    st.write("Rapport de classification :")
    st.text(report)

elif page == pages[4]:
    st.write("### Prédiction")

    # Vérifier si les fichiers existent
    if not os.path.exists('model.pkl') or not os.path.exists('scaler.pkl') or not os.path.exists('columns.pkl'):
        st.write("Un ou plusieurs fichiers nécessaires sont manquants. Veuillez entraîner le modèle d'abord.")
    else:
        # Charger le modèle et les fichiers nécessaires
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        columns = joblib.load('columns.pkl')

        st.write("Veuillez entrer les valeurs pour les caractéristiques du modèle.")

        # Créez un formulaire pour les entrées
        with st.form("prediction_form"):
            user_input = {}
            for column in columns:
                if column in df_clean.columns:
                    if df_clean[column].dtype == 'object':
                        # Utiliser un selectbox pour les colonnes catégorielles
                        unique_values = df_clean[column].unique()
                        user_input[column] = st.selectbox(column, options=unique_values)
                    else:
                        # Utiliser un number_input pour les colonnes numériques
                        user_input[column] = st.number_input(column, value=0.0)

            submit_button = st.form_submit_button("Prédire")

        if submit_button:
            input_df = pd.DataFrame([user_input])
            input_df = pd.get_dummies(input_df)

            # Assurez-vous que les colonnes du dataframe d'entrée sont les mêmes que celles du modèle
            input_df = input_df.reindex(columns=columns, fill_value=0)
            input_df = scaler.transform(input_df)

            prediction = model.predict(input_df)
            st.write(f"### Prédiction : {'Oui' if prediction[0] == 1 else 'Non'}")