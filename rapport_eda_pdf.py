import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Pour empêcher matplotlib d'ouvrir une fenêtre graphique interactive
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from fpdf import FPDF

def generate_eda_report(file_path):
    print("Chargement des données...")
    
    # Chargement du fichier CSV
    try:
        df = pd.read_csv(file_path)
        print(f"Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        return
    
    # Analyse exploratoire des données
    print("Début de l'analyse exploratoire...")
    
    # Feature engineering
    print("Création de nouvelles features...")
    df['AGE_A_OUVERTURE'] = df['AGE'] - (df['CUSTOMER_SINCE'] // 10000)  # Exemple d'une feature
    df['AGE_GROUP'] = pd.cut(df['AGE'], bins=[0, 18, 30, 40, 50, 60, np.inf], labels=["<18", "18-30", "30-40", "40-50", "50-60", ">60"])
    df['TENURE_GROUP'] = pd.cut(df['TENURE'], bins=[0, 5, 10, 20, np.inf], labels=["<5", "5-10", "10-20", ">20"])

    print("Feature engineering terminé.")
    
    # Affichage des informations de base
    print("Aperçu des données:")
    print(df.head())

    # Sélection des variables numériques et catégorielles
    numerical_features = ['AGE', 'TENURE', 'SOLDE_MOYEN', 'OPERATIONS_MENSUELLES_MOY']
    categorical_features = ['GENRE', 'STATUT_MARITAL', 'AGE_GROUP', 'TENURE_GROUP']
    
    print("Analyse des valeurs manquantes...")
    missing_data = df.isnull().sum()
    print("Valeurs manquantes par colonne:")
    print(missing_data[missing_data > 0])
    
    print("Création de graphiques...")
    # Création de graphiques
    try:
        plt.figure(figsize=(10, 6))
        sns.heatmap(df[numerical_features].corr(), annot=True, cmap='coolwarm')
        plt.savefig("correlation_matrix.png")
        print("Matrice de corrélation sauvegardée sous 'correlation_matrix.png'.")

        # Graphiques supplémentaires selon les variables
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='AGE_GROUP', y='SOLDE_MOYEN', data=df)
        plt.savefig("age_group_boxplot.png")
        print("Boxplot sauvegardé sous 'age_group_boxplot.png'.")
    except Exception as e:
        print(f"Erreur lors de la création des graphiques: {e}")
    
    # Sélection de features avec RFE (Recursive Feature Elimination)
    print("Sélection de features avec RFE...")
    X = df[numerical_features].fillna(0)
    y = df['appetent'].astype(int)  # Assurez-vous que 'appetent' est binaire

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    rfe = RFE(model, n_features_to_select=5)
    rfe = rfe.fit(X, y)
    
    selected_features = X.columns[rfe.support_]
    print("Features sélectionnées par RFE:", selected_features)
    
    # Création du rapport PDF
    print("Création du rapport PDF...")
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, txt="Rapport EDA - Analyse des Données Clients", ln=True, align="C")
    pdf.ln(10)
    
    pdf.cell(200, 10, txt="Aperçu des données:", ln=True)
    pdf.multi_cell(0, 10, txt=str(df.head()))
    pdf.ln(10)

    pdf.cell(200, 10, txt="Matrice de corrélation des variables numériques:", ln=True)
    pdf.image('correlation_matrix.png', x=10, y=pdf.get_y(), w=180)
    pdf.ln(70)

    pdf.cell(200, 10, txt="Boxplot par groupe d'âge:", ln=True)
    pdf.image('age_group_boxplot.png', x=10, y=pdf.get_y(), w=180)
    pdf.ln(70)

    pdf.cell(200, 10, txt="Résultats de la sélection de features (RFE):", ln=True)
    pdf.multi_cell(0, 10, txt="Features sélectionnées: " + ", ".join(selected_features))
    
    # Sauvegarde du PDF
    try:
        pdf_output_path = "rapport_EDA_clients_mdm.pdf"
        pdf.output(pdf_output_path)
        print(f"Rapport généré avec succès : {pdf_output_path}")
    except Exception as e:
        print(f"Erreur lors de la création du rapport PDF: {e}")

# Appel de la fonction pour générer le rapport
generate_eda_report("/home/kaoutar/Desktop/data/df_main_prepared.csv")
