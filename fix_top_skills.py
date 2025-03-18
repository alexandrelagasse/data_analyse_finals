import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuration pour une meilleure visualisation
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.family'] = 'sans-serif'

def load_skills_data():
    """Chargement et nettoyage des données de compétences spécifiques"""
    print("Chargement des données de compétences spécifiques...")
    skills_specific = pd.read_csv('dataset/fig_4.2.3.csv')
    
    # Convertir 'Year' en type numérique si nécessaire
    if not pd.api.types.is_numeric_dtype(skills_specific['Year']):
        skills_specific['Year'] = pd.to_numeric(skills_specific['Year'], errors='coerce')
    
    return skills_specific

def plot_top_skills_improved(skills_specific, colors=None):
    """
    Création d'un graphique à barres amélioré pour les compétences les plus demandées
    """
    print("Génération du graphique des compétences les plus demandées...")
    
    # Définir les couleurs si elles ne sont pas fournies
    if colors is None:
        colors = {
            'gradient': plt.cm.Blues(np.linspace(0.4, 0.8, 10))  # Pour les dégradés
        }
    
    # Filtrer pour l'année la plus récente
    latest_year = skills_specific['Year'].max()
    top_skills = skills_specific[skills_specific['Year'] == latest_year]
    
    # Trier et limiter aux 10 meilleures compétences
    top_skills = top_skills.sort_values('Number of AI Job Postings', ascending=True).tail(10)
    
    # Créer une figure avec un style amélioré
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    ax.set_facecolor('#f9f9f9')
    
    # Palette de couleurs visuellement attrayante avec accent sur les compétences principales
    colors_gradient = colors['gradient']
    
    # Créer des barres horizontales
    bars = ax.barh(top_skills['Skill'], 
                  top_skills['Number of AI Job Postings'], 
                  color=colors_gradient,
                  edgecolor='#333333', 
                  linewidth=0.8)
    
    # Ajouter des valeurs sur les barres avec un formatage amélioré
    for i, bar in enumerate(bars):
        value = top_skills['Number of AI Job Postings'].iloc[i]
        ax.text(
            value + (max(top_skills['Number of AI Job Postings']) * 0.02), 
            bar.get_y() + bar.get_height()/2, 
            f"{value:,}",  # Formater avec séparateur de milliers
            va='center',
            fontsize=9,
            fontweight='bold',
            color='#333333'
        )
    
    # Titres et étiquettes améliorés
    ax.set_title(f'Top 10 des compétences IA les plus demandées ({latest_year})', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Nombre d\'offres d\'emploi', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('')  # Supprimer l'étiquette Y pour plus de clarté
    
    # Axes améliorés
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    
    # Grille horizontale discrète
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Formatage numérique pour l'axe X
    ax.get_xaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x)))
    )
    
    # Source des données
    plt.figtext(
        0.01, 0.01, 
        "Source: Stanford AI Index", 
        fontsize=8, 
        style='italic', 
        ha='left'
    )
    
    plt.tight_layout()
    plt.savefig('output/top_skills_improved.png', dpi=300, bbox_inches='tight')
    
    print("Graphique enregistré sous 'output/top_skills_improved.png'")
    
    return fig

# Point d'entrée principal
if __name__ == "__main__":
    # Charger les données
    skills_specific = load_skills_data()
    
    # Générer le graphique
    plot_top_skills_improved(skills_specific)
    
    print("Traitement terminé!") 