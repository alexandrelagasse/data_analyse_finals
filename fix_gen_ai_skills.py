import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuration pour une meilleure visualisation
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.family'] = 'sans-serif'

def load_gen_ai_skills_data():
    """Chargement et nettoyage des données de compétences en IA générative"""
    print("Chargement des données de compétences en IA générative...")
    gen_ai_skills = pd.read_csv('dataset/fig_4.2.5.csv')
    
    # Nettoyage des données relatives aux compétences en IA générative
    if 'Skill share in AI job postings (%)' in gen_ai_skills.columns:
        gen_ai_skills['skill_share'] = gen_ai_skills['Skill share in AI job postings (%)'].str.rstrip('%').astype(float) / 100
    
    return gen_ai_skills

def analyze_gen_ai_skills_simplified(gen_ai_skills):
    """
    Création d'un graphique en camembert pour la distribution des compétences en IA générative
    """
    print("Génération du graphique des compétences en IA générative...")
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    
    if 'Generative AI skill' in gen_ai_skills.columns and 'skill_share' in gen_ai_skills.columns:
        # Trier par skill_share
        sorted_skills = gen_ai_skills.sort_values('skill_share', ascending=False)
        
        # Garder uniquement les meilleures compétences (limité à 6)
        if len(sorted_skills) > 6:
            sorted_skills = sorted_skills.head(6)
        
        # Mettre en évidence les compétences clés
        highlight_skills = ['Generative AI', 'Language Model', 'ChatGPT', 'GPT', 'LLM', 'NLP']
        
        # Créer une palette de couleurs pour le camembert
        pie_colors = ['#3498DB', '#2ECC71', '#9B59B6', '#F39C12', '#1ABC9C', '#34495E']
        
        # Remplacer les couleurs pour les compétences mises en évidence
        for i, skill in enumerate(sorted_skills['Generative AI skill']):
            is_highlight = any(hs.lower() in skill.lower() for hs in highlight_skills)
            if is_highlight:
                pie_colors[i] = '#E74C3C'  # Rouge pour les compétences mises en évidence
        
        # Calculer les valeurs d'explode pour mettre en évidence certaines compétences
        explode = []
        for skill in sorted_skills['Generative AI skill']:
            is_highlight = any(hs.lower() in skill.lower() for hs in highlight_skills)
            explode.append(0.05 if is_highlight else 0)
        
        # Créer le camembert
        wedges, texts, autotexts = ax.pie(
            sorted_skills['skill_share'] * 100,  # Convertir en pourcentage
            labels=None,  # On ajoute une légende séparée pour plus de clarté
            autopct='%1.1f%%',
            startangle=90,
            colors=pie_colors,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
            textprops={'fontsize': 12, 'fontweight': 'bold', 'color': 'white'},
            shadow=False,
            explode=explode  # Détacher légèrement les compétences mises en évidence
        )
        
        # Ajouter une légende personnalisée
        legend_labels = [f"{skill} ({pct:.1f}%)" for skill, pct in 
                        zip(sorted_skills['Generative AI skill'], 
                            sorted_skills['skill_share'] * 100)]
        ax.legend(wedges, legend_labels, title="Compétences en IA générative", 
                 loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    
    # Titre simplifié
    ax.set_title('Distribution des compétences en IA générative', fontsize=14, fontweight='bold')
    
    # Égaliser les axes pour un cercle parfait
    ax.axis('equal')
    
    # Note de source simple
    plt.figtext(
        0.98, 0.01, 
        "Source: Stanford AI Index", 
        fontsize=8, 
        style='italic',
        ha='right'
    )
    
    plt.tight_layout()
    plt.savefig('output/gen_ai_skills_pie_chart.png', dpi=300, bbox_inches='tight')
    
    print("Graphique enregistré sous 'output/gen_ai_skills_pie_chart.png'")
    
    return fig

# Point d'entrée principal
if __name__ == "__main__":
    # Charger les données
    gen_ai_skills = load_gen_ai_skills_data()
    
    # Générer le graphique
    analyze_gen_ai_skills_simplified(gen_ai_skills)
    
    print("Traitement terminé!") 