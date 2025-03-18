import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Configuration pour une meilleure visualisation
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.family'] = 'sans-serif'

def load_skills_cluster_data():
    """Chargement et nettoyage des données de clusters de compétences"""
    print("Chargement des données de clusters de compétences...")
    skills_cluster = pd.read_csv('dataset/fig_4.2.2.csv')
    
    # Nettoyage des données de cluster de compétences
    skills_cluster['adoption_rate'] = skills_cluster['AI job postings (% of all job postings)'].str.rstrip('%').astype(float) / 100
    
    return skills_cluster

def plot_skills_clusters_improved(skills_cluster, colors=None):
    """
    Création d'un graphique amélioré pour l'évolution des compétences par cluster
    """
    print("Génération du graphique d'évolution des compétences par cluster...")
    
    # Définir les couleurs si elles ne sont pas fournies
    if colors is None:
        colors = {
            'neutrals': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        }
    
    skill_clusters = skills_cluster['Skill cluster'].unique()
    
    # Créer une figure avec un style amélioré
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
    ax.set_facecolor('#f9f9f9')
    
    # Tracer chaque cluster avec une meilleure distinction visuelle
    for i, cluster in enumerate(skill_clusters):
        cluster_data = skills_cluster[skills_cluster['Skill cluster'] == cluster].sort_values('Year')
        
        # Convertir en pourcentage pour l'affichage
        cluster_data['adoption_percentage'] = cluster_data['adoption_rate'] * 100
        
        # Attribuer une couleur de la palette
        color = colors['neutrals'][i % len(colors['neutrals'])]
        
        # Tracer avec un style distinct
        ax.plot(cluster_data['Year'], 
                cluster_data['adoption_percentage'], 
                marker='o', 
                linewidth=2.5, 
                label=cluster,
                color=color,
                markersize=7,
                markerfacecolor='white',
                markeredgewidth=1.5,
                markeredgecolor=color)
    
    # Titres et étiquettes améliorés
    ax.set_title('Évolution des compétences IA par groupe', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Année', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Taux d\'adoption (%)', fontsize=12, fontweight='bold', labelpad=10)
    
    # Formatage en pourcentage
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=1))
    
    # Grille discrète
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    ax.set_axisbelow(True)
    
    # Légende optimisée
    legend = ax.legend(
        title="Groupes de compétences",
        title_fontsize=12, 
        fontsize=10,
        frameon=True,
        framealpha=0.95,
        edgecolor='lightgray',
        loc='best'
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
    plt.savefig('output/skill_clusters_improved.png', dpi=300, bbox_inches='tight')
    
    print("Graphique enregistré sous 'output/skill_clusters_improved.png'")
    
    return fig

# Point d'entrée principal
if __name__ == "__main__":
    # Charger les données
    skills_cluster = load_skills_cluster_data()
    
    # Générer le graphique
    plot_skills_clusters_improved(skills_cluster)
    
    print("Traitement terminé!") 