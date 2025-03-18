import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Configuration pour une meilleure visualisation
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.family'] = 'sans-serif'

def load_gen_ai_jobs_data():
    """Chargement des données d'offres d'emploi en IA générative"""
    print("Chargement des données d'offres d'emploi en IA générative...")
    gen_ai_jobs = pd.read_csv('dataset/fig_4.2.4.csv')
    return gen_ai_jobs

def analyze_gen_ai_jobs_improved(gen_ai_jobs):
    """
    Création d'une visualisation améliorée pour les offres d'emploi en IA générative
    par compétence spécifique
    """
    print("Génération du graphique d'offres d'emploi en IA générative par compétence...")
    
    # Créer une figure plus grande pour meilleure lisibilité
    plt.figure(figsize=(14, 8), facecolor='white')
    ax = plt.gca()
    
    # Vérifier la disponibilité des données
    if 'Generative AI skill' in gen_ai_jobs.columns and 'Number of AI Job Postings' in gen_ai_jobs.columns:
        # Trier les données par nombre d'offres décroissant pour une meilleure visualisation
        sorted_data = gen_ai_jobs.sort_values('Number of AI Job Postings')
        
        if not sorted_data.empty:
            # Définir une palette de couleurs qui se distingue bien
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_data)))
            
            # Créer un graphique à barres horizontales
            bars = ax.barh(
                sorted_data['Generative AI skill'],
                sorted_data['Number of AI Job Postings'],
                color=colors,
                height=0.6,
                alpha=0.8,
                edgecolor='white',
                linewidth=1
            )
            
            # Ajouter les valeurs à la fin de chaque barre
            for i, bar in enumerate(bars):
                value = sorted_data['Number of AI Job Postings'].iloc[i]
                ax.text(
                    value + (sorted_data['Number of AI Job Postings'].max() * 0.02),  # Légèrement décalé
                    bar.get_y() + bar.get_height()/2,
                    f"{value:,}",
                    va='center',
                    ha='left',
                    fontsize=11,
                    fontweight='bold',
                    color='#333333'
                )
            
            # Ajouter un indicateur de tendance/importance
            max_value = sorted_data['Number of AI Job Postings'].max()
            top_skill = sorted_data.iloc[-1]['Generative AI skill']
            
            # Ajouter une annotation pour la compétence la plus demandée
            ax.annotate(
                f"Compétence la plus demandée\n{100*sorted_data['Number of AI Job Postings'].iloc[-1]/sorted_data['Number of AI Job Postings'].sum():.1f}% des offres",
                xy=(sorted_data['Number of AI Job Postings'].iloc[-1], sorted_data.index[-1]),
                xytext=(max_value*0.6, len(sorted_data)-0.5),
                fontsize=11,
                fontweight='bold',
                color='#E74C3C',
                arrowprops=dict(
                    arrowstyle='->',
                    lw=2,
                    color='#E74C3C',
                    connectionstyle='arc3,rad=0.2'
                )
            )
    
    # Titre et étiquettes améliorés
    current_year = gen_ai_jobs['Year'].iloc[0] if not gen_ai_jobs.empty else datetime.now().year
    ax.set_title(f'Offres d\'emploi en IA générative par compétence ({current_year})', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Nombre d\'offres d\'emploi', fontsize=12, labelpad=15)
    ax.set_ylabel('Compétence en IA générative', fontsize=12, labelpad=15)
    
    # Améliorer les graduations des axes
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.tick_params(axis='y', which='major', pad=10)  # Plus d'espace pour les étiquettes Y
    
    # Assurer que l'axe X commence à zéro
    ax.set_xlim(left=0)
    
    # Formater l'axe X pour les grands nombres
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else str(int(x))))
    
    # Grille légère uniquement sur l'axe X
    ax.grid(True, linestyle=':', alpha=0.3, axis='x')
    ax.set_axisbelow(True)
    
    # Ajouter une légende contextuelle
    year_text = f"Données pour l'année {current_year}"
    total_jobs = gen_ai_jobs['Number of AI Job Postings'].sum()
    
    plt.figtext(
        0.5, 0.01,
        f"{year_text} | Total: {total_jobs:,} offres d'emploi",
        fontsize=10,
        ha='center',
        bbox=dict(facecolor='#f8f9fa', edgecolor='none', boxstyle='round,pad=0.5', alpha=0.7)
    )
    
    # Source avec plus de détails
    plt.figtext(
        0.98, 0.02, 
        "Source: Stanford AI Index Report", 
        fontsize=9,
        style='italic',
        ha='right'
    )
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Ajuster pour les annotations
    plt.savefig('output/gen_ai_jobs_improved.png', dpi=300, bbox_inches='tight')
    
    print("Graphique amélioré enregistré sous 'output/gen_ai_jobs_improved.png'")
    
    return ax.figure

# Point d'entrée principal
if __name__ == "__main__":
    # Charger les données
    gen_ai_jobs = load_gen_ai_jobs_data()
    
    # Générer le graphique amélioré
    analyze_gen_ai_jobs_improved(gen_ai_jobs)
    
    print("Traitement terminé!") 