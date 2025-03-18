import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Configuration pour une meilleure visualisation
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.family'] = 'sans-serif'

def load_geo_data():
    """Chargement et nettoyage des données géographiques uniquement"""
    print("Chargement des données géographiques...")
    geo_data = pd.read_csv('dataset/fig_4.2.1.csv')
    geo_data['adoption_rate'] = geo_data['AI job postings (% of all job postings)'].str.rstrip('%').astype(float) / 100
    return geo_data

def plot_country_trends_improved(geo_data, countries_to_plot, colors=None):
    """
    Création d'un graphique amélioré pour les tendances d'adoption par pays
    """
    print(f"Génération du graphique des tendances pour {', '.join(countries_to_plot)}...")
    
    # Définir les couleurs si elles ne sont pas fournies - Utiliser les mêmes couleurs que fix_predictions.py
    country_colors = {
        'United States': '#0052CC',  # Bleu riche
        'Spain': '#E31A1C',          # Rouge vif
        'Singapore': '#00994C',      # Vert riche
        'France': '#7928CA',         # Violet vif
        'Sweden': '#FF8C00',         # Orange vif
        'Canada': '#E76F51',         # Terracotta
        'United Kingdom': '#6A0DAD'  # Violet profond
    }
    
    # Couleurs par défaut pour les pays non inclus dans notre palette
    default_colors = ['#1E88E5', '#FF6D00', '#43A047', '#D81B60', '#8E24AA']
    
    plt.figure(figsize=(12, 7), facecolor='white')  # Même taille que dans fix_predictions.py
    
    # Formater les données pour l'affichage en pourcentage
    geo_data['adoption_percentage'] = geo_data['adoption_rate'] * 100
    
    # Configurer les axes avec des lignes de grille améliorées
    ax = plt.gca()
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)  # Placer la grille derrière les données
    
    # Tracer les lignes pour chaque pays avec une palette cohérente
    for i, country in enumerate(countries_to_plot):
        country_data = geo_data[geo_data['Geographic area'] == country].sort_values('Year')
        
        # Utiliser la même couleur que dans fix_predictions.py pour la cohérence
        color = country_colors.get(country, default_colors[i % len(default_colors)])
        
        plt.plot(
            country_data['Year'], 
            country_data['adoption_percentage'], 
            marker='o', 
            linewidth=2.5, 
            label=country,
            color=color,
            markersize=7,  # Ajusté pour être cohérent avec fix_predictions
            markerfacecolor='white',
            markeredgewidth=1.5,
            markeredgecolor=color
        )
    
    # Titre et étiquettes améliorés
    plt.title('Évolution du taux d\'adoption de l\'IA par pays', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Année', fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel('Taux d\'adoption (%)', fontsize=12, fontweight='bold', labelpad=10)
    
    # Formater l'axe Y pour l'affichage en pourcentage
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=1))
    
    # Légende optimisée
    legend = plt.legend(
        title="Pays",
        title_fontsize=12, 
        fontsize=10,
        frameon=True,
        framealpha=0.95,
        edgecolor='lightgray',
        facecolor='white',
        loc='best'
    )
    
    # Ajouter la source des données
    plt.figtext(
        0.01, 0.01, 
        "Source: Stanford AI Index", 
        fontsize=8, 
        style='italic', 
        ha='left'
    )
    
    plt.tight_layout()
    plt.savefig('output/country_trends_improved.png', dpi=300, bbox_inches='tight')
    
    print(f"Graphique enregistré sous 'output/country_trends_improved.png'")
    
    return plt.gcf()  # Retourner la figure pour une utilisation ultérieure

# Point d'entrée principal
if __name__ == "__main__":
    # Charger les données
    geo_data = load_geo_data()
    
    # Pays à représenter - Les mêmes que dans fix_predictions.py pour la cohérence
    countries_to_plot = ['United States', 'Spain', 'Singapore', 'France', 'Sweden']
    
    # Générer le graphique
    plot_country_trends_improved(geo_data, countries_to_plot)
    
    print("Traitement terminé!") 