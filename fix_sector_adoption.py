import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuration pour une meilleure visualisation
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.family'] = 'sans-serif'

def load_sector_data():
    """Chargement et nettoyage des données par secteur uniquement"""
    print("Chargement des données sectorielles...")
    sector_data = pd.read_csv('dataset/fig_4.2.6.csv')
    sector_data['adoption_rate'] = sector_data['AI Job Postings (% of All Job Postings)'].str.rstrip('%').astype(float) / 100
    return sector_data

def plot_sector_adoption_simplified(sector_data, colors=None):
    """
    Création d'un graphique en camembert pour l'adoption par secteur
    avec mise en évidence de l'administration publique
    """
    print("Génération du graphique d'adoption par secteur...")
    
    # Définir les couleurs si elles ne sont pas fournies
    if colors is None:
        colors = {'primary': '#3498DB', 'secondary': '#E74C3C'}
    
    latest_year = sector_data['Year'].max()
    latest_data = sector_data[sector_data['Year'] == latest_year].copy()
    
    # Trier les données par taux d'adoption
    latest_data.sort_values('adoption_rate', ascending=False, inplace=True)
    
    # Limiter à 8 secteurs pour la lisibilité
    if len(latest_data) > 8:
        latest_data = latest_data.head(8)
    
    # Convertir en pourcentage pour l'affichage
    latest_data['adoption_percentage'] = latest_data['adoption_rate'] * 100
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    
    # Identifier le secteur de l'administration publique
    admin_rows = latest_data['Sector'].str.contains('Public Administration|Government', case=False, regex=True)
    
    # Créer une palette de couleurs - mettre en évidence l'administration en rouge
    pie_colors = ['#3498DB', '#2ECC71', '#9B59B6', '#F39C12', '#1ABC9C', '#34495E', '#D35400', '#7F8C8D']
    
    # Remplacer la couleur pour l'administration publique
    for i, is_admin in enumerate(admin_rows):
        if is_admin:
            pie_colors[i] = colors['secondary']  # Rouge pour l'administration
    
    # Créer le camembert
    wedges, texts, autotexts = ax.pie(
        latest_data['adoption_percentage'],
        labels=None,  # On ajoute une légende séparée pour plus de clarté
        autopct='%1.1f%%',
        startangle=90,
        colors=pie_colors,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
        textprops={'fontsize': 12, 'fontweight': 'bold', 'color': 'white'},
        shadow=False,
        explode=[0.05 if is_admin else 0 for is_admin in admin_rows]  # Détacher légèrement l'administration
    )
    
    # Ajouter une légende personnalisée
    legend_labels = [f"{sector} ({pct:.1f}%)" for sector, pct in 
                    zip(latest_data['Sector'], latest_data['adoption_percentage'])]
    ax.legend(wedges, legend_labels, title=f"Secteurs ({latest_year})", 
             loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    
    # Titre simplifié
    ax.set_title(f'Adoption de l\'IA par secteur ({latest_year})', 
                fontsize=14, fontweight='bold')
    
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
    plt.savefig('output/sector_adoption_pie_chart.png', dpi=300, bbox_inches='tight')
    
    print("Graphique enregistré sous 'output/sector_adoption_pie_chart.png'")
    
    return fig

# Point d'entrée principal
if __name__ == "__main__":
    # Charger les données
    sector_data = load_sector_data()
    
    # Définir les couleurs
    colors = {'primary': '#3498DB', 'secondary': '#E74C3C'}
    
    # Générer le graphique
    plot_sector_adoption_simplified(sector_data, colors)
    
    print("Traitement terminé!") 