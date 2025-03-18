import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Configuration pour une meilleure visualisation
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.family'] = 'sans-serif'

def load_sector_data():
    """Chargement et nettoyage des données sectorielles pour la matrice d'opportunités"""
    print("Chargement des données sectorielles...")
    sector_data = pd.read_csv('dataset/fig_4.2.6.csv')
    sector_data['adoption_rate'] = sector_data['AI Job Postings (% of All Job Postings)'].str.rstrip('%').astype(float) / 100
    return sector_data

def analyze_sector_growth(sector_data):
    """Préparation des données pour la matrice d'opportunités"""
    sector_growth = []
    
    for sector in sector_data['Sector'].unique():
        sector_years = sector_data[sector_data['Sector'] == sector].sort_values('Year')
        
        if len(sector_years) >= 2:
            first_record = sector_years.iloc[0]
            last_record = sector_years.iloc[-1]
            years_diff = last_record['Year'] - first_record['Year']
            
            if first_record['adoption_rate'] > 0 and years_diff > 0:
                growth = (last_record['adoption_rate'] - first_record['adoption_rate']) / first_record['adoption_rate']
                annual_growth = ((1 + growth) ** (1/years_diff)) - 1
                
                sector_growth.append({
                    'sector': sector,
                    'total_growth': growth,
                    'annual_growth': annual_growth,
                    'current_adoption': last_record['adoption_rate'],
                    'years_analyzed': years_diff
                })
    
    growth_df = pd.DataFrame(sector_growth)
    return growth_df

def create_opportunity_matrix_simplified(sector_growth_df, colors=None):
    """
    Création d'une matrice d'opportunités simplifiée axée sur les secteurs clés
    """
    print("Génération de la matrice d'opportunités par secteur...")
    
    # Définir les couleurs si elles ne sont pas fournies
    if colors is None:
        colors = {
            'admin': '#E74C3C',  # Rouge pour l'administration
            'other': ['#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']  # Couleurs pour les autres secteurs
        }
    
    # Préparer les données
    if 'opportunity_score' not in sector_growth_df.columns:
        sector_growth_df['opportunity_score'] = sector_growth_df['current_adoption'] * sector_growth_df['annual_growth']
    
    # Sélectionner les meilleurs secteurs par score d'opportunité (limité à 6 pour la clarté)
    top_sectors = sector_growth_df.sort_values('opportunity_score', ascending=False).head(6)
    
    # Obtenir les données du secteur administratif
    admin_sector = top_sectors[top_sectors['sector'].str.contains('Public Administration|Government', case=False, regex=True)]
    
    # Normaliser pour la taille des bulles - calcul simplifié
    min_score = top_sectors['opportunity_score'].min()
    max_score = top_sectors['opportunity_score'].max()
    top_sectors['bubble_size'] = 200 + ((top_sectors['opportunity_score'] - min_score) / (max_score - min_score)) * 800
    
    # Convertir en pourcentage pour l'affichage
    top_sectors['adoption_percentage'] = top_sectors['current_adoption'] * 100
    top_sectors['growth_percentage'] = top_sectors['annual_growth'] * 100
    
    # Créer la figure avec un style simplifié
    fig, ax = plt.subplots(figsize=(12, 9), facecolor='white')
    ax.set_facecolor('white')  # Fond simplifié
    
    # Valeurs moyennes pour les quadrants
    mean_adoption = top_sectors['adoption_percentage'].mean()
    mean_growth = top_sectors['growth_percentage'].mean()
    
    # Tracer les lignes des quadrants
    ax.axhline(y=mean_growth, color='#333333', linestyle='--', alpha=0.3, linewidth=1)
    ax.axvline(x=mean_adoption, color='#333333', linestyle='--', alpha=0.3, linewidth=1)
    
    # Remplir les quadrants avec des couleurs subtiles - SIMPLIFIÉ
    max_x = top_sectors['adoption_percentage'].max() * 1.1
    max_y = top_sectors['growth_percentage'].max() * 1.1
    min_y = top_sectors['growth_percentage'].min() * 1.1 if top_sectors['growth_percentage'].min() < 0 else 0
    
    # Remplissages des quadrants simplifiés avec alpha réduit
    ax.fill_between([0, mean_adoption], [mean_growth, mean_growth], [max_y, max_y], color='#90EE90', alpha=0.05)
    ax.fill_between([mean_adoption, max_x], [mean_growth, mean_growth], [max_y, max_y], color='#87CEFA', alpha=0.05)
    ax.fill_between([0, mean_adoption], [min_y, min_y], [mean_growth, mean_growth], color='#ADD8E6', alpha=0.05)
    ax.fill_between([mean_adoption, max_x], [min_y, min_y], [mean_growth, mean_growth], color='#FFB6C1', alpha=0.05)
    
    # Créer deux ensembles de couleurs - un pour l'admin, un pour les autres secteurs
    admin_color = colors['admin']  # Rouge distinct pour l'admin
    other_colors = colors['other']
    
    # Tracer le nuage de points avec un style simplifié
    for i, (idx, row) in enumerate(top_sectors.iterrows()):
        is_admin = row['sector'].lower().find('public administration') >= 0 or row['sector'].lower().find('government') >= 0
        
        # Utiliser une couleur spéciale pour le secteur administratif
        color = admin_color if is_admin else other_colors[i % len(other_colors)]
        alpha = 0.9 if is_admin else 0.7
        edge_width = 2 if is_admin else 1
        
        # Ajouter la bulle
        ax.scatter(
            row['adoption_percentage'], 
            row['growth_percentage'],
            s=row['bubble_size'], 
            alpha=alpha,
            color=color,
            edgecolors='black',
            linewidths=edge_width
        )
        
        # Annoter uniquement l'administration et 1-2 autres secteurs principaux
        if is_admin or i < 2:
            ax.annotate(
                row['sector'], 
                (row['adoption_percentage'], row['growth_percentage']),
                xytext=(15, 5), 
                textcoords='offset points',
                fontsize=10,
                fontweight='bold' if is_admin else 'normal',
                bbox=dict(
                    boxstyle="round,pad=0.3", 
                    fc="white", 
                    ec="black" if is_admin else "gray", 
                    alpha=0.9 if is_admin else 0.7
                ),
                arrowprops=dict(
                    arrowstyle='->', 
                    color='black' if is_admin else 'gray',
                    linewidth=1
                )
            )
            
            # Ajouter le pourcentage uniquement pour l'administration
            if is_admin:
                ax.annotate(
                    f"Croissance: {row['growth_percentage']:.1f}%", 
                    (row['adoption_percentage'], row['growth_percentage']),
                    xytext=(15, -20), 
                    textcoords='offset points',
                    fontsize=9,
                    fontweight='bold',
                    color=admin_color
                )
    
    # Étiquettes de quadrant simplifiées
    quadrant_style = dict(
        ha='center', 
        fontsize=10, 
        bbox=dict(
            facecolor='white', 
            alpha=0.8, 
            edgecolor='lightgray', 
            boxstyle='round,pad=0.2'
        )
    )
    
    # Étiquettes de quadrant en un mot
    ax.text(mean_adoption/2, mean_growth + (max_y - mean_growth)/2, "ÉMERGENT", **quadrant_style)
    ax.text(mean_adoption + (max_x - mean_adoption)/2, mean_growth + (max_y - mean_growth)/2, "LEADER", **quadrant_style)
    ax.text(mean_adoption/2, mean_growth/2, "LATENT", **quadrant_style)
    ax.text(mean_adoption + (max_x - mean_adoption)/2, mean_growth/2, "MATURE", **quadrant_style)
    
    # Titres et étiquettes simplifiés
    ax.set_title('Matrice d\'opportunités par secteur', fontsize=14, fontweight='bold')
    ax.set_xlabel('Taux d\'adoption actuel (%)', fontsize=11)
    ax.set_ylabel('Croissance annuelle (%)', fontsize=11)
    
    # Formatage en pourcentage pour les axes
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=1))
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=1))
    
    # Grille simplifiée
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Note de source simple
    plt.figtext(
        0.5, 0.01, 
        "Source: Stanford AI Index", 
        ha='center', 
        fontsize=8, 
        style='italic'
    )
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig('output/opportunity_matrix_simplified.png', dpi=300, bbox_inches='tight')
    
    print("Graphique enregistré sous 'output/opportunity_matrix_simplified.png'")
    
    return top_sectors, fig

# Point d'entrée principal
if __name__ == "__main__":
    # Charger les données
    sector_data = load_sector_data()
    
    # Analyser la croissance par secteur
    growth_df = analyze_sector_growth(sector_data)
    
    # Définir les couleurs
    colors = {
        'admin': '#E74C3C',  # Rouge pour l'administration
        'other': ['#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']  # Couleurs pour les autres secteurs
    }
    
    # Générer la matrice d'opportunités
    create_opportunity_matrix_simplified(growth_df, colors)
    
    print("Traitement terminé!") 