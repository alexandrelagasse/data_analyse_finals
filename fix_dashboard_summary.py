import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Configuration pour une meilleure visualisation
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.family'] = 'sans-serif'

def load_data():
    """Chargement des données pour le tableau de bord récapitulatif"""
    print("Chargement des données pour le tableau de bord récapitulatif...")
    
    # Données géographiques
    geo_data = pd.read_csv('dataset/fig_4.2.1.csv')
    geo_data['adoption_rate'] = geo_data['AI job postings (% of all job postings)'].str.rstrip('%').astype(float) / 100
    
    # Données sectorielles
    sector_data = pd.read_csv('dataset/fig_4.2.6.csv')
    sector_data['adoption_rate'] = sector_data['AI Job Postings (% of All Job Postings)'].str.rstrip('%').astype(float) / 100
    
    # Données de compétences spécifiques
    skills_specific = pd.read_csv('dataset/fig_4.2.3.csv')
    if not pd.api.types.is_numeric_dtype(skills_specific['Year']):
        skills_specific['Year'] = pd.to_numeric(skills_specific['Year'], errors='coerce')
    
    return geo_data, sector_data, skills_specific

def create_dashboard_summary(geo_data, sector_data, skills_specific):
    """
    Création d'un tableau de bord récapitulatif avec les indicateurs clés
    """
    print("Génération du tableau de bord récapitulatif...")
    
    # Définir des couleurs cohérentes
    colors = {
        'primary': '#1f77b4',    # Bleu principal
        'secondary': '#ff7f0e',  # Orange
        'tertiary': '#2ca02c',   # Vert
        'quaternary': '#d62728', # Rouge
    }
    
    plt.figure(figsize=(16, 10), facecolor='white')
    plt.suptitle('TABLEAU DE BORD DE L\'ADOPTION DE L\'IA', fontsize=20, fontweight='bold', y=0.98)
    
    # Configuration de la grille pour les sous-graphiques
    gs = plt.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1], wspace=0.3, hspace=0.4)
    
    # ----- 1. Évolution globale de l'adoption de l'IA -----
    ax1 = plt.subplot(gs[0, 0])
    ax1.set_facecolor('#f9f9f9')
    
    # Calculer la moyenne globale par année
    global_adoption = geo_data.groupby('Year')['adoption_rate'].mean() * 100  # Convertir en pourcentage
    
    years = global_adoption.index
    adoption_rates = global_adoption.values
    
    # Tracer la courbe d'évolution globale
    ax1.plot(years, adoption_rates, marker='o', linewidth=3, color=colors['primary'],
            markersize=8, markerfacecolor='white', markeredgewidth=1.5, markeredgecolor=colors['primary'])
    
    # Ajouter un remplissage sous la courbe
    ax1.fill_between(years, 0, adoption_rates, alpha=0.2, color=colors['primary'])
    
    # Ajouter des valeurs sur les points
    for i, (year, rate) in enumerate(zip(years, adoption_rates)):
        ax1.annotate(f"{rate:.1f}%", 
                    (year, rate), 
                    textcoords="offset points",
                    xytext=(0, 10), 
                    ha='center',
                    fontsize=9,
                    fontweight='bold')
    
    # Calculer et afficher le TCAC (taux de croissance annuel composé)
    if len(years) >= 2:
        first_year, last_year = years[0], years[-1]
        first_rate, last_rate = adoption_rates[0], adoption_rates[-1]
        years_diff = last_year - first_year
        
        if first_rate > 0 and years_diff > 0:
            cagr = (((last_rate / first_rate) ** (1/years_diff)) - 1) * 100
            ax1.text(0.05, 0.95, f"TCAC: {cagr:.1f}%",
                    transform=ax1.transAxes,
                    fontsize=12,
                    fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
    
    ax1.set_title('Évolution globale de l\'adoption de l\'IA', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Année', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Taux d\'adoption moyen (%)', fontsize=10, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_axisbelow(True)
    
    # ----- 2. Top 5 des pays par adoption -----
    ax2 = plt.subplot(gs[0, 1])
    ax2.set_facecolor('#f9f9f9')
    
    # Obtenir les données les plus récentes
    latest_year = geo_data['Year'].max()
    top5_countries = geo_data[geo_data['Year'] == latest_year].sort_values('adoption_rate', ascending=False).head(5)
    
    # Créer des barres horizontales avec dégradé de couleurs
    countries = top5_countries['Geographic area']
    rates = top5_countries['adoption_rate'] * 100  # Convertir en pourcentage
    
    colors_gradient = plt.cm.Blues(np.linspace(0.4, 0.8, len(countries)))
    bars = ax2.barh(countries, rates, color=colors_gradient, edgecolor='#333333', linewidth=0.8)
    
    # Ajouter des valeurs sur les barres
    for i, bar in enumerate(bars):
        ax2.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2, 
                f"{rates.iloc[i]:.1f}%", 
                va='center', 
                fontsize=9,
                fontweight='bold')
    
    ax2.set_title('Top 5 des pays par adoption de l\'IA', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Taux d\'adoption (%)', fontsize=10, fontweight='bold')
    ax2.set_xlim(0, max(rates) * 1.2)  # Marge pour les étiquettes
    ax2.grid(True, linestyle='--', alpha=0.7, axis='x')
    ax2.set_axisbelow(True)
    
    # Supprimer les bordures inutiles
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    
    # ----- 3. Top 5 des secteurs par adoption -----
    ax3 = plt.subplot(gs[1, 0])
    ax3.set_facecolor('#f9f9f9')
    
    # Obtenir les données les plus récentes
    latest_year_sector = sector_data['Year'].max()
    top5_sectors = sector_data[sector_data['Year'] == latest_year_sector].sort_values('adoption_rate', ascending=False).head(5)
    
    # Créer des barres horizontales avec dégradé de couleurs
    sectors = top5_sectors['Sector']
    sector_rates = top5_sectors['adoption_rate'] * 100  # Convertir en pourcentage
    
    colors_gradient_sector = plt.cm.Greens(np.linspace(0.4, 0.8, len(sectors)))
    bars_sector = ax3.barh(sectors, sector_rates, color=colors_gradient_sector, edgecolor='#333333', linewidth=0.8)
    
    # Ajouter des valeurs sur les barres
    for i, bar in enumerate(bars_sector):
        ax3.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2, 
                f"{sector_rates.iloc[i]:.1f}%", 
                va='center', 
                fontsize=9,
                fontweight='bold')
    
    ax3.set_title('Top 5 des secteurs par adoption de l\'IA', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Taux d\'adoption (%)', fontsize=10, fontweight='bold')
    ax3.set_xlim(0, max(sector_rates) * 1.2)  # Marge pour les étiquettes
    ax3.grid(True, linestyle='--', alpha=0.7, axis='x')
    ax3.set_axisbelow(True)
    
    # Supprimer les bordures inutiles
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    
    # ----- 4. Top 5 des compétences IA les plus demandées -----
    ax4 = plt.subplot(gs[1, 1])
    ax4.set_facecolor('#f9f9f9')
    
    # Obtenir les données les plus récentes
    latest_year_skills = skills_specific['Year'].max()
    top5_skills = skills_specific[skills_specific['Year'] == latest_year_skills].sort_values('Number of AI Job Postings', ascending=False).head(5)
    
    # Créer des barres horizontales avec dégradé de couleurs
    skills = top5_skills['Skill']
    job_counts = top5_skills['Number of AI Job Postings']
    
    colors_gradient_skills = plt.cm.Oranges(np.linspace(0.4, 0.8, len(skills)))
    bars_skills = ax4.barh(skills, job_counts, color=colors_gradient_skills, edgecolor='#333333', linewidth=0.8)
    
    # Ajouter des valeurs sur les barres
    for i, bar in enumerate(bars_skills):
        ax4.text(bar.get_width() + job_counts.max()*0.02, bar.get_y() + bar.get_height()/2, 
                f"{job_counts.iloc[i]:,}", 
                va='center', 
                fontsize=9,
                fontweight='bold')
    
    ax4.set_title('Top 5 des compétences IA les plus demandées', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Nombre d\'offres d\'emploi', fontsize=10, fontweight='bold')
    ax4.set_xlim(0, max(job_counts) * 1.2)  # Marge pour les étiquettes
    ax4.grid(True, linestyle='--', alpha=0.7, axis='x')
    ax4.set_axisbelow(True)
    
    # Supprimer les bordures inutiles
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['left'].set_visible(False)
    
    # Formatage numérique pour l'axe X
    ax4.get_xaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x)))
    )
    
    # Ajouter une note sur les données
    plt.figtext(0.5, 0.01, 
               "Source: Stanford AI Index | Données analysées le " + datetime.now().strftime("%d/%m/%Y"), 
               ha="center", fontsize=9, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('output/dashboard_summary.png', dpi=300, bbox_inches='tight')
    
    print("Tableau de bord enregistré sous 'output/dashboard_summary.png'")
    
    return plt.gcf()

# Point d'entrée principal
if __name__ == "__main__":
    # Charger les données
    geo_data, sector_data, skills_specific = load_data()
    
    # Générer le tableau de bord
    create_dashboard_summary(geo_data, sector_data, skills_specific)
    
    print("Traitement terminé!") 