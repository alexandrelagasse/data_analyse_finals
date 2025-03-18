import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuration pour une meilleure visualisation
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.family'] = 'sans-serif'

def load_data():
    """Chargement des données pour les recommandations d'investissement"""
    print("Chargement des données pour les recommandations d'investissement...")
    
    # Données géographiques
    geo_data = pd.read_csv('dataset/fig_4.2.1.csv')
    geo_data['adoption_rate'] = geo_data['AI job postings (% of all job postings)'].str.rstrip('%').astype(float) / 100
    
    # Données sectorielles
    sector_data = pd.read_csv('dataset/fig_4.2.6.csv')
    sector_data['adoption_rate'] = sector_data['AI Job Postings (% of All Job Postings)'].str.rstrip('%').astype(float) / 100
    
    return geo_data, sector_data

def analyze_sector_growth(sector_data):
    """Analyse de la croissance par secteur"""
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

def generate_recommendations_simplified(geo_data, sector_growth_df):
    """
    Création d'un graphique simplifié pour les recommandations d'investissement
    """
    print("Génération du graphique des recommandations d'investissement...")
    
    # Calcul des scores d'opportunité par pays (simplifié)
    country_growth = []
    
    for country in geo_data['Geographic area'].unique():
        country_data = geo_data[geo_data['Geographic area'] == country].sort_values('Year')
        if len(country_data) >= 2:
            first_year = country_data.iloc[0]
            last_year = country_data.iloc[-1]
            years_diff = last_year['Year'] - first_year['Year']
            
            if first_year['adoption_rate'] > 0 and years_diff > 0:
                growth = (last_year['adoption_rate'] - first_year['adoption_rate']) / first_year['adoption_rate']
                annual_growth = ((1 + growth) ** (1/years_diff)) - 1
                
                opportunity_score = last_year['adoption_rate'] * annual_growth
                
                country_growth.append({
                    'country': country,
                    'current_adoption': last_year['adoption_rate'],
                    'annual_growth': annual_growth,
                    'opportunity_score': opportunity_score
                })
    
    country_growth_df = pd.DataFrame(country_growth)
    top_country_opportunities = country_growth_df.sort_values('opportunity_score', ascending=False).head(3)
    
    # Meilleurs secteurs pour l'investissement
    if 'opportunity_score' not in sector_growth_df.columns:
        sector_growth_df['opportunity_score'] = sector_growth_df['current_adoption'] * sector_growth_df['annual_growth']
    
    top_sector_opportunities = sector_growth_df.sort_values('opportunity_score', ascending=False).head(3)
    
    # Figure simplifiée - colonne unique au lieu de deux colonnes
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
    
    # Combiner les pays et les secteurs dans un seul dataframe pour la visualisation
    combined_data = pd.DataFrame({
        'category': ["Secteur: " + s for s in top_sector_opportunities['sector']] + 
                   ["Pays: " + c for c in top_country_opportunities['country']],
        'score': list(top_sector_opportunities['opportunity_score']) + 
                list(top_country_opportunities['opportunity_score']),
        'growth': list(top_sector_opportunities['annual_growth'] * 100) + 
                 list(top_country_opportunities['annual_growth'] * 100),
        'adoption': list(top_sector_opportunities['current_adoption'] * 100) + 
                   list(top_country_opportunities['current_adoption'] * 100),
        'type': ['sector']*len(top_sector_opportunities) + ['country']*len(top_country_opportunities)
    })
    
    # Trier par score
    combined_data = combined_data.sort_values('score', ascending=True)
    
    # Créer des couleurs personnalisées - mettre en évidence le secteur admin en rouge
    bar_colors = []
    for idx, row in combined_data.iterrows():
        if 'Administration Publique' in row['category'] or 'Public Administration' in row['category'] or 'Government' in row['category']:
            bar_colors.append('#E74C3C')  # Rouge pour l'administration
        elif row['type'] == 'sector':
            bar_colors.append('#3498DB')  # Bleu pour les secteurs
        else:
            bar_colors.append('#2ECC71')  # Vert pour les pays
    
    # Créer les barres
    bars = ax.barh(combined_data['category'], combined_data['score'], color=bar_colors)
    
    # Ajouter les valeurs de score
    for i, (idx, row) in enumerate(combined_data.iterrows()):
        # Formater le score
        score_text = f"{row['score']:.4f}"
        
        # Ajouter le texte
        ax.text(
            row['score'] + max(combined_data['score'])*0.01,
            i,
            score_text,
            va='center',
            fontsize=9,
            fontweight='bold' if ('Administration Publique' in row['category'] or 'Public Administration' in row['category'] or 'Government' in row['category']) else 'normal'
        )
        
        # Ajouter les informations de croissance et d'adoption uniquement pour les meilleures opportunités
        if i >= len(combined_data) - 3:
            # Formater le texte d'information
            info_text = f"Croissance: {row['growth']:.1f}%\nAdoption: {row['adoption']:.1f}%"
            
            # Ajouter le texte légèrement à gauche de la barre
            ax.text(
                row['score']/2,
                i,
                info_text,
                va='center',
                ha='center',
                fontsize=9,
                color='white' if row['score'] > 0.001 else 'black',
                fontweight='bold' if ('Administration Publique' in row['category'] or 'Public Administration' in row['category'] or 'Government' in row['category']) else 'normal'
            )
    
    # Titre et étiquettes
    ax.set_title('Recommandations stratégiques d\'investissement', fontsize=14, fontweight='bold')
    ax.set_xlabel('Score d\'opportunité', fontsize=11)
    
    # Nettoyer les axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Supprimer les graduations de l'axe Y mais conserver les étiquettes
    ax.tick_params(axis='y', which='both', length=0)
    
    # Grille légère
    ax.grid(True, linestyle=':', alpha=0.3, axis='x')
    ax.set_axisbelow(True)
    
    # Note méthodologique simple
    plt.figtext(
        0.5, 0.01, 
        "Score d'opportunité = Taux d'adoption × Taux de croissance annuel | Source: Stanford AI Index", 
        ha='center', 
        fontsize=8, 
        style='italic'
    )
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig('output/investment_recommendations_simplified.png', dpi=300, bbox_inches='tight')
    
    print("Graphique enregistré sous 'output/investment_recommendations_simplified.png'")
    
    return top_country_opportunities, top_sector_opportunities, fig

# Point d'entrée principal
if __name__ == "__main__":
    # Charger les données
    geo_data, sector_data = load_data()
    
    # Analyser la croissance par secteur
    growth_df = analyze_sector_growth(sector_data)
    
    # Générer le graphique des recommandations
    generate_recommendations_simplified(geo_data, growth_df)
    
    print("Traitement terminé!") 