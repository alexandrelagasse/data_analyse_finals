import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Configuration for better visualization
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.family'] = 'sans-serif'

# 1. Loading datasets
def load_data():
    print("Loading datasets...")
    geo_data = pd.read_csv('dataset/fig_4.2.1.csv')
    skills_cluster = pd.read_csv('dataset/fig_4.2.2.csv')
    skills_specific = pd.read_csv('dataset/fig_4.2.3.csv')
    gen_ai_jobs = pd.read_csv('dataset/fig_4.2.4.csv')
    gen_ai_skills = pd.read_csv('dataset/fig_4.2.5.csv')
    sector_data = pd.read_csv('dataset/fig_4.2.6.csv')
    
    # Cleaning geographic data
    geo_data['adoption_rate'] = geo_data['AI job postings (% of all job postings)'].str.rstrip('%').astype(float) / 100
    
    # Cleaning skill cluster data
    skills_cluster['adoption_rate'] = skills_cluster['AI job postings (% of all job postings)'].str.rstrip('%').astype(float) / 100
    
    # Cleaning sector data
    sector_data['adoption_rate'] = sector_data['AI Job Postings (% of All Job Postings)'].str.rstrip('%').astype(float) / 100
    
    # Cleaning generative AI skills data
    if 'Skill share in AI job postings (%)' in gen_ai_skills.columns:
        gen_ai_skills['skill_share'] = gen_ai_skills['Skill share in AI job postings (%)'].str.rstrip('%').astype(float) / 100
    
    return geo_data, skills_cluster, skills_specific, gen_ai_jobs, gen_ai_skills, sector_data

# 2. Geographic analysis
def analyze_geo_trends(geo_data):
    print("\nAnalyzing trends by country...")
    
    # Top countries by adoption rate in the latest year
    latest_year = geo_data['Year'].max()
    top_countries = geo_data[geo_data['Year'] == latest_year].sort_values('adoption_rate', ascending=False).head(10)
    
    # Fastest growing countries
    growth_by_country = []
    
    for country in geo_data['Geographic area'].unique():
        country_data = geo_data[geo_data['Geographic area'] == country].sort_values('Year')
        if len(country_data) >= 2:
            first_year = country_data.iloc[0]
            last_year = country_data.iloc[-1]
            years_diff = last_year['Year'] - first_year['Year']
            
            if first_year['adoption_rate'] > 0 and years_diff > 0:
                growth = (last_year['adoption_rate'] - first_year['adoption_rate']) / first_year['adoption_rate']
                annual_growth = ((1 + growth) ** (1/years_diff)) - 1
                
                growth_by_country.append({
                    'country': country,
                    'total_growth': growth,
                    'annual_growth': annual_growth,
                    'current_adoption': last_year['adoption_rate'],
                    'years_analyzed': years_diff
                })
    
    growth_df = pd.DataFrame(growth_by_country)
    fastest_growing = growth_df.sort_values('annual_growth', ascending=False).head(5)
    
    print("\nTop 5 countries by current adoption:")
    print(top_countries[['Geographic area', 'adoption_rate']])
    print("\nTop 5 fastest growing countries:")
    print(fastest_growing[['country', 'annual_growth', 'current_adoption']])
    
    return top_countries, fastest_growing

def plot_country_trends(geo_data, countries_to_plot):
    print("\nCreating country trend chart...")
    plt.figure(figsize=(12, 8))
    
    for country in countries_to_plot:
        country_data = geo_data[geo_data['Geographic area'] == country].sort_values('Year')
        plt.plot(country_data['Year'], country_data['adoption_rate'], marker='o', linewidth=2, label=country)
    
    plt.title('Evolution of AI Adoption by Country', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Adoption Rate (% of job postings)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/country_trends.png', dpi=300)
    plt.show()

def predict_country_growth(geo_data, countries, years_ahead=3):
    print(f"\nPrédiction de croissance à {years_ahead} ans pour les pays sélectionnés...")
    
    predictions = {}
    
    for country in countries:
        country_data = geo_data[geo_data['Geographic area'] == country].sort_values('Year')
        
        if len(country_data) >= 3:  # Au moins 3 points pour une prédiction fiable
            X = country_data[['Year']].values
            y = country_data['adoption_rate'].values
            
            # Modèle polynomial pour capturer les tendances non-linéaires
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)
            
            model = LinearRegression()
            model.fit(X_poly, y)
            
            # Prédire pour les années futures
            last_year = country_data['Year'].max()
            future_years = np.array([[last_year + i + 1] for i in range(years_ahead)])
            future_years_poly = poly.transform(future_years)
            future_predictions = model.predict(future_years_poly)
            
            predictions[country] = {
                'years': [last_year + i + 1 for i in range(years_ahead)],
                'predicted_rates': future_predictions,
                'r2_score': model.score(X_poly, y)  # Ajouter le R² pour évaluer la qualité du modèle
            }
    
    # Visualisation des prédictions avec un design amélioré
    plt.figure(figsize=(14, 9))
    
    # Palette de couleurs pour différencier historique/prédiction
    colors = plt.cm.tab10(np.linspace(0, 1, len(predictions)))
    
    # Zone ombrée pour indiquer la période de prédiction
    min_year = min([min(geo_data[geo_data['Geographic area'] == country]['Year']) for country in predictions.keys()])
    max_hist_year = max([max(geo_data[geo_data['Geographic area'] == country]['Year']) for country in predictions.keys()])
    plt.axvspan(max_hist_year + 0.5, max_hist_year + years_ahead + 0.5, color='lavender', alpha=0.3, label='Période de prédiction')
    
    # Fond stylisé
    plt.gca().set_facecolor('#f8f9fa')
    
    # Tracer les données historiques et les prédictions
    for i, (country, pred) in enumerate(predictions.items()):
        # Données historiques
        country_data = geo_data[geo_data['Geographic area'] == country].sort_values('Year')
        plt.plot(country_data['Year'], country_data['adoption_rate'], 
                marker='o', linewidth=3, color=colors[i], label=f"{country} (données historiques)")
        
        # Prédictions
        plt.plot(pred['years'], pred['predicted_rates'], 
                marker='x', linestyle='--', linewidth=3, color=colors[i], 
                label=f"{country} (prédiction, R²={pred['r2_score']:.2f})")
        
        # Zone de confiance (estimation simplifiée)
        upper_bound = pred['predicted_rates'] * 1.1
        lower_bound = pred['predicted_rates'] * 0.9
        plt.fill_between(pred['years'], lower_bound, upper_bound, color=colors[i], alpha=0.1)
    
    # Annotations pour aider à l'interprétation
    max_country = max(predictions.items(), key=lambda x: x[1]['predicted_rates'][-1])[0]
    max_rate = max(predictions.items(), key=lambda x: x[1]['predicted_rates'][-1])[1]['predicted_rates'][-1]
    plt.annotate(f"{max_country} : leader prévu en {max_hist_year + years_ahead}",
                xy=(max_hist_year + years_ahead, max_rate),
                xytext=(max_hist_year + years_ahead - 1, max_rate * 1.1),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='black'),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8),
                fontsize=12)
    
    plt.title(f'Prédiction de l\'adoption de l\'IA à {years_ahead} ans', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Année', fontsize=14, fontweight='bold')
    plt.ylabel('Taux d\'adoption prédit (% des offres d\'emploi)', fontsize=14, fontweight='bold')
    
    # Amélioration de la légende
    legend = plt.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1, 1))
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    
    # Grille et graduation améliorées
    plt.grid(True, alpha=0.3, linestyle='-')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
    
    # Annotations pour les méthodes
    plt.figtext(0.5, 0.01, "Modèle: Régression polynomiale de degré 2", 
                ha="center", fontsize=10, bbox={"boxstyle":"round", "alpha":0.1})
    
    plt.tight_layout()
    plt.savefig('output/country_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return predictions

# 3. Sector analysis
def analyze_sector_trends(sector_data):
    print("\nAnalyzing trends by sector...")
    
    # Latest available year
    latest_year = sector_data['Year'].max()
    
    # Top current sectors
    current_top_sectors = sector_data[sector_data['Year'] == latest_year].sort_values('adoption_rate', ascending=False).head(10)
    
    # Growth by sector
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
    fastest_growing_sectors = growth_df.sort_values('annual_growth', ascending=False).head(5)
    
    print("\nTop 5 sectors by current adoption:")
    print(current_top_sectors[['Sector', 'adoption_rate']].head())
    print("\nTop 5 fastest growing sectors:")
    print(fastest_growing_sectors[['sector', 'annual_growth', 'current_adoption']])
    
    return current_top_sectors, fastest_growing_sectors, growth_df

def plot_sector_adoption(sector_data):
    print("\nCreating sector adoption chart...")
    latest_year = sector_data['Year'].max()
    latest_data = sector_data[sector_data['Year'] == latest_year]
    
    plt.figure(figsize=(14, 10))
    bars = sns.barplot(x='adoption_rate', y='Sector', data=latest_data.sort_values('adoption_rate', ascending=False))
    
    # Add values on bars
    for i, v in enumerate(latest_data.sort_values('adoption_rate', ascending=False)['adoption_rate']):
        bars.text(v + 0.001, i, f"{v:.2%}", va='center')
    
    plt.title(f'AI Adoption by Sector in {latest_year}', fontsize=16)
    plt.xlabel('Adoption Rate (% of job postings)', fontsize=14)
    plt.tight_layout()
    plt.savefig('output/sector_adoption.png', dpi=300)
    plt.show()

# 4. Skills analysis
def analyze_skills_trends(skills_cluster, skills_specific):
    print("\nAnalyzing in-demand skills...")
    
    # Evolution of skill groups
    skill_clusters = skills_cluster['Skill cluster'].unique()
    
    plt.figure(figsize=(14, 8))
    
    for cluster in skill_clusters:
        cluster_data = skills_cluster[skills_cluster['Skill cluster'] == cluster].sort_values('Year')
        plt.plot(cluster_data['Year'], cluster_data['adoption_rate'], marker='o', linewidth=2, label=cluster)
    
    plt.title('Evolution of AI Skills by Cluster', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Adoption Rate (% of job postings)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/skill_clusters.png', dpi=300)
    plt.show()
    
    # Convert 'Year' type if necessary
    if not pd.api.types.is_numeric_dtype(skills_specific['Year']):
        skills_specific['Year'] = pd.to_numeric(skills_specific['Year'], errors='coerce')
    
    # Top specific skills
    latest_year = skills_specific['Year'].max()
    top_skills = skills_specific[skills_specific['Year'] == latest_year].sort_values('Number of AI Job Postings', ascending=False).head(10)
    
    plt.figure(figsize=(12, 8))
    bars = sns.barplot(x='Number of AI Job Postings', y='Skill', data=top_skills)
    
    # Add values on bars
    for i, v in enumerate(top_skills['Number of AI Job Postings']):
        bars.text(v + 10, i, str(v), va='center')
    
    plt.title('Top 10 Most In-Demand AI Skills', fontsize=16)
    plt.tight_layout()
    plt.savefig('output/top_skills.png', dpi=300)
    plt.show()
    
    return skill_clusters, top_skills

# 5. Generative AI analysis
def analyze_gen_ai(gen_ai_jobs, gen_ai_skills):
    print("\nAnalyzing generative AI...")
    
    # Evolution of generative AI job postings
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Year', y='Number of AI Job Postings', data=gen_ai_jobs, marker='o', linewidth=3)
    plt.title('Evolution of Generative AI Job Postings', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Number of Postings', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/gen_ai_jobs.png', dpi=300)
    plt.show()
    
    # Distribution of generative AI skills
    plt.figure(figsize=(12, 6))
    bars = sns.barplot(x='skill_share', y='Generative AI skill', data=gen_ai_skills.sort_values('skill_share', ascending=False))
    
    # Add values on bars
    for i, v in enumerate(gen_ai_skills.sort_values('skill_share', ascending=False)['skill_share']):
        bars.text(v + 0.01, i, f"{v:.2%}", va='center')
    
    plt.title('Distribution of Generative AI Skills', fontsize=16)
    plt.xlabel('Share in Job Postings (%)', fontsize=14)
    plt.tight_layout()
    plt.savefig('output/gen_ai_skills.png', dpi=300)
    plt.show()

# 6. Opportunity identification
def create_opportunity_matrix(sector_growth_df):
    print("\nCréation de la matrice d'opportunités...")
    
    plt.figure(figsize=(14, 10))
    
    # Calcul du score d'opportunité
    sector_growth_df['opportunity_score'] = sector_growth_df['current_adoption'] * sector_growth_df['annual_growth']
    
    # Normaliser les scores pour la taille des bulles
    min_score = sector_growth_df['opportunity_score'].min()
    max_score = sector_growth_df['opportunity_score'].max()
    sector_growth_df['bubble_size'] = 300 + ((sector_growth_df['opportunity_score'] - min_score) / (max_score - min_score)) * 1500
    
    scatter = plt.scatter(
        sector_growth_df['current_adoption'], 
        sector_growth_df['annual_growth'],
        s=sector_growth_df['bubble_size'], 
        alpha=0.7,
        c=sector_growth_df['opportunity_score'],
        cmap='viridis',
        edgecolors='black',
        linewidths=1
    )
    
    # Amélioration de l'apparence des quadrants
    mean_adoption = sector_growth_df['current_adoption'].mean()
    mean_growth = sector_growth_df['annual_growth'].mean()
    
    plt.axhline(y=mean_growth, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    plt.axvline(x=mean_adoption, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    
    # Quadrants plus marqués
    plt.fill_between([0, mean_adoption], [mean_growth, mean_growth], [2, 2], color='lightgreen', alpha=0.1)
    plt.fill_between([mean_adoption, max(sector_growth_df['current_adoption'])*1.1], [mean_growth, mean_growth], [2, 2], color='gold', alpha=0.1)
    plt.fill_between([0, mean_adoption], [mean_growth, -1], [0, 0], color='lightgray', alpha=0.1)
    plt.fill_between([mean_adoption, max(sector_growth_df['current_adoption'])*1.1], [mean_growth, -1], [0, 0], color='lightskyblue', alpha=0.1)
    
    # Amélioration des étiquettes des quadrants
    plt.text(mean_adoption*0.5, mean_growth + (max(sector_growth_df['annual_growth']) - mean_growth)*0.25, 
             "OPPORTUNITÉS ÉMERGENTES", 
             ha='center', fontsize=14, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.text(mean_adoption + (max(sector_growth_df['current_adoption']) - mean_adoption)*0.5, 
             mean_growth + (max(sector_growth_df['annual_growth']) - mean_growth)*0.25, 
             "LEADERS EN CROISSANCE", 
             ha='center', fontsize=14, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.text(mean_adoption*0.5, mean_growth*0.5, 
             "MARCHÉS EN RETARD", 
             ha='center', fontsize=14, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.text(mean_adoption + (max(sector_growth_df['current_adoption']) - mean_adoption)*0.5, 
             mean_growth*0.5, 
             "MARCHÉS MATURES", 
             ha='center', fontsize=14, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Ajout des étiquettes pour les secteurs clés
    top_opportunities = sector_growth_df.sort_values('opportunity_score', ascending=False).head(3)
    for idx, row in top_opportunities.iterrows():
        plt.annotate(
            row['sector'], 
            (row['current_adoption'], row['annual_growth']),
            xytext=(15, 15), 
            textcoords='offset points',
            fontsize=12,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='black')
        )
    
    plt.xlabel('Taux d\'adoption actuel', fontsize=14, fontweight='bold')
    plt.ylabel('Croissance annuelle', fontsize=14, fontweight='bold')
    plt.title('Matrice d\'opportunités d\'investissement par secteur', fontsize=18, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='-')
    
    cbar = plt.colorbar(scatter, label='Potentiel de disruption')
    cbar.set_label('Potentiel de disruption (Adoption × Croissance)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('output/opportunity_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return top_opportunities

# 7. Integrated analysis and recommendations
def generate_recommendations(geo_data, sector_growth_df, skills_data):
    print("\nGénération des recommandations d'investissement...")
    
    # Calcul de scores d'opportunité par pays
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
                
                # Score d'opportunité: combinaison d'adoption actuelle et croissance
                opportunity_score = last_year['adoption_rate'] * annual_growth
                
                country_growth.append({
                    'country': country,
                    'current_adoption': last_year['adoption_rate'],
                    'annual_growth': annual_growth,
                    'opportunity_score': opportunity_score
                })
    
    country_growth_df = pd.DataFrame(country_growth)
    top_country_opportunities = country_growth_df.sort_values('opportunity_score', ascending=False).head(3)
    
    # Meilleurs secteurs d'investissement
    if 'opportunity_score' not in sector_growth_df.columns:
        sector_growth_df['opportunity_score'] = sector_growth_df['current_adoption'] * sector_growth_df['annual_growth']
    
    top_sector_opportunities = sector_growth_df.sort_values('opportunity_score', ascending=False).head(3)
    
    # Résumé des recommandations
    print("\nRésumé des recommandations d'investissement:")
    print("\n1. Marchés géographiques recommandés:")
    for i, (idx, row) in enumerate(top_country_opportunities.iterrows(), 1):
        print(f"  {i}. {row['country']} - Taux d'adoption actuel: {row['current_adoption']:.2%}, Croissance annuelle: {row['annual_growth']:.2%}")
    
    print("\n2. Secteurs recommandés pour l'investissement:")
    for i, (idx, row) in enumerate(top_sector_opportunities.iterrows(), 1):
        print(f"  {i}. {row['sector']} - Taux d'adoption actuel: {row['current_adoption']:.2%}, Croissance annuelle: {row['annual_growth']:.2%}")
    
    # Visualisation des recommandations - Design amélioré
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), facecolor='#f8f9fa')
    fig.suptitle('RECOMMANDATIONS D\'INVESTISSEMENT STRATÉGIQUES', fontsize=22, fontweight='bold', y=0.98)
    
    # Palette de couleurs harmonisée
    country_colors = plt.cm.Blues(np.linspace(0.6, 0.9, len(top_country_opportunities)))
    sector_colors = plt.cm.Greens(np.linspace(0.6, 0.9, len(top_sector_opportunities)))
    
    # Graphique des pays recommandés avec design amélioré
    countries = top_country_opportunities['country']
    country_scores = top_country_opportunities['opportunity_score']
    
    bars1 = ax1.barh(countries, country_scores, color=country_colors, edgecolor='black', linewidth=1)
    ax1.set_title('Top 3 des pays à fort potentiel', fontsize=18, fontweight='bold', pad=20)
    ax1.set_xlabel('Score d\'opportunité', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()  # Pour que le pays #1 soit en haut
    
    # Ajouter des annotations sur les barres
    for i, (bar, value) in enumerate(zip(bars1, country_scores)):
        growth = top_country_opportunities.iloc[i]['annual_growth']
        adoption = top_country_opportunities.iloc[i]['current_adoption']
        ax1.text(bar.get_width() + bar.get_width()*0.01, bar.get_y() + bar.get_height()/2, 
                f"Score: {value:.4f}\nCroissance: {growth:.1%}\nAdoption: {adoption:.2%}", 
                va='center', fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
    
    # Graphique des secteurs recommandés avec design amélioré
    sectors = top_sector_opportunities['sector']
    sector_scores = top_sector_opportunities['opportunity_score']
    
    bars2 = ax2.barh(sectors, sector_scores, color=sector_colors, edgecolor='black', linewidth=1)
    ax2.set_title('Top 3 des secteurs à fort potentiel', fontsize=18, fontweight='bold', pad=20)
    ax2.set_xlabel('Score d\'opportunité', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()  # Pour que le secteur #1 soit en haut
    
    # Ajouter des annotations sur les barres
    for i, (bar, value) in enumerate(zip(bars2, sector_scores)):
        growth = top_sector_opportunities.iloc[i]['annual_growth']
        adoption = top_sector_opportunities.iloc[i]['current_adoption']
        ax2.text(bar.get_width() + bar.get_width()*0.01, bar.get_y() + bar.get_height()/2, 
                f"Score: {value:.4f}\nCroissance: {growth:.1%}\nAdoption: {adoption:.2%}", 
                va='center', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
    
    # Formater les axes
    for ax in [ax1, ax2]:
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        # Éliminer les bordures
        for spine in ax.spines.values():
            spine.set_visible(False)
        # Ajouter un fond de couleur claire aux graphiques
        ax.set_facecolor('#f8f9fa')
    
    # Ajouter une note méthodologique
    fig.text(0.5, 0.01, 
            "Méthodologie: Le score d'opportunité combine le taux d'adoption actuel et le taux de croissance annuel\n" +
            "Source: Stanford AI Index, analyse par régression polynomiale de degré 2",
            ha='center', fontsize=12, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('output/investment_recommendations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return top_country_opportunities, top_sector_opportunities


def create_scenario_analysis(top_country_opportunities, top_sector_opportunities):
    print("\nCréation de l'analyse de scénarios...")
    
    # Définir les scénarios
    scenarios = [
        {
            'name': "Investissement dans l'IA pour l'administration publique",
            'pros': [
                "Taux de croissance extraordinaire (88,6%)",
                "Marché important et stable",
                "Moins de concurrence que dans les secteurs traditionnels"
            ],
            'cons': [
                "Cycles de vente longs",
                "Contraintes réglementaires",
                "Résistance culturelle au changement"
            ],
            'potential_score': 8.5,
            'risk_score': 6.8,
            'time_to_market': 24,  # mois
            'applications': [
                "Systèmes de prédiction pour services publics",
                "Automatisation administrative",
                "Chatbots d'assistance aux citoyens"
            ]
        },
        {
            'name': "Expansion sur le marché espagnol de l'IA",
            'pros': [
                "Croissance annuelle impressionnante (25,7%)",
                "Coûts d'implantation plus faibles qu'aux États-Unis",
                "Accès au marché européen"
            ],
            'cons': [
                "Marché plus petit que les États-Unis",
                "Compétition avec d'autres hubs européens",
                "Adaptation linguistique et culturelle nécessaire"
            ],
            'potential_score': 7.5,
            'risk_score': 5.2,
            'time_to_market': 12,  # mois
            'applications': [
                "Solutions d'IA adaptées aux PME espagnoles",
                "Plateformes NLP multilingues",
                "Solutions IA pour les secteurs clés locaux"
            ]
        },
        {
            'name': "IA générative pour l'éducation",
            'pros': [
                "Secteur éducatif en croissance (6,0%)",
                "Demande de personnalisation de l'apprentissage",
                "Potentiel de marché mondial"
            ],
            'cons': [
                "Budgets contraints dans l'éducation publique",
                "Questions éthiques concernant l'IA en éducation",
                "Cycle d'adoption potentiellement plus lent"
            ],
            'potential_score': 6.8,
            'risk_score': 4.5,
            'time_to_market': 18,  # mois
            'applications': [
                "Systèmes de tutorat adaptatif basés sur l'IA générative",
                "Outils d'évaluation automatisée",
                "Création de contenu éducatif personnalisé"
            ]
        }
    ]
    
    # Créer un dataframe pour le graphique
    scenario_df = pd.DataFrame({
        'Scénario': [s['name'] for s in scenarios],
        'Potentiel': [s['potential_score'] for s in scenarios],
        'Risque': [s['risk_score'] for s in scenarios],
        'Délai de mise sur le marché (mois)': [s['time_to_market'] for s in scenarios]
    })
    
    # Créer une matrice de comparaison
    fig = plt.figure(figsize=(15, 12), facecolor='#f8f9fa')
    
    # Layout avec GridSpec
    gs = plt.GridSpec(2, 2, height_ratios=[3, 2], width_ratios=[3, 2])
    
    # Graphique principal - Matrice Potentiel vs Risque
    ax1 = fig.add_subplot(gs[0, 0])
    scatter = ax1.scatter(
        scenario_df['Risque'], 
        scenario_df['Potentiel'],
        s=scenario_df['Délai de mise sur le marché (mois)']*20,
        alpha=0.7,
        c=range(len(scenarios)),
        cmap='viridis',
        edgecolors='black',
        linewidths=1.5
    )
    
    # Ajouter les noms des scénarios
    for i, row in scenario_df.iterrows():
        ax1.annotate(
            row['Scénario'].split(' ')[0] + '...',  # Nom court pour clarté
            (row['Risque'], row['Potentiel']),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=12,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='black')
        )
    
    ax1.set_xlabel('Risque (score sur 10)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Potentiel (score sur 10)', fontsize=14, fontweight='bold')
    ax1.set_title('Matrice de comparaison des scénarios d\'investissement', fontsize=18, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    
    # Légende pour la taille des bulles
    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6, num=3, 
                                             func=lambda s: s/20, fmt="{x:.0f} mois")
    legend1 = ax1.legend(handles, labels, loc="upper right", title="Délai de mise sur marché")
    
    # Tableau de pros et cons
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    
    # Création d'un tableau texte pour le premier scénario
    scenario_text = f"ANALYSE DÉTAILLÉE: {scenarios[0]['name']}\n\n"
    scenario_text += "AVANTAGES:\n"
    for pro in scenarios[0]['pros']:
        scenario_text += f"✓ {pro}\n"
    
    scenario_text += "\nINCONVÉNIENTS:\n"
    for con in scenarios[0]['cons']:
        scenario_text += f"✗ {con}\n"
    
    scenario_text += "\nAPPLICATIONS IA/ML RECOMMANDÉES:\n"
    for app in scenarios[0]['applications']:
        scenario_text += f"• {app}\n"
    
    ax2.text(0, 0.5, scenario_text, va='center', fontsize=12,
            bbox=dict(boxstyle="round,pad=1", fc="#f0f8ff", ec="black", alpha=0.8))
    
    # Graphique de comparaison des scores
    ax3 = fig.add_subplot(gs[1, :])
    
    # Réorganiser les données pour barplot
    categories = ['Potentiel', 'Risque', 'Délai de mise sur le marché (mois)']
    data_dict = {
        scenarios[0]['name'].split(' ')[0] + '...': [scenarios[0]['potential_score'], scenarios[0]['risk_score'], scenarios[0]['time_to_market']],
        scenarios[1]['name'].split(' ')[0] + '...': [scenarios[1]['potential_score'], scenarios[1]['risk_score'], scenarios[1]['time_to_market']],
        scenarios[2]['name'].split(' ')[0] + '...': [scenarios[2]['potential_score'], scenarios[2]['risk_score'], scenarios[2]['time_to_market']]
    }
    
    # Créer un dataframe pour le graphique
    comparison_df = pd.DataFrame(data_dict, index=categories)
    
    # Normaliser les valeurs pour une meilleure comparaison visuelle
    comparison_df.loc['Délai de mise sur le marché (mois)'] = comparison_df.loc['Délai de mise sur le marché (mois)'] / 10  # Diviser par 10 pour l'échelle
    
    # Plot
    comparison_df.T.plot(kind='bar', ax=ax3, width=0.7, colormap='viridis', 
                        edgecolor='black', linewidth=1)
    
    ax3.set_title('Comparaison des scénarios d\'investissement', fontsize=16, fontweight='bold')
    ax3.set_ylabel('Score (Échelle 0-10)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_axisbelow(True)
    
    # Légende
    ax3.legend(title='Métriques', loc='upper left', bbox_to_anchor=(1, 1))
    
    # Note sur le délai
    ax3.annotate('*Délai de mise sur le marché divisé par 10 pour l\'échelle', 
                xy=(0.98, 0.02), xycoords='axes fraction', 
                fontsize=10, ha='right', style='italic')
    
    # Titre global
    fig.suptitle('ANALYSE DES SCÉNARIOS D\'INVESTISSEMENT EN IA', fontsize=22, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('output/scenario_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return scenarios

# 8. Run complete analysis
def run_complete_analysis():
    print("Démarrage de l'analyse complète de l'adoption de l'IA...")
    
    # Chargement des données
    geo_data, skills_cluster, skills_specific, gen_ai_jobs, gen_ai_skills, sector_data = load_data()
    
    # Analyse géographique
    top_countries, fastest_growing_countries = analyze_geo_trends(geo_data)
    plot_country_trends(geo_data, top_countries['Geographic area'].head(5))
    country_predictions = predict_country_growth(geo_data, top_countries['Geographic area'].head(3))
    
    # Analyse sectorielle
    top_sectors, fastest_growing_sectors, sector_growth_df = analyze_sector_trends(sector_data)
    plot_sector_adoption(sector_data)
    
    # Analyse des compétences
    skill_clusters, top_skills = analyze_skills_trends(skills_cluster, skills_specific)
    
    # Analyse de l'IA générative
    analyze_gen_ai(gen_ai_jobs, gen_ai_skills)
    
    # Matrice d'opportunités
    top_opportunities = create_opportunity_matrix(sector_growth_df)
    
    # Recommandations d'investissement
    top_country_opportunities, top_sector_opportunities = generate_recommendations(geo_data, sector_growth_df, skills_specific)
    
    # Analyse de scénarios (nouvelle fonction)
    scenarios = create_scenario_analysis(top_country_opportunities, top_sector_opportunities)
    
    print("\nAnalyse complète terminée. Tous les graphiques ont été sauvegardés.")
    
    return {
        'top_countries': top_countries,
        'fastest_growing_countries': fastest_growing_countries,
        'top_sectors': top_sectors,
        'fastest_growing_sectors': fastest_growing_sectors,
        'top_skills': top_skills,
        'top_opportunities': top_opportunities,
        'country_predictions': country_predictions,
        'top_country_opportunities': top_country_opportunities,
        'top_sector_opportunities': top_sector_opportunities,
        'scenarios': scenarios
    }



# Main function
if __name__ == "__main__":
    results = run_complete_analysis()