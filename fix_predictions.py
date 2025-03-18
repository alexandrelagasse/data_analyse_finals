import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Configuration for better visualization
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.family'] = 'sans-serif'

def load_geo_data():
    """Loading and cleaning geographical data only"""
    print("Loading geographical data...")
    geo_data = pd.read_csv('dataset/fig_4.2.1.csv')
    geo_data['adoption_rate'] = geo_data['AI job postings (% of all job postings)'].str.rstrip('%').astype(float) / 100
    return geo_data

def predict_country_growth_enhanced(geo_data, countries, years_ahead=3):
    """
    Creates an enhanced chart for country growth predictions
    with a clear separation between historical data and predictions
    """
    print(f"Generating predictions for {', '.join(countries)}...")
    
    # Limit to 5 countries for clarity
    if len(countries) > 5:
        keep_countries = []
        for priority_country in ['United States', 'Spain', 'Singapore', 'France', 'Sweden']:
            if priority_country in countries:
                keep_countries.append(priority_country)
        
        # Complete with other countries if necessary
        remaining_slots = 5 - len(keep_countries)
        if remaining_slots > 0:
            for country in countries:
                if country not in keep_countries:
                    keep_countries.append(country)
                    remaining_slots -= 1
                    if remaining_slots == 0:
                        break
        
        countries = keep_countries[:5]
    
    predictions = {}
    r2_scores = {}
    
    for country in countries:
        country_data = geo_data[geo_data['Geographic area'] == country].sort_values('Year')
        
        if len(country_data) >= 3:  # At least 3 points for prediction
            X = country_data[['Year']].values
            y = country_data['adoption_rate'].values
            
            # Polynomial model
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)
            
            model = LinearRegression()
            model.fit(X_poly, y)
            
            # Store R2 score
            r2_scores[country] = model.score(X_poly, y)
            
            # Predict for future years
            last_year = country_data['Year'].max()
            future_years = np.array([[last_year + i + 1] for i in range(years_ahead)])
            future_years_poly = poly.transform(future_years)
            future_predictions = model.predict(future_years_poly)
            
            # Ensure predictions are not negative
            future_predictions = np.maximum(future_predictions, 0)
            
            predictions[country] = {
                'years': [last_year + i + 1 for i in range(years_ahead)],
                'predicted_rates': future_predictions
            }
    
    # Create an enhanced chart with strong visual contrast
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
    
    # Enhanced color palette with more vivid colors for better contrast
    country_colors = {
        'United States': '#0052CC',  # Rich blue
        'Spain': '#E31A1C',          # Bright red
        'Singapore': '#00994C',      # Rich green
        'France': '#7928CA',         # Bright purple
        'Sweden': '#FF8C00',         # Bright orange
        'Canada': '#E76F51',         # Terracotta
        'United Kingdom': '#6A0DAD'  # Deep purple
    }
    
    # Default colors for countries not included in our palette - more vivid alternatives
    default_colors = ['#1E88E5', '#FF6D00', '#43A047', '#D81B60', '#8E24AA']
    
    # Clear indication of prediction period
    max_hist_year = max([max(geo_data[geo_data['Geographic area'] == country]['Year']) for country in predictions.keys()])
    
    # Create a much more visible demarcation between historical and prediction zones
    # Use a stronger background for the prediction area
    pred_zone = ax.axvspan(max_hist_year + 0.5, max_hist_year + years_ahead + 0.5, 
                          color='#ECECFC', alpha=0.5, zorder=0)  # Light bluish background
    
    # Add a vertical separation line
    ax.axvline(x=max_hist_year + 0.5, color='#666666', linestyle='-', linewidth=1.5, alpha=0.7)
    
    # Add a more visible "Predictions" label
    ax.text(max_hist_year + years_ahead/2, 0.01, 
            "Predictions", ha='center', va='bottom', 
            fontsize=13, fontweight='bold', color='#333333',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='#999999', boxstyle="round,pad=0.3"))
    
    # Plot historical data and predictions with accentuated visual differences
    for i, country in enumerate(predictions.keys()):
        # Get the color for this country
        color = country_colors.get(country, default_colors[i % len(default_colors)])
        
        # Historical data - enhanced style
        country_data = geo_data[geo_data['Geographic area'] == country].sort_values('Year')
        hist_line = ax.plot(country_data['Year'], country_data['adoption_rate'] * 100, 
                marker='o', linewidth=2.5, color=color, 
                label=f"{country}",
                markersize=7, 
                markerfacecolor='white', 
                markeredgewidth=1.5, 
                markeredgecolor=color)
        
        # Get coordinates of the last historical point
        last_hist_x = country_data['Year'].iloc[-1]
        last_hist_y = country_data['adoption_rate'].iloc[-1] * 100
        
        # Create predictions with a visibly different style
        # Increase the width of the prediction line and use a different dash pattern
        pred_line = ax.plot(predictions[country]['years'], 
                predictions[country]['predicted_rates'] * 100, 
                linestyle=(0, (3, 1, 1, 1)), linewidth=3, color=color, 
                marker='s', markersize=6, markerfacecolor=color, markeredgewidth=1,
                markeredgecolor='white', alpha=0.85)
        
        # Connect the last historical point to the first prediction point with a dotted line
        first_pred_x = predictions[country]['years'][0]
        first_pred_y = predictions[country]['predicted_rates'][0] * 100
        
        ax.plot([last_hist_x, first_pred_x], [last_hist_y, first_pred_y], 
                linestyle=':', linewidth=1.5, color=color, alpha=0.6)
        
        # Add R² as an end point annotation for all countries
        # Last prediction point
        last_x = predictions[country]['years'][-1]
        last_y = predictions[country]['predicted_rates'][-1] * 100
        
        ax.annotate(
            f"R²={r2_scores[country]:.2f}", 
            (last_x, last_y),
            xytext=(8, 0), 
            textcoords='offset points',
            fontsize=10,
            fontweight='bold',
            color=color,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor=None, boxstyle="round,pad=0.1")
        )
    
    # Enhanced title and labels
    ax.set_title('AI Adoption 3-Year Prediction', fontsize=16, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Adoption Rate (%)', fontsize=12, fontweight='bold')
    
    # Percentage formatting
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=1))
    
    # Enhanced grid
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    
    # Add a custom legend with separate entries for history and prediction
    # First get existing legend handles and labels
    handles, labels = ax.get_legend_handles_labels()
    
    # Create a custom legend - first countries, then indicators for historical vs prediction
    legend = ax.legend(handles, labels, 
                     loc='best', 
                     frameon=True, 
                     framealpha=0.9, 
                     fontsize=10,
                     title="Countries",
                     title_fontsize=12)
    
    # Add an annotation to explain the visual coding
    plt.figtext(0.15, 0.01, 
              "—— Historical Data  ······ Predictions", 
              ha="left", fontsize=9)
    
    # Methodological note
    plt.figtext(0.75, 0.01, 
            "Polynomial regression (degree 2) | Source: Stanford AI Index", 
            ha="right", fontsize=8, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig('output/country_predictions_enhanced.png', dpi=300, bbox_inches='tight')
    
    print(f"Chart saved to 'output/country_predictions_enhanced.png'")
    
    return predictions, fig

# Main entry point
if __name__ == "__main__":
    # Load data
    geo_data = load_geo_data()
    
    # Countries to predict
    countries_to_predict = ['United States', 'Spain', 'Singapore', 'France', 'Sweden']
    
    # Generate predictions
    predict_country_growth_enhanced(geo_data, countries_to_predict, years_ahead=3)
    
    print("Processing completed!") 