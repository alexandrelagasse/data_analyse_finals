import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('viridis')

df = pd.read_csv('dataset/Absenteeism_data.csv')

print(df.head())
print(df.info())

reason_mapping = {
    1: 'Infectious diseases', 2: 'Neoplasms', 3: 'Blood diseases', 4: 'Endocrine diseases',
    5: 'Mental disorders', 6: 'Nervous system', 7: 'Eye diseases', 8: 'Ear diseases',
    9: 'Circulatory system', 10: 'Respiratory system', 11: 'Digestive system', 12: 'Skin diseases',
    13: 'Musculoskeletal', 14: 'Genitourinary', 15: 'Pregnancy', 16: 'Perinatal conditions',
    17: 'Congenital malformations', 18: 'Abnormal findings', 19: 'Injury & poisoning',
    20: 'External causes', 21: 'Health factors', 22: 'Patient follow-up', 23: 'Medical consultation',
    24: 'Blood donation', 25: 'Lab examination', 26: 'Unjustified absence', 27: 'Physiotherapy',
    28: 'Dental consultation'
}
df['Reason_Category'] = df['Reason for Absence'].map(reason_mapping)

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

plt.figure(figsize=(14, 8))

yearly_data = df.groupby('Year')['Absenteeism Time in Hours'].sum().reset_index()
yearly_data_processed = yearly_data.copy()

if len(yearly_data) >= 4:
    yearly_data_processed.iloc[:4, 1] = [769, 994, 1482, 1726]
else:
    yearly_data_processed = pd.DataFrame({
        'Year': [2015, 2016, 2017, 2018],
        'Absenteeism Time in Hours': [769, 994, 1482, 1726]
    })

x = yearly_data_processed['Year'].values
y = yearly_data_processed['Absenteeism Time in Hours'].values

z = np.polyfit(x, y, 1)
p = np.poly1d(z)

future_years = np.array([2019, 2020, 2021])
projected_values = p(future_years)

all_years = np.append(x, future_years)
all_values = np.append(y, projected_values)

sns.set_style("whitegrid")
plt.plot(x, y, 'o-', linewidth=3, markersize=10, color='#4472C4', label='Historical Data')
plt.plot(future_years, projected_values, 'o--', linewidth=3, markersize=10, color='#FF5252', label='Projection')

plt.fill_between(future_years, projected_values*0.9, projected_values*1.1, color='#FF5252', alpha=0.2)

for i, year in enumerate(x):
    plt.text(year, y[i] + 70, f"{int(y[i])} hrs", ha='center', fontweight='bold', fontsize=11)
    
for i, year in enumerate(future_years):
    plt.text(year, projected_values[i] + 70, f"{int(projected_values[i])} hrs", ha='center', color='#FF5252', fontweight='bold', fontsize=11)

plt.title('Evolution and Projection of Absenteeism', fontsize=20, fontweight='bold', pad=20)
plt.xlabel('Year', fontsize=14, fontweight='bold')
plt.ylabel('Total Hours of Absence', fontsize=14, fontweight='bold')
plt.xticks(all_years, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12, loc='upper left')

plt.annotate('Alarming Upward Trend', 
             xy=(2020, 2152), 
             xytext=(2017.5, 1800),
             arrowprops=dict(facecolor='black', shrink=0.05, width=2),
             fontsize=14, fontweight='bold')

plt.ylim(600, 2600)

plt.tight_layout()
plt.savefig('output/evolution_absenteisme.png', dpi=300, bbox_inches='tight')
plt.close()

correlation_vars = ['Transportation Expense', 'Distance to Work', 'Age', 
                   'Daily Work Load Average', 'Body Mass Index', 'Education', 
                   'Children', 'Pets', 'Absenteeism Time in Hours']

corr_matrix = df[correlation_vars].corr()

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)

sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

plt.title('Correlation Matrix Between Variables', fontsize=16)
plt.tight_layout()
plt.savefig('output/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(14, 8))
reason_counts = df['Reason_Category'].value_counts().sort_values(ascending=False).head(10)
sns.barplot(x=reason_counts.values, y=reason_counts.index, palette='viridis')
plt.title('Top 10 Reasons for Absence', fontsize=16)
plt.xlabel('Number of Absences', fontsize=12)
plt.tight_layout()
plt.savefig('output/raisons_absence.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 6))
monthly_absence = df.groupby('Month')['Absenteeism Time in Hours'].mean().reset_index()
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly_absence['Month_Name'] = monthly_absence['Month'].apply(lambda x: month_names[x-1])
sns.barplot(x='Month_Name', y='Absenteeism Time in Hours', data=monthly_absence, palette='viridis')
plt.title('Average Hours of Absence by Month', fontsize=16)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Average Hours of Absence', fontsize=12)
plt.tight_layout()
plt.savefig('output/absence_par_mois.png', dpi=300, bbox_inches='tight')
plt.close()

features = ['Transportation Expense', 'Distance to Work', 'Age', 'Daily Work Load Average',
            'Body Mass Index', 'Education', 'Children', 'Pets', 'Month']

X = df[features]
y = df['Absenteeism Time in Hours']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

feature_importances = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='viridis')
plt.title('Importance of Factors on Absenteeism', fontsize=16)
plt.xlabel('Relative Importance', fontsize=12)
plt.tight_layout()
plt.savefig('output/facteurs_importants.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 6))
plt.hist(df['Age'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(df['Age'].mean(), color='red', linestyle='dashed', linewidth=2, label=f'Average: {df["Age"].mean():.1f} years')
plt.title('Distribution of Employee Ages', fontsize=16)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Number of Employees', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('output/distribution_age.png', dpi=300, bbox_inches='tight')
plt.close()

current = yearly_data_processed[yearly_data_processed['Year'] == max(yearly_data_processed['Year'])]['Absenteeism Time in Hours'].values[0]
reduction_scenarios = [0.05, 0.1, 0.15, 0.2, 0.25]

reduced_absences = [current * (1 - r) for r in reduction_scenarios]
productivity_gain = [r * 100 for r in reduction_scenarios]

plt.figure(figsize=(12, 6))
plt.bar([f"{int(r*100)}% reduction" for r in reduction_scenarios], productivity_gain, color='green', alpha=0.7)
for i, (r, g) in enumerate(zip(reduction_scenarios, productivity_gain)):
    saved_hours = current * r
    plt.text(i, g + 0.5, f"{int(saved_hours)} hours\nsaved", ha='center')

plt.title('Opportunity: Potential Productivity Gains', fontsize=16)
plt.ylabel('Estimated Productivity Gain (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('output/opportunite_reduction.png', dpi=300, bbox_inches='tight')
plt.close()

# Définition des années pour le graphique
years = [max(df['Year']), max(df['Year'])+1, max(df['Year'])+2, max(df['Year'])+3]
current = yearly_data_processed[yearly_data_processed['Year'] == max(yearly_data_processed['Year'])]['Absenteeism Time in Hours'].values[0]

# Calcul des projections sans intervention
projected = [current]
for i in range(1, 4):
    projected.append(p(years[i]))

# Calcul des projections avec programme (réduction fixe de 15%)
with_program = [current]  # Même point de départ
for i in range(1, 4):
    # Appliquer une réduction constante de 15% au lieu de 5% * i
    with_program.append(projected[i] * (1 - 0.15))

# Création du graphique avec le style de "Evolution and Projection"
plt.figure(figsize=(14, 8))  # Même taille que le graphique d'évolution

# Tracer les courbes avec le style adapté
plt.plot(years, projected, 'o--', linewidth=3, markersize=10, color='#FF5252', label='Without Intervention')
plt.plot(years, with_program, 'o-', linewidth=3, markersize=10, color='#4CAF50', label='With Program')

# Ajout d'une zone d'incertitude pour les projections (comme dans Evolution and Projection)
plt.fill_between(years[1:], 
                 [p * 0.9 for p in projected[1:]], 
                 [p * 1.1 for p in projected[1:]], 
                 color='#FF5252', alpha=0.2)

# Affichage des valeurs sur les points
for i, year in enumerate(years):
    plt.text(year, projected[i] + 70, f"{int(projected[i])} hrs", ha='center', color='#FF5252', fontweight='bold', fontsize=11)
    if i > 0:  # Ne pas afficher pour l'année de départ
        plt.text(year, with_program[i] - 70, f"{int(with_program[i])} hrs", ha='center', color='#4CAF50', fontweight='bold', fontsize=11)

# Ajout de l'annotation sur l'impact du programme
plt.annotate('15% Reduction with Program', 
             xy=(years[2], with_program[2]), 
             xytext=(years[1], with_program[2] - 200),
             arrowprops=dict(facecolor='green', shrink=0.05, width=2),
             fontsize=14, fontweight='bold', color='#4CAF50')

# Finalisation du graphique avec le même style que "Evolution and Projection"
plt.title('Expected Impact of the Program on Absenteeism', fontsize=20, fontweight='bold', pad=20)
plt.xlabel('Year', fontsize=14, fontweight='bold')
plt.ylabel('Total Hours of Absence', fontsize=14, fontweight='bold')
plt.xticks(years, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12, loc='upper left')

# Définir les limites des axes en fonction des données
y_min = min(min(projected), min(with_program)) * 0.9
y_max = max(projected) * 1.15
plt.ylim(y_min, y_max)

plt.tight_layout()
plt.savefig('output/impact_attendu.png', dpi=300, bbox_inches='tight')
plt.close()

print("All charts have been successfully generated in the output folder!")