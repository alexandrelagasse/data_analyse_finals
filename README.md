# Workplace Absenteeism Data Analysis

This project analyzes workplace absenteeism data and generates visualizations to identify trends and factors influencing employee absences.

## Prerequisites

To run this project, you need Python 3.7+ and the following libraries:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## Installation

1. Clone this repository or download the source files

2. Install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Project Structure

```
project/
│
├── dataset/
│   └── Absenteeism_data.csv    # Absenteeism dataset file
│
├── output/                     # Folder containing generated visualizations
│
├── main.py                     # Main data analysis script
│
└── README.md                   # This file
```

## Running the Project

1. Make sure the dataset file is present in the `dataset/` folder

2. Run the main script:

```bash
python main.py
```

3. Visualizations will be generated in the `output/` folder

## Generated Visualizations

The script generates the following visualizations:

1. `evolution_absenteisme.png` - Evolution and projection of absenteeism over time
2. `correlation_matrix.png` - Correlation matrix between variables
3. `raisons_absence.png` - Top 10 reasons for absence
4. `absence_par_mois.png` - Average hours of absence by month
5. `facteurs_importants.png` - Relative importance of factors influencing absenteeism
6. `distribution_age.png` - Distribution of employee ages
7. `opportunite_reduction.png` - Potential productivity gains under different scenarios
8. `impact_attendu.png` - Expected impact of intervention program on absenteeism

## Contact

For any questions regarding this project, please contact [your@email.com]. 