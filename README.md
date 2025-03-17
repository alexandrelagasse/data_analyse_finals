# AI Trends Analysis in the Job Market

This project analyzes the adoption trends of artificial intelligence across different countries, sectors, and skill domains based on job posting data.

## Description

The analysis covers several dimensions:
- Geographic trends in AI adoption
- Business sectors most affected by AI
- Most in-demand AI skills
- Future growth predictions
- Impact of generative AI

## Prerequisites

- Python 3.7+
- pip (Python package manager)
- Virtual environment (recommended)

## Installation

1. Clone this repository (if not already done)

```bash
git clone <repo-url>
cd data_analyse_finals
```

2. Create and activate a virtual environment (recommended)

```bash
# On Windows
python -m venv monenv
monenv\Scripts\activate

# On macOS/Linux
python -m venv monenv
source monenv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirement.txt
```

## Project Structure

```
.
├── data_analysis.py      # Main analysis script
├── requirement.txt       # Project dependencies
├── dataset/              # Source data
│   ├── fig_4.2.1.csv     # Geographic data
│   ├── fig_4.2.2.csv     # Skill cluster data
│   ├── fig_4.2.3.csv     # Specific skills data
│   ├── fig_4.2.4.csv     # Generative AI jobs data
│   ├── fig_4.2.5.csv     # Generative AI skills data
│   └── fig_4.2.6.csv     # Sectoral data
├── output/               # Folder for generated visualizations
└── README.md             # This file
```

## Usage

To run the complete analysis, simply execute:

```bash
python data_analysis.py
```

This script will:
1. Load data from the various CSV files
2. Perform various analyses (geographic, sectoral, skills)
3. Generate visualizations in the `output/` folder
4. Display results and recommendations

## Features

- **Geographic Analysis**: AI adoption trends by country
- **Sectoral Analysis**: AI impacts by business sector
- **Skills Analysis**: Most in-demand AI skills
- **Predictions**: Growth projections for upcoming years
- **Recommendations**: Suggestions based on identified opportunities

## Notes

- Make sure the `output/` folder exists before running the analysis. If it doesn't exist, create it:
```bash
mkdir output
```

- All generated charts will be saved in PNG format in this folder.

## License

[Specify your license here]
