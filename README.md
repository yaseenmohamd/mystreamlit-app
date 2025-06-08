# myBuxi MCDA Framework Visualization Tool

This Streamlit application provides an interactive visualization and analysis tool for the Multi-Criteria Decision Analysis (MCDA) framework used to identify suitable regions for myBuxi's mobility-on-demand service expansion.

## Features

- **CSV Data Input**: Uses your Final_MCDA_Data(1).csv file directly
- **Interactive Weight Adjustment**: Dynamically adjust criteria weights and see how rankings change
- **Enhanced Population Filter**: Population range slider up to 25,000
- **Railway Station Filter**: Option to show only gemeinden with railway stations
- **Map Visualization**: View MCDA scores on interactive maps with color gradients
- **Sensitivity Analysis**: Compare how different weighting schemes affect gemeinde rankings
- **Data Explorer**: Analyze correlations between criteria and explore raw data

## Getting Started

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Python 3.12 Compatibility

This app is compatible with Python 3.12. If you encounter installation issues:

1. Install setuptools and wheel first:
   ```bash
   pip install setuptools wheel
   ```

2. Use the `--no-build-isolation` flag when installing requirements:
   ```bash
   pip install --no-build-isolation -r requirements.txt
   ```

## MCDA Methodology

The app follows the same normalization and weighting methodology as the Excel-based MCDA framework:

1. **Normalization**: Each criterion is normalized using min-max normalization:
   - For benefit criteria: (Value - Min) / (Max - Min)
   - For cost criteria: (Max - Value) / (Max - Min)

2. **Weighting**: Criteria are grouped into categories with assigned weights:
   - Demographic Characteristics (30%)
   - Socio-Economic Factors (25%)
   - Mobility Demand (15%)
   - Operational Feasibility (15%)
   - Public Transport Integration (10%)
   - Geographic Constraints (5%)

3. **Scoring**: The final MCDA score is calculated by summing the weighted normalized values for each criterion

## Map Visualization

For map visualization to work properly, the app looks for:
- A GeoPackage file with gemeinde boundaries (included in the data folder)
- The gemeinde names in your CSV file should match those in the GeoPackage
