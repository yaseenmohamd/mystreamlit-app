import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64
import os

st.set_page_config(
    page_title="myBuxi MCDA Framework Visualization",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

#custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2563EB;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #EFF6FF;
        border-left: 5px solid #2563EB;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .stSlider > div > div > div > div {
        background-color: #2563EB;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>myBuxi MCDA Framework Visualization</h1>", unsafe_allow_html=True)

st.markdown("""
<div class='info-box'>
This app visualizes and analyzes MCDA framework for identifying suitable regions for myBuxi expansion.
Usethe interactive controls to adjust weights and explore results.
</div>
""", unsafe_allow_html=True)

# Sidebar for controls
st.sidebar.markdown("<h2 class='sub-header'>Controls</h2>", unsafe_allow_html=True)

# Initialize session state for storing data
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.region_data = None
    st.session_state.weights = None
    st.session_state.results = None
    st.session_state.gemeinde_boundaries = None

# Function to load data from CSV
def load_data_from_csv():
    try:
       
        data_file = './Final_MCDA_Data(1).csv'
        if os.path.exists(data_file):
            df = pd.read_csv(data_file)
        
        # Ensuring all numeric columns are properly typed
        for col in df.columns:
            if col not in ['Gemeindename']:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
        
        # Default weights based on CEO interview
        weights = {
            'Demographic Characteristics': 0.30,
            'Socio-Economic Factors': 0.25,
            'Mobility Demand': 0.15,
            'Operational Feasibility': 0.15,
            'Public Transport Integration': 0.10,
            'Geographic Constraints': 0.05
        }
        
        return df, weights, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

# Function to normalize data
def normalize_data(df, benefit_cost_map):
    normalized_df = pd.DataFrame()
    normalized_df['Region'] = df['Gemeindename']
    
    for col in df.columns:
        if col in benefit_cost_map and col not in ['Gemeinde_Code', 'Gemeindename', 'Region']:
            is_benefit = benefit_cost_map[col] == 'Benefit'
            
            # Handle division by zero
            min_val = df[col].min()
            max_val = df[col].max()
            
            if max_val == min_val:
                normalized_df[col] = 1 if is_benefit else 0
            else:
                if is_benefit:
                    # For benefit criteria forUmula: (Value - Min) / (Max - Min)
                    normalized_df[col] = (df[col] - min_val) / (max_val - min_val)
                else:
                    # For cost criteria formula: (Max - Value) / (Max - Min)
                    normalized_df[col] = (max_val - df[col]) / (max_val - min_val)
    
    return normalized_df

def calculate_mcda_scores(normalized_df, weights, criteria_categories):
    
    scores_df = pd.DataFrame()
    scores_df['Region'] = normalized_df['Region']
    scores_df['Total Score'] = 0
    
    # Calculate weighted scores for each criterion
    for col in normalized_df.columns[1:]:
        if col in criteria_categories:
            category = criteria_categories[col]
            if category in weights:
               
                category_criteria_count = sum(1 for c in criteria_categories.values() if c == category)
               
                criterion_weight = weights[category] / category_criteria_count
                
                scores_df[f'{col} Score'] = normalized_df[col] * criterion_weight
                
                scores_df['Total Score'] += scores_df[f'{col} Score']
    
  #ranks calculating
        ranks = scores_df['Total Score'].rank(ascending=False, method='min', na_option='bottom')

        scores_df['Rank'] = ranks.astype('Int64')

    
    return scores_df



def to_excel_download_link(df, filename, sheet_name):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    b64 = base64.b64encode(output.getvalue()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href


region_data, weights, results = load_data_from_csv()

if region_data is not None and weights is not None:
    st.session_state.data_loaded = True
    st.session_state.region_data = region_data
    st.session_state.weights = weights
    st.session_state.results = results
    
    # benefit/cost type for each criterion
    benefit_cost_map = {
        'Gemeinde_Code': 'Neutral',
        'Population': 'Benefit',
        'Gemeindename': 'Neutral',
        'Area_km2': 'Cost',
        'Population_Density': 'Benefit',
        'Incoming_Commuters': 'Benefit',
        'Outgoing_Commuters': 'Benefit',
        'Age_0_20': 'Benefit',
        'Age_20_40': 'Benefit',
        'Age_40_65': 'Benefit',
        'Age_65_plus': 'Benefit',
        'Has_Railway_Station': 'Benefit',
        'PT_Class': 'Cost',
        'PT_Gap_20min': 'Benefit',
        'Settlement_Type': 'Benefit',
        'PT_Inadequacy_Score': 'Benefit',
        'Commuter_Flow_Total': 'Benefit'
    }
    
    # criteria to categories
    criteria_categories = {
        'Population': 'Demographic Characteristics',
        'Population_Density': 'Demographic Characteristics',
        'Age_40_65': 'Demographic Characteristics',
        'Incoming_Commuters': 'Socio-Economic Factors',
        'Outgoing_Commuters': 'Socio-Economic Factors',
        'Commuter_Flow_Total': 'Mobility Demand',
        'Area_km2': 'Operational Feasibility',
        'Settlement_Type': 'Geographic Constraints',
        'PT_Class': 'Public Transport Integration',
        'PT_Gap_20min': 'Public Transport Integration',
        'Has_Railway_Station': 'Public Transport Integration',
        'PT_Inadequacy_Score': 'Geographic Constraints'
    }
    
    st.session_state.benefit_cost_map = benefit_cost_map
    st.session_state.criteria_categories = criteria_categories
    

    try:
    
        boundary_file_data = './historisierte-administrative_grenzen_g0_2015-01-01_2056.gpkg'

        
        if os.path.exists(boundary_file_data):
            gemeinde_boundaries = gpd.read_file(boundary_file_data, layer='Communes_G0_20150101')
           
            gemeinde_boundaries = gemeinde_boundaries.rename(columns={'GDENAME': 'Region'})
            st.session_state.gemeinde_boundaries = gemeinde_boundaries
    except Exception as e:
        st.warning(f"Could not load gemeinde boundaries: {e}")
        st.session_state.gemeinde_boundaries = None


if st.session_state.data_loaded:
    st.sidebar.markdown("<h3>Adjust Criteria Weights</h3>", unsafe_allow_html=True)
    

    adjusted_weights = {}
    for category, weight in st.session_state.weights.items():
        adjusted_weights[category] = st.sidebar.slider(
            f"{category} ({weight*100:.1f}%)",
            min_value=0.0,
            max_value=1.0,
            value=weight,
            step=0.05,
            format="%.2f"
        )
    
    total_weight = sum(adjusted_weights.values())
    if total_weight > 0:
        for category in adjusted_weights:
            adjusted_weights[category] = adjusted_weights[category] / total_weight
    

    st.sidebar.markdown("<h3>Filters</h3>", unsafe_allow_html=True)
    

    railway_filter = st.sidebar.checkbox("Only show gemeinden with railway stations", value=False)

    pt_class_filter = st.sidebar.selectbox(
        "Public Transport Quality Filter",
        options=["All PT Classes", "Exclude PT Class 4", "Exclude PT Class 5", "Exclude PT Classes 4 & 5", "Only PT Classes 4 & 5"],
        index=0
    )

 
    try:
    
        population_col = pd.to_numeric(st.session_state.region_data['Population'], errors='coerce')
       
        population_col = population_col.dropna()
        
        if len(population_col) > 0:
            population_min = population_col.min()
            population_max = max(population_col.max(), 25000)  # Ensure max is at least 25000
        else:
           
            population_min = 0
            population_max = 25000
            
        population_range = st.sidebar.slider(
            "Population Range",
            min_value=int(population_min),
            max_value=int(population_max),
            value=(2500, 10000),
            step=500
        )
    except Exception as e:
        st.sidebar.warning(f"Could not determine population range: {e}")
      
        population_range = (2500, 10000)
    
 
    filtered_data = st.session_state.region_data.copy()
    
    if railway_filter:
        try:
            filtered_data = filtered_data[filtered_data['Has_Railway_Station'] == 1]
        except Exception as e:
            st.warning(f"Could not filter by railway station: {e}")
   
    if pt_class_filter != "All PT Classes":
        try:
            if pt_class_filter == "Exclude PT Class 4":
                filtered_data = filtered_data[filtered_data['PT_Class'] != 4]
            elif pt_class_filter == "Exclude PT Class 5":
                filtered_data = filtered_data[filtered_data['PT_Class'] != 5]
            elif pt_class_filter == "Exclude PT Classes 4 & 5":
                filtered_data = filtered_data[~filtered_data['PT_Class'].isin([4, 5])]
            elif pt_class_filter == "Only PT Classes 4 & 5":
                filtered_data = filtered_data[filtered_data['PT_Class'].isin([4, 5])]
        except Exception as e:
            st.warning(f"Could not filter by PT Class: {e}")

    try:
     
        filtered_data['Population'] = pd.to_numeric(filtered_data['Population'], errors='coerce')
        filtered_data = filtered_data[
            (filtered_data['Population'] >= population_range[0]) & 
            (filtered_data['Population'] <= population_range[1])
        ]
    except Exception as e:
        st.warning(f"Could not filter by population range: {e}")
    
  
    normalized_data = normalize_data(filtered_data, st.session_state.benefit_cost_map)
    
    
    scores = calculate_mcda_scores(normalized_data, adjusted_weights, st.session_state.criteria_categories)
    
  
    scores_sorted = scores.sort_values('Total Score', ascending=False).reset_index(drop=True)
    

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Results", "Map Visualization", "Sensitivity Analysis", "Data Explorer", "Additional Visualizations"])
    
    with tab1:
        st.markdown("<h2 class='sub-header'>MCDA Results</h2>", unsafe_allow_html=True)
        
     
        num_top_gemeinden = st.slider("Number of top gemeinden to display", min_value=10, max_value=100, value=20, step=10)
        
      
        st.markdown(f"<h3>Top {num_top_gemeinden} Gemeinden</h3>", unsafe_allow_html=True)
        
        # Scale scores to 0-100 for better readability
        display_scores = scores_sorted.copy()
        display_scores['Total Score'] = display_scores['Total Score'] * 100
        
    
        st.dataframe(display_scores[['Rank', 'Region', 'Total Score']].head(num_top_gemeinden).style.format({'Total Score': '{:.2f}'}))
        
     
        st.markdown(to_excel_download_link(display_scores, "mcda_results.xlsx", "Results"), unsafe_allow_html=True)
        
        # Bar chart of top 10 gemeinden
        st.markdown("<h3>Top 10 Gemeinden by MCDA Score</h3>", unsafe_allow_html=True)
        
        fig = px.bar(
            display_scores.head(10),
            x='Region',
            y='Total Score',
            color='Total Score',
            color_continuous_scale='Blues',
            labels={'Total Score': 'MCDA Score (0-100)'},
            title='Top 10 Gemeinden by MCDA Score'
        )
        fig.update_layout(xaxis_title='Gemeinde', yaxis_title='MCDA Score')
        st.plotly_chart(fig, use_container_width=True)
        
        # Pie chart of weights
        st.markdown("<h3>Criteria Weights Distribution</h3>", unsafe_allow_html=True)
        
        weights_df = pd.DataFrame({
            'Category': list(adjusted_weights.keys()),
            'Weight': list(adjusted_weights.values())
        })
        
        fig = px.pie(
            weights_df,
            values='Weight',
            names='Category',
            title='Criteria Weights Distribution',
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("<h2 class='sub-header'>Map Visualization</h2>", unsafe_allow_html=True)
        
        if st.session_state.gemeinde_boundaries is not None:
         
            gdf = st.session_state.gemeinde_boundaries.merge(scores_sorted, on='Region', how='inner')
            
          
            gdf = gdf.merge(filtered_data[['Gemeindename', 'Population', 'Incoming_Commuters', 'Outgoing_Commuters', 'PT_Inadequacy_Score', 'PT_Class', 'Settlement_Type']], 
                           left_on='Region', right_on='Gemeindename', how='left')
            
       
            gdf['Score_100'] = gdf['Total Score'] * 100
            
      
            st.markdown("<h3>MCDA Scores by Gemeinde</h3>", unsafe_allow_html=True)
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            gdf.plot(
                column='Score_100',
                cmap='Blues',
                linewidth=0.8,
                ax=ax,
                edgecolor='0.8',
                legend=True,
                legend_kwds={'label': "MCDA Score (0-100)"}
            )
            
        
            top_10 = gdf.sort_values('Total Score', ascending=False).head(10)
            for idx, row in top_10.iterrows():
                ax.annotate(
                    text=row['Region'],
                    xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                    ha='center',
                    fontsize=8,
                    color='black',
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle="round,pad=0.3")
                )
            
            ax.set_title('MCDA Scores by Gemeinde')
            ax.set_axis_off()
            st.pyplot(fig)
            
           
            st.markdown("<h3>Interactive Map</h3>", unsafe_allow_html=True)
            
        
            gdf_wgs84 = gdf.to_crs(epsg=4326)
            
          
            gdf_wgs84['geometry'] = gdf_wgs84['geometry'].simplify(0.001)
            
           
            geojson = gdf_wgs84.__geo_interface__
            
         
            settlement_map = {1: "Urban", 2: "Intermediate", 3: "Rural"}
            gdf_wgs84['Settlement_Type_Text'] = gdf_wgs84['Settlement_Type'].map(settlement_map)
            
         
            fig = px.choropleth_mapbox(
                gdf_wgs84,
                geojson=geojson,
                locations=gdf_wgs84.index,
                color='Score_100',
                color_continuous_scale='Blues',
                mapbox_style="carto-positron",
                zoom=7,
                center={"lat": 46.8, "lon": 8.2},
                opacity=0.7,
                labels={
                    'Score_100': 'MCDA Score (0-100)',
                    'Population': 'Population',
                    'Incoming_Commuters': 'Incoming Commuters',
                    'Outgoing_Commuters': 'Outgoing Commuters',
                    'PT_Inadequacy_Score': 'PT Inadequacy Score',
                    'PT_Class': 'PT Class',
                    'Settlement_Type_Text': 'Settlement Type'
                },
                hover_data={
                    'Region': True,
                    'Score_100': ':.2f',
                    'Population': True,
                    'Incoming_Commuters': True,
                    'Outgoing_Commuters': True,
                    'PT_Inadequacy_Score': ':.2f',
                    'PT_Class': True,
                    'Settlement_Type_Text': True,
                    'Rank': True
                }
            )
            fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig, use_container_width=True)
            
           
            st.markdown("<h3>Map by Different Criteria</h3>", unsafe_allow_html=True)
            
            map_criteria = st.selectbox(
                "Select criterion to visualize on map",
                options=['MCDA Score', 'Population', 'PT_Class', 'PT_Inadequacy_Score', 'Settlement_Type', 'Has_Railway_Station'],
                index=0
            )
            
            if map_criteria == 'MCDA Score':
                color_column = 'Score_100'
                color_scale = 'Blues'
                color_label = 'MCDA Score (0-100)'
            elif map_criteria == 'Population':
                color_column = 'Population'
                color_scale = 'Viridis'
                color_label = 'Population'
            elif map_criteria == 'PT_Class':
                color_column = 'PT_Class'
                color_scale = 'RdBu_r'  # Reversed so that 5 (best) is blue and 1 (worst) is red
                color_label = 'PT Class (1=least, 5=best)'
            elif map_criteria == 'PT_Inadequacy_Score':
                color_column = 'PT_Inadequacy_Score'
                color_scale = 'Reds'
                color_label = 'PT Inadequacy Score'
            elif map_criteria == 'Settlement_Type':
                color_column = 'Settlement_Type'
                color_scale = 'Viridis'
                color_label = 'Settlement Type (1=Urban, 2=Intermediate, 3=Rural)'
            elif map_criteria == 'Has_Railway_Station':
                color_column = 'Has_Railway_Station Score' #edited later
                color_scale = ['#EF4444', '#3B82F6']  # Red for No, Blue for Yes
                color_label = 'Has Railway Station (0=No, 1=Yes)'
                  # Calculate number of municipalities with/without railway station
                with_station = gdf_wgs84[gdf_wgs84['Has_Railway_Station'] == 1].shape[0]
                without_station = gdf_wgs84[gdf_wgs84['Has_Railway_Station'] == 0].shape[0]
    

                st.markdown(f"**Municipalities with railway station:** {with_station}")
                st.markdown(f"**Municipalities without railway station:** {without_station}")
            
            fig = px.choropleth_mapbox(
                gdf_wgs84,
                geojson=geojson,
                locations=gdf_wgs84.index,
                color=color_column,
                color_continuous_scale=color_scale if not isinstance(color_scale, list) else None,
                color_discrete_sequence=color_scale if isinstance(color_scale, list) else None,
                mapbox_style="carto-positron",
                zoom=7,
                center={"lat": 46.8, "lon": 8.2},
                opacity=0.7,
                labels={color_column: color_label},
                hover_data={
                    'Region': True,
                    'Score_100': ':.2f',
                    'Population': True,
                    'Incoming_Commuters': True,
                    'Outgoing_Commuters': True,
                    'PT_Inadequacy_Score': ':.2f',
                    'PT_Class': True,
                    'Settlement_Type_Text': True
                }
            )
            fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Gemeinde boundaries not available. Upload the GeoPackage file to enable map visualization.")
    
    with tab3:
        st.markdown("<h2 class='sub-header'>Sensitivity Analysis</h2>", unsafe_allow_html=True)
        
        # original vs adjusted weights
        st.markdown("<h3>Weight Comparison</h3>", unsafe_allow_html=True)
        
        weight_comparison = pd.DataFrame({
            'Category': list(st.session_state.weights.keys()),
            'Original Weight': [st.session_state.weights[cat] * 100 for cat in st.session_state.weights.keys()],
            'Adjusted Weight': [adjusted_weights[cat] * 100 for cat in adjusted_weights.keys()]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=weight_comparison['Category'],
            y=weight_comparison['Original Weight'],
            name='Original Weights',
            marker_color='#93C5FD'
        ))
        fig.add_trace(go.Bar(
            x=weight_comparison['Category'],
            y=weight_comparison['Adjusted Weight'],
            name='Adjusted Weights',
            marker_color='#2563EB'
        ))
        fig.update_layout(
            title='Original vs Adjusted Weights',
            xaxis_title='Category',
            yaxis_title='Weight (%)',
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # alternative weight scenarios
        st.markdown("<h3>Alternative Weight Scenarios</h3>", unsafe_allow_html=True)
        
        # alternative scenarios
        scenarios = {
            "Current": adjusted_weights,
            "Equal Weights": {cat: 1/6 for cat in adjusted_weights.keys()},
            "Focus on Demographics": {
                'Demographic Characteristics': 0.5,
                'Socio-Economic Factors': 0.1,
                'Mobility Demand': 0.1,
                'Operational Feasibility': 0.1,
                'Public Transport Integration': 0.1,
                'Geographic Constraints': 0.1
            },
            "Focus on Transport": {
                'Demographic Characteristics': 0.1,
                'Socio-Economic Factors': 0.1,
                'Mobility Demand': 0.1,
                'Operational Feasibility': 0.1,
                'Public Transport Integration': 0.5,
                'Geographic Constraints': 0.1
            },
            "Focus on Rural Areas": {
                'Demographic Characteristics': 0.1,
                'Socio-Economic Factors': 0.1,
                'Mobility Demand': 0.1,
                'Operational Feasibility': 0.2,
                'Public Transport Integration': 0.1,
                'Geographic Constraints': 0.4
            }
        }
        
        # results for each scenario
        scenario_results = {}
        for name, weights in scenarios.items():
            results = calculate_mcda_scores(normalized_data, weights, st.session_state.criteria_categories)
            results = results.sort_values('Total Score', ascending=False).reset_index(drop=True)
            scenario_results[name] = results
        
        # Create comparison chart
        scenario_weights = pd.DataFrame({
            'Category': list(adjusted_weights.keys()),
            **{name: [weights[cat] * 100 for cat in adjusted_weights.keys()] for name, weights in scenarios.items()}
        })
        
        fig = px.bar(
            scenario_weights,
            x='Category',
            y=list(scenarios.keys()),
            barmode='group',
            title='Weight Comparison Across Scenarios',
            labels={'value': 'Weight (%)'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
      
        num_gemeinden_stability = st.slider("Number of gemeinden for rank stability analysis", min_value=10, max_value=50, value=20, step=5)
        
        # Rank stability analysis
        st.markdown("<h3>Rank Stability Analysis</h3>", unsafe_allow_html=True)
        
        #  showing top gemeinden across all scenarios
        top_gemeinden = set()
        for name, results in scenario_results.items():
            top_gemeinden.update(results.head(num_gemeinden_stability)['Region'].tolist())
        
     
        rank_comparison = pd.DataFrame({'Region': list(top_gemeinden)})
        for name, results in scenario_results.items():
            rank_map = dict(zip(results['Region'], results['Rank']))
            rank_comparison[f'{name} Rank'] = rank_comparison['Region'].map(lambda x: rank_map.get(x, float('nan')))
        
       
        rank_comparison = rank_comparison.sort_values('Current Rank').head(num_gemeinden_stability)
        
       
        st.dataframe(rank_comparison)
        
       
        rank_data = []
        for _, row in rank_comparison.head(10).iterrows():
            for scenario in scenarios.keys():
                if not pd.isna(row[f'{scenario} Rank']):
                    rank_data.append({
                        'Region': row['Region'],
                        'Scenario': scenario,
                        'Rank': row[f'{scenario} Rank']
                    })
        
        rank_df = pd.DataFrame(rank_data)
        
        fig = px.line(
            rank_df,
            x='Scenario',
            y='Rank',
            color='Region',
            markers=True,
            title='Rank Stability Across Scenarios',
            labels={'Rank': 'Rank Position'},
        )
        fig.update_layout(yaxis={'autorange': 'reversed'})  # Reverse y-axis so rank 1 is at the top
        st.plotly_chart(fig, use_container_width=True)
        
      
        st.markdown("<h3>Criteria Contribution Analysis</h3>", unsafe_allow_html=True)
        
       
        top_gemeinden_list = display_scores.head(20)['Region'].tolist()
        selected_gemeinde = st.selectbox("Select gemeinde for criteria contribution analysis", options=top_gemeinden_list)
        
       
        gemeinde_scores = scores[scores['Region'] == selected_gemeinde].iloc[0]
        
      
        criteria_scores = {}
        for col in gemeinde_scores.index:
            if col.endswith(' Score') and col != 'Total Score':
                criteria_name = col.replace(' Score', '')
                criteria_scores[criteria_name] = gemeinde_scores[col]
        
     
        criteria_df = pd.DataFrame({
            'Criterion': list(criteria_scores.keys()),
            'Score': list(criteria_scores.values())
        })
        
     
        criteria_df = criteria_df.sort_values('Score', ascending=False)
        

        fig = px.bar(
            criteria_df,
            x='Criterion',
            y='Score',
            title=f'Criteria Contribution for {selected_gemeinde}',
            color='Score',
            color_continuous_scale='Blues'
        )
        fig.update_layout(xaxis_title='Criterion', yaxis_title='Weighted Score')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("<h2 class='sub-header'>Data Explorer</h2>", unsafe_allow_html=True)
        
      
        st.markdown("<h3>Raw Data</h3>", unsafe_allow_html=True)
        st.dataframe(filtered_data)
        
    
        st.markdown("<h3>Correlation Matrix</h3>", unsafe_allow_html=True)
        
        try:
          
            numeric_data = filtered_data.select_dtypes(include=[np.number])
            
          
            corr = numeric_data.corr()
            
      
            fig = px.imshow(
                corr,
                text_auto='.2f',
                color_continuous_scale='RdBu_r',
                title='Correlation Matrix of Criteria'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Scatter plot matrix
            st.markdown("<h3>Scatter Plot Matrix</h3>", unsafe_allow_html=True)
            
            # Let user select columns for scatter plot
            selected_columns = st.multiselect(
                "Select columns for scatter plot matrix",
                options=numeric_data.columns.tolist(),
                default=numeric_data.columns[:4].tolist() if len(numeric_data.columns) >= 4 else numeric_data.columns.tolist()
            )
            
            if selected_columns:
                fig = px.scatter_matrix(
                    filtered_data,
                    dimensions=selected_columns,
                    color='Has_Railway_Station' if 'Has_Railway_Station' in filtered_data.columns else None,
                    title='Scatter Plot Matrix'
                )
                fig.update_layout(height=800)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate correlation matrix or scatter plots: {e}")
            st.info("This may be due to non-numeric data or other data quality issues.")
    
    with tab5:
        st.markdown("<h2 class='sub-header'>Additional Visualizations</h2>", unsafe_allow_html=True)
        
     
        st.markdown("<h3>Population vs MCDA Score</h3>", unsafe_allow_html=True)
        
             
        scatter_data = filtered_data.merge(display_scores[['Region', 'Total Score', 'Rank']], left_on='Gemeindename', right_on='Region', how='inner')
        
       
        scatter_data_clean = scatter_data.dropna(subset=['Population', 'Total Score', 'PT_Inadequacy_Score', 'Has_Railway_Station'])
        
       
        if len(scatter_data_clean) > 0:
            fig = px.scatter(
                scatter_data_clean,
                x='Population',
                y='Total Score',
                color='Has_Railway_Station',
                size='PT_Inadequacy_Score',
                hover_name='Gemeindename',
                hover_data=['Rank', 'PT_Class', 'Settlement_Type'],
                title='Population vs MCDA Score',
                labels={
                    'Population': 'Population',
                    'Total Score': 'MCDA Score (0-100)',
                    'Has_Railway_Station': 'Has Railway Station',
                    'PT_Inadequacy_Score': 'PT Inadequacy Score'
                },
                color_discrete_sequence=['#EF4444', '#3B82F6']  # Red for No, Blue for Yes
            )
            
            fig.add_vline(x=2500, line_dash="dash", line_color="green", annotation_text="Min Ideal")
            fig.add_vline(x=10000, line_dash="dash", line_color="green", annotation_text="Max Ideal")
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No complete data available for scatter plot visualization.")

        
        # Settlement Type Distribution
        st.markdown("<h3>Settlement Type Distribution in Top Gemeinden</h3>", unsafe_allow_html=True)
        
    
        num_gemeinden_analysis = st.slider("Number of top gemeinden for analysis", min_value=10, max_value=100, value=50, step=10)
        
     
        top_gemeinden_data = filtered_data.merge(
            display_scores[['Region', 'Rank']].head(num_gemeinden_analysis), 
            left_on='Gemeindename', 
            right_on='Region', 
            how='inner'
        )
        
       
        settlement_map = {1: "Urban", 2: "Intermediate", 3: "Rural"}
        top_gemeinden_data['Settlement_Type_Text'] = top_gemeinden_data['Settlement_Type'].map(settlement_map)
        
        
        fig = px.pie(
            top_gemeinden_data,
            names='Settlement_Type_Text',
            title=f'Settlement Type Distribution in Top {num_gemeinden_analysis} Gemeinden',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # PT Class Distribution
        st.markdown("<h3>PT Class Distribution in Top Gemeinden</h3>", unsafe_allow_html=True)
        
        # Create histogram
        fig = px.histogram(
            top_gemeinden_data,
            x='PT_Class',
            title=f'PT Class Distribution in Top {num_gemeinden_analysis} Gemeinden',
            labels={'PT_Class': 'PT Class (1=least, 5=best)'},
            color_discrete_sequence=['#3B82F6']
        )
        st.plotly_chart(fig, use_container_width=True)
        
       
        st.markdown("<h3>Railway Station Impact Analysis</h3>", unsafe_allow_html=True)
        
       
        railway_comparison = display_scores.merge(
            filtered_data[['Gemeindename', 'Has_Railway_Station']], 
            left_on='Region', 
            right_on='Gemeindename', 
            how='inner'
        )
        
        railway_stats = railway_comparison.groupby('Has_Railway_Station')['Total Score'].agg(['mean', 'count']).reset_index()
        railway_stats['Has_Railway_Station'] = railway_stats['Has_Railway_Station'].map({0: 'No', 1: 'Yes'})
        railway_stats.columns = ['Has Railway Station', 'Average MCDA Score', 'Count']
        
        # Create bar chart
        fig = px.bar(
            railway_stats,
            x='Has Railway Station',
            y='Average MCDA Score',
            color='Has Railway Station',
            text='Count',
            title='Average MCDA Score by Railway Station Presence',
            labels={'Average MCDA Score': 'Average MCDA Score (0-100)'},
            color_discrete_sequence=['#EF4444', '#3B82F6']  # Red for No, Blue for Yes
        )
        fig.update_traces(texttemplate='%{text} gemeinden', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        # Commuter Flow Analysis
        st.markdown("<h3>Commuter Flow Analysis</h3>", unsafe_allow_html=True)
        
        # scatter plot
        fig = px.scatter(
            filtered_data.merge(display_scores[['Region', 'Total Score', 'Rank']], left_on='Gemeindename', right_on='Region', how='inner'),
            x='Incoming_Commuters',
            y='Outgoing_Commuters',
            color='Total Score',
            size='Population',
            hover_name='Gemeindename',
            hover_data=['Rank', 'PT_Class', 'Settlement_Type'],
            title='Incoming vs Outgoing Commuters',
            labels={
                'Incoming_Commuters': 'Incoming Commuters',
                'Outgoing_Commuters': 'Outgoing Commuters',
                'Total Score': 'MCDA Score (0-100)',
                'Population': 'Population'
            },
            color_continuous_scale='Blues'
        )
        
       
        max_val = max(
            filtered_data['Incoming_Commuters'].max(),
            filtered_data['Outgoing_Commuters'].max()
        )
        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode='lines',
                line=dict(color='gray', dash='dash'),
                name='Equal In/Out'
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        
        st.markdown("<h3>PT Inadequacy vs PT Class</h3>", unsafe_allow_html=True)
        
        
        fig = px.scatter(
            filtered_data.merge(display_scores[['Region', 'Total Score', 'Rank']], left_on='Gemeindename', right_on='Region', how='inner'),
            x='PT_Class',
            y='PT_Inadequacy_Score',
            color='Total Score',
            size='Population',
            hover_name='Gemeindename',
            hover_data=['Rank', 'Settlement_Type', 'Has_Railway_Station'],
            title='PT Class vs PT Inadequacy Score',
            labels={
                'PT_Class': 'PT Class (1=least, 5=best)',
                'PT_Inadequacy_Score': 'PT Inadequacy Score',
                'Total Score': 'MCDA Score (0-100)',
                'Population': 'Population'
            },
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        
        st.markdown("<h3>Geographic Distribution of Top Gemeinden</h3>", unsafe_allow_html=True)
        
        if st.session_state.gemeinde_boundaries is not None:
        
            top_n = st.slider("Number of top gemeinden to highlight", min_value=10, max_value=100, value=30, step=10)
            
            # Merge scores with boundaries
            gdf = st.session_state.gemeinde_boundaries.merge(scores_sorted, on='Region', how='inner')
            
            # Scale scores to 0-100 for better readability
            gdf['Score_100'] = gdf['Total Score'] * 100
            
            # Create map highlighting top N
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # Plot all gemeinden in light gray
            gdf.plot(
                color='lightgray',
                linewidth=0.5,
                ax=ax,
                edgecolor='white'
            )
            
            # 
            top_n_gdf = gdf.sort_values('Total Score', ascending=False).head(top_n)
            top_n_gdf.plot(
                column='Score_100',
                cmap='Blues',
                linewidth=0.8,
                ax=ax,
                edgecolor='white',
                legend=True,
                legend_kwds={'label': f"MCDA Score (0-100) - Top {top_n}"}
            )
            
            # Add labels for top 10
            top_10 = gdf.sort_values('Total Score', ascending=False).head(10)
            for idx, row in top_10.iterrows():
                ax.annotate(
                    text=f"{row['Region']} (#{row['Rank']})",
                    xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                    ha='center',
                    fontsize=8,
                    color='black',
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle="round,pad=0.3")
                )
            
            ax.set_title(f'Geographic Distribution of Top {top_n} Gemeinden')
            ax.set_axis_off()
            st.pyplot(fig)
        else:
            st.warning("Gemeinde boundaries not available. Upload the GeoPackage file to enable map visualization.")
else:
    
    st.markdown("""
    <div class='info-box'>
    <h3>Getting Started</h3>
    <p>This app uses the Final_MCDA_Data(1).csv file to visualize and analyze MCDA results.</p>
    <p>The CSV file should contain the following columns:</p>
    <ul>
        <li><strong>Gemeinde_Code</strong>: Unique identifier for each gemeinde</li>
        <li><strong>Gemeindename</strong>: Name of the gemeinde</li>
        <li><strong>Population</strong>: Population of the gemeinde</li>
        <li><strong>Area_km2</strong>: Area in square kilometers</li>
        <li><strong>Population_Density</strong>: Population density</li>
        <li><strong>Incoming_Commuters</strong>: Number of incoming commuters</li>
        <li><strong>Outgoing_Commuters</strong>: Number of outgoing commuters</li>
        <li><strong>Age_*</strong>: Age demographics</li>
        <li><strong>Has_Railway_Station</strong>: Whether the gemeinde has a railway station</li>
        <li><strong>PT_Class</strong>: Public transport class (1=least, 5=best)</li>
        <li><strong>PT_Gap_20min</strong>: Public transport gap</li>
        <li><strong>Settlement_Type</strong>: Type of settlement (1=Urban, 2=Intermediate, 3=Rural)</li>
        <li><strong>PT_Inadequacy_Score</strong>: Public transport inadequacy score</li>
        <li><strong>Commuter_Flow_Total</strong>: Total commuter flow</li>
    </ul>
    <p>Once the data is loaded, you can:</p>
    <ul>
        <li>Adjust criteria weights using the sliders</li>
        <li>Filter gemeinden by railway station presence and population</li>
        <li>View results in tables and charts</li>
        <li>Explore the data on maps (if gemeinde boundaries are available)</li>
        <li>Perform sensitivity analysis</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 3rem; padding: 1rem; background-color: #000080;">
    <p>myBuxi MCDA Framework Visualization Tool</p>
    <p style="font-size: 0.8rem;">Created for expansion planning and sensitivity analysis</p>
</div>
""", unsafe_allow_html=True)
