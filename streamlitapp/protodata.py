import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.title("Machine Learning App")

# File uploader
st.info("Please upload your dataset.")
with st.expander("Data Upload"):
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "parquet", "feather"])
    if uploaded_file is not None:
        try:
            # Get file extension as fallback
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            # Determine file type by MIME type first, then fallback to extension
            if uploaded_file.type == "csv" or file_extension == "csv":
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" or file_extension == "xlsx":
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.type == "application/x-parquet" or file_extension == "parquet":
                df = pd.read_parquet(uploaded_file)
            elif uploaded_file.type == "application/octet-stream" or file_extension == "feather":
                df = pd.read_feather(uploaded_file)
            else:
                st.error(f"Unsupported file type: {uploaded_file.type}")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.warning("Please ensure you're uploading a valid file in the supported format.")

        st.dataframe(df)


# Data summary
with st.expander("Data Summary"):
    if 'df' in locals():
        st.write("### Dataset Overview")
        st.write(f"**Number of rows: {df.shape[0]}**")
        st.write(f"**Number of columns: {df.shape[1]}**")
        st.write("### Sample Data")
        st.dataframe(df.sample(5))
        st.write("### Data Types")
        st.dataframe(pd.DataFrame(df.dtypes, columns=["Data Type"]))
        st.write("### Statistical Summary for categorical columns")
        st.dataframe(df.describe(include=['object', 'string', 'category']).T)
        st.write("### Statistical Summary for numerical columns")
        st.dataframe(df.describe(include=['number']).T)

        st.write("### Unique Values")
        unique_counts = {col: df[col].nunique() for col in df.columns}
        st.dataframe(pd.DataFrame(unique_counts.items(), columns=["Column", "Unique Values"]))

        st.write("### Unique Values List")
        unique_df = pd.DataFrame({col: [df[col].unique().tolist()] for col in df.select_dtypes(include=['object', 'string', 'category']).columns}).T.rename(columns={0: 'unique_values', 'index': 'column_name'})
        st.dataframe(unique_df)

        st.write("### Missing Values")
        st.dataframe(pd.DataFrame(df.isnull().sum(), columns=["Missing Values"]))

    else:
        st.warning("Please upload a dataset to see the summary.")

# Data visualization
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

with st.expander(" Data Visualization"):
    if 'df' in locals():
        st.write("### Smart Visualization Generator")
        
        # Analyze data types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        boolean_cols = df.select_dtypes(include=['bool']).columns.tolist()
        
        st.write(f"**Data Summary:** {len(numeric_cols)} numeric, {len(categorical_cols)} categorical, {len(datetime_cols)} datetime, {len(boolean_cols)} boolean columns")
        
        # Column selection with smart defaults
        all_columns = df.columns.tolist()
        
        # Tabs for different visualization types
        viz_tabs = st.tabs(["🔍 Single Variable", "📈 Two Variables", "🌐 Multi-Variable", "📊 Statistical", "🎨 Advanced"])
        
        # ===== SINGLE VARIABLE ANALYSIS =====
        with viz_tabs[0]:
            st.subheader("Single Variable Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                single_col = st.selectbox("Select a column to analyze", all_columns, key="single_var")
            with col2:
                plot_types = ["Auto", "Histogram", "Box Plot", "Violin Plot", "Bar Chart", "Pie Chart", "Count Plot"]
                plot_type = st.selectbox("Plot Type", plot_types, key="single_plot")
            
            if single_col:
                col_data = df[single_col].dropna()
                
                # Determine best plot automatically
                if plot_type == "Auto":
                    if single_col in numeric_cols:
                        if col_data.nunique() > 20:
                            plot_type = "Histogram"
                        else:
                            plot_type = "Bar Chart"
                    else:
                        if col_data.nunique() > 10:
                            plot_type = "Count Plot"
                        else:
                            plot_type = "Pie Chart"
                
                # Create plots based on selection
                if plot_type == "Histogram" and single_col in numeric_cols:
                    fig = px.histogram(df, x=single_col, nbins=100, title=f"Distribution of {single_col}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif plot_type == "Box Plot" and single_col in numeric_cols:
                    fig = px.box(df, y=single_col, title=f"Box Plot of {single_col}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif plot_type == "Violin Plot" and single_col in numeric_cols:
                    fig = px.violin(df, y=single_col, title=f"Violin Plot of {single_col}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif plot_type in ["Bar Chart", "Count Plot"]:
                    value_counts = col_data.value_counts().head(20)  # Top 20 values
                    fig = px.bar(x=value_counts.index, y=value_counts.values, 
                                title=f"Value Counts for {single_col}",
                                labels={'x': single_col, 'y': 'Count'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif plot_type == "Pie Chart":
                    value_counts = col_data.value_counts().head(10)  # Top 10 values
                    fig = px.pie(values=value_counts.values, names=value_counts.index, 
                                title=f"Distribution of {single_col}")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show statistics
                with st.expander(f"Statistics for {single_col}"):
                    if single_col in numeric_cols:
                        stats_df = pd.DataFrame({
                            'Statistic': ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'],
                            'Value': [
                                col_data.count(),
                                col_data.mean(),
                                col_data.std(),
                                col_data.min(),
                                col_data.quantile(0.25),
                                col_data.median(),
                                col_data.quantile(0.75),
                                col_data.max()
                            ]
                        })
                        st.dataframe(stats_df, use_container_width=True)
                    else:
                        st.write("**Value Counts:**")
                        st.dataframe(col_data.value_counts().to_frame('Count'), use_container_width=True)
        
        # ===== TWO VARIABLE ANALYSIS =====
        with viz_tabs[1]:
            st.subheader("Two Variable Analysis")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                x_col = st.selectbox("X-axis column", all_columns, key="two_var_x")
            with col2:
                y_col = st.selectbox("Y-axis column", all_columns, key="two_var_y")
            with col3:
                two_var_plots = ["Auto", "Scatter", "Line", "Bar", "Box", "Violin", "Heatmap"]
                two_plot_type = st.selectbox("Plot Type", two_var_plots, key="two_plot")
            
            if x_col and y_col and x_col != y_col:
                # Auto-determine plot type
                if two_plot_type == "Auto":
                    if x_col in numeric_cols and y_col in numeric_cols:
                        two_plot_type = "Scatter"
                    elif x_col in categorical_cols and y_col in numeric_cols:
                        two_plot_type = "Box"
                    elif x_col in numeric_cols and y_col in categorical_cols:
                        two_plot_type = "Bar"
                    else:
                        two_plot_type = "Heatmap"
                
                # Create plots
                if two_plot_type == "Scatter":
                    fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif two_plot_type == "Line":
                    fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif two_plot_type == "Bar":
                    if x_col in categorical_cols and y_col in numeric_cols:
                        grouped = df.groupby(x_col)[y_col].mean().reset_index()
                        fig = px.bar(grouped, x=x_col, y=y_col, title=f"Average {y_col} by {x_col}")
                    else:
                        fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif two_plot_type == "Box":
                    fig = px.box(df, x=x_col, y=y_col, title=f"{y_col} distribution by {x_col}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif two_plot_type == "Violin":
                    fig = px.violin(df, x=x_col, y=y_col, title=f"{y_col} distribution by {x_col}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif two_plot_type == "Heatmap":
                    # Create a crosstab for categorical variables
                    crosstab = pd.crosstab(df[x_col], df[y_col])
                    fig = px.imshow(crosstab, title=f"Relationship between {x_col} and {y_col}")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Correlation if both numeric
                if x_col in numeric_cols and y_col in numeric_cols:
                    correlation = df[x_col].corr(df[y_col])
                    st.metric("**Correlation Coefficient**", f"{correlation:.3f}",border=True)
        
        # ===== MULTI-VARIABLE ANALYSIS =====
        with viz_tabs[2]:
            st.subheader("Multi-Variable Analysis")
            
            # Select multiple columns
            multi_cols = st.multiselect("Select columns for analysis", all_columns, key="multi_var")
            
            if len(multi_cols) >= 2:
                multi_viz_type = st.radio("Visualization Type", 
                                        ["Correlation Matrix", "Pair Plot", "Parallel Coordinates", "3D Scatter"], 
                                        horizontal=True)
                
                if multi_viz_type == "Correlation Matrix" and len([col for col in multi_cols if col in numeric_cols]) >= 2:
                    numeric_subset = [col for col in multi_cols if col in numeric_cols]
                    corr_matrix = df[numeric_subset].corr()
                    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                  title="Correlation Matrix")
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif multi_viz_type == "Pair Plot":
                    if len(multi_cols) <= 5:  # Limit to prevent overcrowding
                        fig = px.scatter_matrix(df[multi_cols], title="Pair Plot")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Please select 5 or fewer columns for pair plot")
                        
                elif multi_viz_type == "Parallel Coordinates":
                    # Add color column if available
                    color_col = None
                    if categorical_cols:
                        color_col = st.selectbox("Color by (optional)", [None] + categorical_cols)
                    
                    fig = px.parallel_coordinates(df, dimensions=multi_cols, color=color_col,
                                                title="Parallel Coordinates Plot")
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif multi_viz_type == "3D Scatter" and len([col for col in multi_cols if col in numeric_cols]) >= 3:
                    numeric_subset = [col for col in multi_cols if col in numeric_cols][:3]
                    color_col = categorical_cols[0] if categorical_cols else None
                    
                    fig = px.scatter_3d(df, x=numeric_subset[0], y=numeric_subset[1], 
                                      z=numeric_subset[2], color=color_col,
                                      title="3D Scatter Plot")
                    st.plotly_chart(fig, use_container_width=True)
        
        # ===== STATISTICAL PLOTS =====
        with viz_tabs[3]:
            st.subheader("Statistical Analysis")
            
            stat_type = st.selectbox("Statistical Plot Type", 
                                   ["Distribution Comparison", "Regression Plot", "Residual Plot", "QQ Plot"])
            
            if stat_type == "Distribution Comparison" and len(numeric_cols) >= 1:
                compare_col = st.selectbox("Select numeric column", numeric_cols)
                group_col = st.selectbox("Group by (optional)", [None] + categorical_cols)
                
                if group_col:
                    fig = px.histogram(df, x=compare_col, color=group_col, 
                                     title=f"Distribution of {compare_col} by {group_col}")
                else:
                    fig = px.histogram(df, x=compare_col, title=f"Distribution of {compare_col}")
                st.plotly_chart(fig, use_container_width=True)
                
            elif stat_type == "Regression Plot" and len(numeric_cols) >= 2:
                reg_x = st.selectbox("X variable", numeric_cols, key="reg_x")
                reg_y = st.selectbox("Y variable", numeric_cols, key="reg_y")
                
                if reg_x != reg_y:
                    fig = px.scatter(df, x=reg_x, y=reg_y, trendline="ols",
                                   title=f"Regression: {reg_y} vs {reg_x}")
                    st.plotly_chart(fig, use_container_width=True)
        
        # ===== ADVANCED PLOTS =====
        with viz_tabs[4]:
            st.subheader("Advanced Visualizations")
            
            advanced_type = st.selectbox("Advanced Plot Type", 
                                       ["Sunburst", "Treemap", "Sankey", "Radar Chart", "Waterfall"])
            
            if advanced_type == "Sunburst" and len(categorical_cols) >= 2:
                path_cols = st.multiselect("Select categorical columns for hierarchy", 
                                         categorical_cols[:3], key="sunburst")
                if len(path_cols) >= 2:
                    fig = px.sunburst(df, path=path_cols, title="Sunburst Chart")
                    st.plotly_chart(fig, use_container_width=True)
                    
            elif advanced_type == "Treemap" and categorical_cols and numeric_cols:
                treemap_cat = st.selectbox("Categorical column", categorical_cols)
                treemap_num = st.selectbox("Values column", numeric_cols)
                
                grouped = df.groupby(treemap_cat)[treemap_num].sum().reset_index()
                fig = px.treemap(grouped, path=[treemap_cat], values=treemap_num,
                               title="Treemap")
                st.plotly_chart(fig, use_container_width=True)
                
            elif advanced_type == "Radar Chart" and len(numeric_cols) >= 3:
                radar_cols = st.multiselect("Select numeric columns", numeric_cols[:6])
                if len(radar_cols) >= 3:
                    # Create radar chart with mean values
                    means = df[radar_cols].mean()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=means.values,
                        theta=means.index,
                        fill='toself',
                        name='Mean Values'
                    ))
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True)),
                        showlegend=True,
                        title="Radar Chart"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Please upload a dataset first to see visualizations.")
        st.info("💡 The visualization tool will automatically detect your data types and suggest the best plots!")


