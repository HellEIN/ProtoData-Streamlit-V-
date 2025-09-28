import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


# Set page configuration
st.set_page_config(page_title="ProtoData", layout="centered", initial_sidebar_state="expanded",)

st.title("Advanced Data Explorer & ML Model Trainer")

# Initialize session state for storing trained models and UI state
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = None
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = None
if 'target_col' not in st.session_state:
    st.session_state.target_col = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'target_encoder' not in st.session_state:
    st.session_state.target_encoder = None
if 'feature_encoders' not in st.session_state:
    st.session_state.feature_encoders = {}
if 'categorical_features' not in st.session_state:
    st.session_state.categorical_features = []
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'model_trained_successfully' not in st.session_state:
    st.session_state.model_trained_successfully = False

# File uploader
st.info("Please upload your dataset.")
with st.expander("Data Upload"):
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "parquet", "feather"])
    if uploaded_file is not None:
        try:
            # Get file extension as fallback
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            # Determine file type by MIME type first, then fallback to extension
            if uploaded_file.type == "text/csv" or file_extension == "csv":
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" or file_extension == "xlsx":
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.type == "application/x-parquet" or file_extension == "parquet":
                df = pd.read_parquet(uploaded_file)
            elif uploaded_file.type == "application/octet-stream" or file_extension == "feather":
                df = pd.read_feather(uploaded_file)
            else:
                st.error(f"Unsupported file type: {uploaded_file.type}")
                df = None # Ensure df is None on error
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.warning("Please ensure you're uploading a valid file in the supported format.")
            df = None # Ensure df is None on error

        if df is not None:
            st.dataframe(df)
            # Store original dataframe in session state
            st.session_state.df_original = df.copy()
            st.session_state.trained_model = None
            st.session_state.model_trained_successfully = False
            st.session_state.prediction_result = None


# Data summary
with st.expander("Data Summary"):
    if 'df' in locals() and df is not None:
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
with st.expander("Comprehensive Data Visualization"):
    if 'df' in locals() and df is not None:
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
        viz_tabs = st.tabs(["📊 Single Variable", "📈 Two Variables", "🌐 Multi-Variable", "📊 Statistical", "🎨 Advanced"])
        
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
                    fig = px.histogram(df, x=single_col, nbins=30, title=f"Distribution of {single_col}")
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
                with st.expander(f"📈 Statistics for {single_col}"):
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
                    st.metric("**Correlation Coefficient**", f"{correlation:.3f}")
        
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
                                    ["Distribution Comparison", "Regression Plot"])
            
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

# Machine Learning Models
with st.expander("Machine Learning Models", expanded=st.session_state.model_trained_successfully):
    if 'df' in locals() and df is not None:
        st.write("### Machine Learning Model Training")
        
        # Get column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
        all_columns = df.columns.tolist()
        
        # Model selection
        model_type = st.selectbox("Select Model Type", 
                                    ["Linear Regression", "Logistic Regression"])
        
        st.write("---")
        
        if model_type == "Linear Regression":
            st.subheader("📈 Linear Regression")
            st.info("Linear Regression is used for predicting continuous numerical values.")
            
            # Target selection (must be numeric for linear regression)
            if not numeric_cols:
                st.error("No numeric columns found for Linear Regression. Please ensure your dataset has numeric columns.")
            else:
                target_col = st.selectbox("Select Target Variable (what you want to predict)", 
                                        numeric_cols, key="lr_target")
                
                # Feature selection
                available_features = [col for col in all_columns if col != target_col]
                feature_cols = st.multiselect("Select Feature Variables (predictors)", 
                                            available_features, 
                                            default=available_features[:3] if len(available_features) >= 3 else available_features,
                                            key="lr_features")
                
                if target_col and feature_cols:
                    # Data preprocessing
                    st.write("### Data Preprocessing")
                    
                    # Handle missing values
                    df_ml = df[[target_col] + feature_cols].copy()
                    initial_rows = len(df_ml)
                    df_ml = df_ml.dropna()
                    final_rows = len(df_ml)
                    
                    if initial_rows != final_rows:
                        st.warning(f"Removed {initial_rows - final_rows} rows with missing values. Using {final_rows} rows for training.")
                    
                    # Encode categorical variables
                    categorical_features = [col for col in feature_cols if col in categorical_cols]
                    feature_encoders = {} # Local scope for this training instance
                    if categorical_features:
                        st.write("**Encoding categorical variables:**")
                        for col in categorical_features:
                            le = LabelEncoder()
                            df_ml[col] = le.fit_transform(df_ml[col].astype(str))
                            feature_encoders[col] = le
                            st.write(f"- {col}: {len(le.classes_)} unique categories encoded")
                    
                    # Train-test split
                    test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
                    random_state = st.number_input("Random State (for reproducibility)", 0, 100, 42)
                    
                    if st.button("Train Linear Regression Model", type="primary"):
                        try:
                            # Prepare data
                            X = df_ml[feature_cols]
                            y = df_ml[target_col]
                            
                            # Split data
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=test_size, random_state=random_state)
                            
                            # Scale features
                            scaler = StandardScaler()
                            X_train_scaled = scaler.fit_transform(X_train)
                            X_test_scaled = scaler.transform(X_test)
                            
                            # Train model
                            model = LinearRegression()
                            model.fit(X_train_scaled, y_train)
                            
                            # Store model and preprocessing objects in session state
                            st.session_state.trained_model = model
                            st.session_state.model_type = "Linear Regression"
                            st.session_state.feature_cols = feature_cols
                            st.session_state.target_col = target_col
                            st.session_state.scaler = scaler
                            st.session_state.feature_encoders = feature_encoders
                            st.session_state.categorical_features = categorical_features
                            
                            st.session_state.model_trained_successfully = True
                            st.success("Model trained successfully!")
                            
                            # Make predictions
                            y_train_pred = model.predict(X_train_scaled)
                            y_test_pred = model.predict(X_test_scaled)
                            
                            # Calculate metrics
                            train_r2 = r2_score(y_train, y_train_pred)
                            test_r2 = r2_score(y_test, y_test_pred)
                            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                            
                            # Display results
                            st.write("### Model Performance")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Training R² Score", f"{train_r2:.4f}")
                                st.metric("Training RMSE", f"{train_rmse:.4f}")
                            with col2:
                                st.metric("Test R² Score", f"{test_r2:.4f}")
                                st.metric("Test RMSE", f"{test_rmse:.4f}")
                            
                            # Feature importance
                            st.write("### Feature Importance")
                            feature_importance = pd.DataFrame({
                                'Feature': feature_cols,
                                'Coefficient': model.coef_,
                                'Abs_Coefficient': np.abs(model.coef_)
                            }).sort_values('Abs_Coefficient', ascending=False)
                            
                            fig_importance = px.bar(feature_importance, 
                                                    x='Abs_Coefficient', 
                                                    y='Feature', 
                                                    orientation='h',
                                                    title="Feature Importance (Absolute Coefficients)")
                            st.plotly_chart(fig_importance, use_container_width=True)
                            
                            # Predictions vs Actual plot
                            st.write("### Predictions vs Actual Values")
                            
                            fig_pred = go.Figure()
                            
                            # Training data
                            fig_pred.add_trace(go.Scatter(
                                x=y_train, y=y_train_pred,
                                mode='markers',
                                name='Training Data',
                                opacity=0.6
                            ))
                            
                            # Test data
                            fig_pred.add_trace(go.Scatter(
                                x=y_test, y=y_test_pred,
                                mode='markers',
                                name='Test Data',
                                opacity=0.8
                            ))
                            
                            # Perfect prediction line
                            min_val = min(y.min(), min(y_train_pred.min(), y_test_pred.min()))
                            max_val = max(y.max(), max(y_train_pred.max(), y_test_pred.max()))
                            fig_pred.add_trace(go.Scatter(
                                x=[min_val, max_val], y=[min_val, max_val],
                                mode='lines',
                                name='Perfect Prediction',
                                line=dict(dash='dash', color='red')
                            ))
                            
                            fig_pred.update_layout(
                                title='Predictions vs Actual Values',
                                xaxis_title=f'Actual {target_col}',
                                yaxis_title=f'Predicted {target_col}'
                            )
                            st.plotly_chart(fig_pred, use_container_width=True)
                            
                            # Residuals plot
                            st.write("### Residuals Analysis")
                            residuals_test = y_test - y_test_pred
                            
                            fig_residuals = px.scatter(x=y_test_pred, y=residuals_test,
                                                        title="Residuals vs Predicted Values",
                                                        labels={'x': 'Predicted Values', 'y': 'Residuals'})
                            fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
                            st.plotly_chart(fig_residuals, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Error training model: {str(e)}")
        
        elif model_type == "Logistic Regression":
            st.subheader("📊 Logistic Regression")
            st.info("Logistic Regression is used for predicting categorical outcomes (classification).")
            
            # Target selection
            target_col = st.selectbox("Select Target Variable (what you want to predict)", 
                                    all_columns, key="log_target")
            
            if target_col:
                # Check if target is suitable for classification
                unique_values = df[target_col].nunique()
                if unique_values > 10:
                    st.warning(f"Target variable has {unique_values} unique values. Consider if this is appropriate for classification.")
                
                # Feature selection
                available_features = [col for col in all_columns if col != target_col]
                feature_cols = st.multiselect("Select Feature Variables (predictors)", 
                                            available_features, 
                                            default=available_features[:3] if len(available_features) >= 3 else available_features,
                                            key="log_features")
                
                if target_col and feature_cols:
                    # Data preprocessing
                    st.write("### Data Preprocessing")
                    
                    # Handle missing values
                    df_ml = df[[target_col] + feature_cols].copy()
                    initial_rows = len(df_ml)
                    df_ml = df_ml.dropna()
                    final_rows = len(df_ml)
                    
                    if initial_rows != final_rows:
                        st.warning(f"Removed {initial_rows - final_rows} rows with missing values. Using {final_rows} rows for training.")
                    
                    # Encode target variable
                    target_encoder = LabelEncoder()
                    df_ml[target_col] = target_encoder.fit_transform(df_ml[target_col].astype(str))
                    st.write(f"**Target variable encoding:** {len(target_encoder.classes_)} classes")
                    st.write(f"Classes: {', '.join(target_encoder.classes_)}")
                    
                    # Encode categorical features
                    categorical_features = [col for col in feature_cols if col in categorical_cols]
                    feature_encoders = {}
                    if categorical_features:
                        st.write("**Encoding categorical features:**")
                        for col in categorical_features:
                            le = LabelEncoder()
                            df_ml[col] = le.fit_transform(df_ml[col].astype(str))
                            feature_encoders[col] = le
                            st.write(f"- {col}: {len(le.classes_)} unique categories encoded")
                    
                    # Train-test split
                    test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05, key="log_test_size")
                    random_state = st.number_input("Random State (for reproducibility)", 0, 100, 42, key="log_random_state")
                    
                    if st.button("Train Logistic Regression Model", type="primary"):
                        try:
                            # Prepare data
                            X = df_ml[feature_cols]
                            y = df_ml[target_col]
                            
                            # Split data
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=test_size, random_state=random_state, stratify=y)
                            
                            # Scale features
                            scaler = StandardScaler()
                            X_train_scaled = scaler.fit_transform(X_train)
                            X_test_scaled = scaler.transform(X_test)
                            
                            # Train model
                            model = LogisticRegression(random_state=random_state, max_iter=1000)
                            model.fit(X_train_scaled, y_train)
                            
                            # Store model and preprocessing objects in session state
                            st.session_state.trained_model = model
                            st.session_state.model_type = "Logistic Regression"
                            st.session_state.feature_cols = feature_cols
                            st.session_state.target_col = target_col
                            st.session_state.scaler = scaler
                            st.session_state.target_encoder = target_encoder
                            st.session_state.feature_encoders = feature_encoders
                            st.session_state.categorical_features = categorical_features
                            
                            st.session_state.model_trained_successfully = True
                            st.success("Model trained successfully!")

                            # Make predictions
                            y_train_pred = model.predict(X_train_scaled)
                            y_test_pred = model.predict(X_test_scaled)
                            y_test_pred_proba = model.predict_proba(X_test_scaled)
                            
                            # Calculate metrics
                            train_accuracy = accuracy_score(y_train, y_train_pred)
                            test_accuracy = accuracy_score(y_test, y_test_pred)
                            
                            # Display results
                            st.write("### Model Performance")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Training Accuracy", f"{train_accuracy:.4f}")
                            with col2:
                                st.metric("Test Accuracy", f"{test_accuracy:.4f}")
                            
                            # Classification Report
                            st.write("### Classification Report")
                            class_report = classification_report(y_test, y_test_pred, 
                                                                target_names=target_encoder.classes_, 
                                                                output_dict=True)
                            
                            # Convert to DataFrame for better display
                            report_df = pd.DataFrame(class_report).iloc[:-1, :].T  # Exclude 'accuracy' row
                            st.dataframe(report_df.round(4))
                            
                            # Confusion Matrix
                            st.write("### Confusion Matrix")
                            cm = confusion_matrix(y_test, y_test_pred)
                            
                            fig_cm = px.imshow(cm, 
                                                text_auto=True, 
                                                aspect="auto",
                                                color_continuous_scale='Blues',
                                                title="Confusion Matrix",
                                                labels=dict(x="Predicted Label", y="True Label"),
                                                x=target_encoder.classes_,
                                                y=target_encoder.classes_)
                            st.plotly_chart(fig_cm, use_container_width=True)
                            
                            # Feature importance (coefficients)
                            st.write("### Feature Importance")
                            
                            if len(target_encoder.classes_) <= 2:
                                # Binary classification - single set of coefficients
                                feature_importance = pd.DataFrame({
                                    'Feature': feature_cols,
                                    'Coefficient': model.coef_[0],
                                    'Abs_Coefficient': np.abs(model.coef_[0])
                                }).sort_values('Abs_Coefficient', ascending=False)
                                
                                fig_importance = px.bar(feature_importance, 
                                                        x='Coefficient', 
                                                        y='Feature', 
                                                        orientation='h',
                                                        title="Feature Importance (Coefficients)",
                                                        color='Coefficient',
                                                        color_continuous_scale='RdBu')
                                st.plotly_chart(fig_importance, use_container_width=True)
                                
                            else:
                                # Multi-class classification
                                st.write("**Multi-class coefficients:**")
                                for i, class_name in enumerate(target_encoder.classes_):
                                    feature_importance = pd.DataFrame({
                                        'Feature': feature_cols,
                                        'Coefficient': model.coef_[i],
                                        'Abs_Coefficient': np.abs(model.coef_[i])
                                    }).sort_values('Abs_Coefficient', ascending=False)
                                    
                                    with st.expander(f"Coefficients for class: {class_name}"):
                                        fig_importance = px.bar(feature_importance, 
                                                                x='Coefficient', 
                                                                y='Feature', 
                                                                orientation='h',
                                                                title=f"Feature Importance for {class_name}",
                                                                color='Coefficient',
                                                                color_continuous_scale='RdBu')
                                        st.plotly_chart(fig_importance, use_container_width=True)
                            
                            # Prediction Probabilities (for binary classification)
                            if len(target_encoder.classes_) == 2:
                                st.write("### Prediction Probability Distribution")
                                
                                prob_df = pd.DataFrame({
                                    'Actual': [target_encoder.classes_[i] for i in y_test],
                                    'Predicted_Probability': y_test_pred_proba[:, 1]  # Probability of positive class
                                })
                                
                                fig_prob = px.histogram(prob_df, 
                                                        x='Predicted_Probability', 
                                                        color='Actual',
                                                        title="Distribution of Prediction Probabilities",
                                                        nbins=20,
                                                        opacity=0.7)
                                st.plotly_chart(fig_prob, use_container_width=True)
                            
                            # Model coefficients table
                            st.write("### Model Coefficients")
                            if len(target_encoder.classes_) <= 2:
                                coef_df = pd.DataFrame({
                                    'Feature': feature_cols + ['Intercept'],
                                    'Coefficient': np.append(model.coef_[0], model.intercept_[0])
                                })
                            else:
                                coef_data = {}
                                for i, class_name in enumerate(target_encoder.classes_):
                                    coef_data[f'{class_name}_coef'] = np.append(model.coef_[i], model.intercept_[i])
                                coef_data['Feature'] = feature_cols + ['Intercept']
                                coef_df = pd.DataFrame(coef_data)
                            
                            st.dataframe(coef_df.round(4))
                            
                        except Exception as e:
                            st.error(f"Error training model: {str(e)}")
                            st.write("**Possible issues:**")
                            st.write("- Insufficient data for the number of classes")
                            st.write("- All samples may belong to one class in the training split")
                            st.write("- Features may need more preprocessing")
        
        # Quick Model Insights Section
        st.write("---")
        st.write("### 🔍 Quick Model Insights")
        
        insight_tabs = st.tabs(["📊 Data Quality", "🎯 Model Recommendations", "📈 Performance Tips"])
        
        with insight_tabs[0]:
            st.write("**Data Quality Assessment:**")
            
            # Missing values
            missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
            high_missing = missing_pct[missing_pct > 20]
            
            if not high_missing.empty:
                st.warning(f"⚠️ Columns with >20% missing values: {', '.join(high_missing.index)}")
            else:
                st.success("✅ No columns with excessive missing values")
            
            # Data types
            st.write(f"📊 **Dataset composition:** {len(numeric_cols)} numeric, {len(categorical_cols)} categorical columns")
            
            # Dataset size
            if len(df) < 100:
                st.warning("⚠️ Small dataset (<100 rows) - results may not be reliable")
            elif len(df) < 1000:
                st.info("💡 Medium dataset - consider cross-validation for better estimates")
            else:
                st.success("✅ Good dataset size for machine learning")
        
        with insight_tabs[1]:
            st.write("**Model Selection Guide:**")
            
            if len(numeric_cols) >= 1:
                st.write("**For predicting numeric values:**")
                st.write("• 📈 **Linear Regression** - Best for linear relationships")
                st.write("• 🌳 **Random Forest** - Good for non-linear relationships (not implemented yet)")
            
            if len(categorical_cols) >= 1 or df[target_col].nunique() < 20:
                st.write("**For predicting categories:**")
                st.write("• 📊 **Logistic Regression** - Good baseline, interpretable")
                st.write("• 🌳 **Random Forest** - Better for complex patterns (not implemented yet)")
            
            st.info("💡 **Tip:** Start with simpler models (Linear/Logistic Regression) to establish baselines!")
        
        with insight_tabs[2]:
            st.write("**Performance Improvement Tips:**")
            
            st.write("**📊 Data Preprocessing:**")
            st.write("• Handle missing values appropriately")
            st.write("• Scale/normalize numeric features")
            st.write("• Encode categorical variables properly")
            st.write("• Remove or transform outliers")
            
            st.write("**🎯 Feature Engineering:**")
            st.write("• Create interaction terms between features")
            st.write("• Transform skewed distributions")
            st.write("• Select relevant features")
            
            st.write("**🔍 Model Validation:**")
            st.write("• Use cross-validation for robust estimates")
            st.write("• Check for overfitting (training vs test performance)")
            st.write("• Validate assumptions (linearity, normality, etc.)")
        
    else:
        st.warning("⚠️ Please upload a dataset first to use machine learning models.")
        st.info("💡 **Machine Learning Features:**")
        st.write("• **Linear Regression** - Predict continuous values")
        st.write("• **Logistic Regression** - Classify categories")  
        st.write("• **Automatic preprocessing** - Handle missing values and encoding")
        st.write("• **Performance metrics** - R², RMSE, Accuracy, Confusion Matrix")
        st.write("• **Feature importance** - Understand which features matter most")
        st.write("• **Visualizations** - Residual plots, prediction vs actual, probability distributions")

# Sidebar for predictions
if st.session_state.trained_model is not None and st.session_state.df_original is not None:
    st.sidebar.title("Make Predictions")
    st.sidebar.write(f"**Model:** {st.session_state.model_type}")
    st.sidebar.write(f"**Target:** {st.session_state.target_col}")
    st.sidebar.write("---")
    
    # Create input widgets for each feature
    input_data = {}
    df_original = st.session_state.df_original
    
    st.sidebar.write("### Enter Feature Values:")
    
    for feature in st.session_state.feature_cols:
        if feature in st.session_state.categorical_features:
            # Categorical feature - use selectbox
            unique_values = df_original[feature].dropna().unique().tolist()
            input_data[feature] = st.sidebar.selectbox(
                f"Select {feature}:",
                options=unique_values,
                key=f"input_{feature}"
            )
        else:
            # Numeric feature - use number input
            min_val = float(df_original[feature].min()) if not pd.isna(df_original[feature].min()) else 0.0
            max_val = float(df_original[feature].max()) if not pd.isna(df_original[feature].max()) else 100.0
            mean_val = float(df_original[feature].mean()) if not pd.isna(df_original[feature].mean()) else (min_val + max_val) / 2
            
            input_data[feature] = st.sidebar.number_input(
                f"Enter {feature}:",
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=(max_val - min_val) / 100 if (max_val - min_val) > 0 else 1.0,
                key=f"input_{feature}"
            )
    
    st.sidebar.write("---")
    
    # Make prediction button
    if st.sidebar.button("🔮 Make Prediction", type="primary"):
        try:
            # Prepare input data
            input_df = pd.DataFrame([input_data])
            
            # Encode categorical features using stored encoders
            for col in st.session_state.categorical_features:
                if col in input_df.columns and col in st.session_state.feature_encoders:
                    encoder = st.session_state.feature_encoders[col]
                    # Handle unseen categories by mapping to a known category (e.g., the first one)
                    current_value = input_data[col]
                    known_classes = list(encoder.classes_)
                    if current_value not in known_classes:
                        st.sidebar.warning(f"Value '{current_value}' for '{col}' was not seen during training. Using '{known_classes[0]}' as a fallback.")
                        input_df[col] = known_classes[0] # Fallback to the first known class
                    
                    input_df[col] = encoder.transform(input_df[col])[0]
            
            # Scale features
            input_scaled = st.session_state.scaler.transform(input_df[st.session_state.feature_cols])
            
            # Make prediction
            prediction = st.session_state.trained_model.predict(input_scaled)[0]
            
            result_data = {}
            if st.session_state.model_type == "Linear Regression":
                result_data = {
                    'type': 'regression',
                    'prediction': prediction,
                    'target': st.session_state.target_col
                }
            elif st.session_state.model_type == "Logistic Regression":
                prediction_proba = st.session_state.trained_model.predict_proba(input_scaled)[0]
                predicted_class = st.session_state.target_encoder.inverse_transform([int(prediction)])[0]
                max_probability = max(prediction_proba)
                result_data = {
                    'type': 'classification',
                    'prediction': predicted_class,
                    'probability': max_probability,
                    'probabilities': prediction_proba,
                    'classes': st.session_state.target_encoder.classes_,
                    'target': st.session_state.target_col
                }
            
            st.session_state.prediction_result = result_data
            
        except Exception as e:
            st.session_state.prediction_result = {
                'type': 'error',
                'message': f"Prediction Error: {e}"
            }
    
    # Display prediction results from session state
    if st.session_state.prediction_result:
        result = st.session_state.prediction_result
        
        if result['type'] == 'regression':
            st.sidebar.success(f"**Predicted {result['target']}:** {result['prediction']:.4f}")
            
        elif result['type'] == 'classification':
            st.sidebar.success(f"**Predicted {result['target']}:** {result['prediction']}")
            st.sidebar.metric("Confidence", f"{result['probability']:.1%}")
            
            with st.sidebar.expander("Show all class probabilities"):
                for i, class_name in enumerate(result['classes']):
                    prob = result['probabilities'][i]
                    st.write(f"**{class_name}**")
                    st.progress(prob, text=f"{prob:.1%}")
                
        elif result['type'] == 'error':
            st.sidebar.error(result['message'])
            st.sidebar.write("Please check your input values and model compatibility.")

else:
    st.sidebar.title("🎯 Predictions")
    st.sidebar.info("Train a model first to make predictions!")
    st.sidebar.write("**Steps:**")
    st.sidebar.write("1. Upload your dataset")
    st.sidebar.write("2. Go to the 'Machine Learning Models' section") 
    st.sidebar.write("3. Train a Linear or Logistic Regression model")
    st.sidebar.write("4. The prediction panel will appear here!")
