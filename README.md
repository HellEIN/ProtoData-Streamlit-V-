# 📊 Data Visualization Dashboard

A Streamlit app that automatically creates comprehensive visualizations for any dataset with smart plot selection and statistical analysis.

## 🚀 Quick Start

### Install Dependencies
```bash
pip install streamlit pandas numpy plotly
```

### Run the Application
1. Save the code as `app.py`
2. Run Streamlit:
```bash
streamlit run app.py
```
3. Open your browser to `http://localhost:8501`

## 📁 How to Use

1. **Upload Data**: Load your CSV/Excel file or prepare a pandas DataFrame named `df`
2. **Open Visualizations**: Expand the "📊 Comprehensive Data Visualization" section
3. **Explore**: Navigate through 5 tabs:
   - 🔍 **Single Variable**: Histograms, box plots, bar charts
   - 📈 **Two Variables**: Scatter plots, correlations, grouped analysis
   - 🌐 **Multi-Variable**: Correlation matrices, 3D plots, pair plots
   - 📊 **Statistical**: Regression analysis, distribution comparisons
   - 🎨 **Advanced**: Sunburst, treemap, radar charts

## 🎯 Key Features

- **Auto-Detection**: Automatically identifies data types (numeric, categorical, datetime)
- **Smart Plotting**: Chooses the best visualization based on your data
- **15+ Plot Types**: From basic charts to advanced multi-dimensional plots
- **Interactive**: Plotly-based interactive visualizations
- **Statistics**: Automatic correlation analysis and summary statistics

## 📊 Supported Data Types

- Numeric columns → Histograms, box plots, scatter plots
- Categorical columns → Bar charts, pie charts, count plots
- Mixed data → Grouped analysis, cross-tabulations
- Multi-column → Correlation matrices, parallel coordinates

## 🛠️ File Structure

```
your-project/
├── app.py              # Main Streamlit application
└── data/              # Your datasets (CSV, Excel...etc)
```

## 💡 Pro Tips

- Use **"Auto"** mode for instant best-fit visualizations
- Select multiple columns for advanced multi-variable analysis
- Check the summary report for dataset overview
- Works best with datasets under 100k rows for performance

## 🐛 Troubleshooting

**"DataFrame not found"**: Ensure your data is loaded as `df` before expanding visualizations

**Performance issues**: For large datasets, sample your data: `df = df.sample(10000)`

**Plot errors**: Check that selected columns have compatible data types

---

**Ready to visualize? Just run `streamlit run app.py` and upload your data! 🚀**
