from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import numpy as np
from werkzeug.utils import secure_filename
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def convert_to_json_serializable(obj):
    """Convert pandas/numpy objects to JSON serializable types"""
    if pd.isna(obj):
        return None
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'isoformat'):  # datetime objects
        return obj.isoformat()
    elif hasattr(obj, 'strftime'):  # timestamp objects
        return str(obj)
    else:
        return obj

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_dataframe(df):
    """Generate comprehensive analysis of the dataframe with first column as x-axis"""
    # Clean and prepare the dataframe
    df_clean = df.copy()
    
    # Remove completely empty rows and columns
    df_clean = df_clean.dropna(how='all').dropna(axis=1, how='all')
    
    # Clean column names (remove spaces, special characters)
    df_clean.columns = df_clean.columns.str.strip()
    
    analysis = {
        'basic_info': {
            'rows': len(df_clean),
            'columns': len(df_clean.columns),
            'column_names': df_clean.columns.tolist(),
            'dtypes': df_clean.dtypes.astype(str).to_dict()
        },
        'summary_stats': {},
        'charts': []
    }
    
    # Get column information
    first_col = df_clean.columns[0]
    all_cols = df_clean.columns.tolist()
    
    # Try to convert columns to appropriate numeric types
    for col in df_clean.columns:
        if col != first_col:  # Don't convert first column yet
            # Try to convert to numeric, coerce errors to NaN
            numeric_series = pd.to_numeric(df_clean[col], errors='coerce')
            if not numeric_series.isna().all():  # If at least some values converted successfully
                df_clean[col] = numeric_series
    
    # Now get numeric columns after conversion
    numeric_cols = df_clean.select_dtypes(include=['number']).columns
    
    # Try to detect if first column contains dates
    is_date_column = False
    try:
        # Try different date parsing approaches
        test_dates = pd.to_datetime(df_clean[first_col].head(5), errors='coerce')
        if not test_dates.isna().all():
            is_date_column = True
    except:
        pass
    
    # If first column is not dates, try to convert to numeric
    if not is_date_column:
        numeric_first = pd.to_numeric(df_clean[first_col], errors='coerce')
        if not numeric_first.isna().all():
            df_clean[first_col] = numeric_first
            # Update numeric columns list
            numeric_cols = df_clean.select_dtypes(include=['number']).columns
    
    # Generate summary statistics for numeric columns
    if len(numeric_cols) > 0:
        stats_dict = df_clean[numeric_cols].describe().to_dict()
        # Convert all numpy values to JSON serializable types
        for col, stats in stats_dict.items():
            for stat_name, value in stats.items():
                stats_dict[col][stat_name] = convert_to_json_serializable(value)
        analysis['summary_stats'] = stats_dict
    
    # Generate charts - START WITH DATA PLOTTING
    charts = []
    
    # 1. Data overview table (first 10 rows)
    table_data = []
    for col in df_clean.columns:
        col_data = [convert_to_json_serializable(val) for val in df_clean[col].head(10)]
        table_data.append(col_data)
    
    table_chart = {
        'data': [{
            'type': 'table',
            'header': {'values': df_clean.columns.tolist()},
            'cells': {'values': table_data}
        }],
        'layout': {
            'title': {'text': 'Data Preview (First 10 Rows)'}
        }
    }
    
    charts.append({
        'type': 'table',
        'title': 'Data Preview',
        'chart': table_chart
    })
    
    # 2. MAIN DATA PLOTS - Using first column as x-axis
    if len(all_cols) > 1:
        # Prepare plotting dataframe
        plot_df = df_clean.copy()
        
        # Convert first column to datetime if it looks like dates
        if is_date_column:
            try:
                plot_df[first_col] = pd.to_datetime(plot_df[first_col], errors='coerce')
                # Remove rows where date conversion failed
                plot_df = plot_df.dropna(subset=[first_col])
            except:
                pass
        
        # Plot each numeric column against the first column
        for col in numeric_cols:
            if col != first_col:  # Don't plot first column against itself
                # Remove rows with NaN values for this specific plot
                plot_data = plot_df[[first_col, col]].dropna()
                
                if len(plot_data) > 0:  # Only plot if we have data
                    # Manually create chart data to avoid binary encoding issues
                    x_values = plot_data[first_col].tolist()
                    y_values = plot_data[col].tolist()
                    
                    # Convert all values to JSON serializable types
                    x_values = [convert_to_json_serializable(x) for x in x_values]
                    y_values = [convert_to_json_serializable(y) for y in y_values]
                    
                    chart_data = {
                        'data': [{
                            'x': x_values,
                            'y': y_values,
                            'type': 'scatter',
                            'mode': 'lines+markers',
                            'name': col,
                            'line': {'width': 2}
                        }],
                        'layout': {
                            'title': {'text': f'{col} vs {first_col}'},
                            'xaxis': {'title': {'text': first_col}},
                            'yaxis': {'title': {'text': col}},
                            'hovermode': 'x unified',
                            'showlegend': False
                        }
                    }
                    
                    charts.append({
                        'type': 'time_series' if is_date_column else 'line_plot',
                        'title': f'{col} vs {first_col}',
                        'chart': chart_data
                    })
        
        # If we have multiple numeric columns (excluding first if it's numeric), create multi-line plot
        numeric_cols_to_plot = [col for col in numeric_cols if col != first_col]
        if len(numeric_cols_to_plot) > 1:
            # Create clean data for multi-line plot
            multi_plot_data = plot_df[[first_col] + numeric_cols_to_plot].dropna()
            
            if len(multi_plot_data) > 0:
                # Manual multi-line chart creation
                x_values = [convert_to_json_serializable(x) for x in multi_plot_data[first_col].tolist()]
                
                chart_traces = []
                for col in numeric_cols_to_plot[:6]:  # Limit to 6 lines for readability
                    y_values = [convert_to_json_serializable(y) for y in multi_plot_data[col].tolist()]
                    chart_traces.append({
                        'x': x_values,
                        'y': y_values,
                        'type': 'scatter',
                        'mode': 'lines+markers',
                        'name': col,
                        'line': {'width': 2}
                    })
                
                multi_chart = {
                    'data': chart_traces,
                    'layout': {
                        'title': {'text': f'All Numeric Variables vs {first_col}'},
                        'xaxis': {'title': {'text': first_col}},
                        'yaxis': {'title': {'text': 'Values'}},
                        'hovermode': 'x unified',
                        'legend': {
                            'orientation': 'h',
                            'yanchor': 'bottom',
                            'y': 1.02,
                            'xanchor': 'right',
                            'x': 1
                        }
                    }
                }
                
                charts.append({
                    'type': 'multi_line',
                    'title': f'All Variables vs {first_col}',
                    'chart': multi_chart
                })
    
    # 3. CORRELATION ANALYSIS - After plotting the data
    if len(numeric_cols) > 1:
        corr_matrix = df_clean[numeric_cols].corr()
        
        # Create correlation heatmap manually
        col_names = corr_matrix.columns.tolist()
        corr_values = []
        
        # Convert correlation matrix to format suitable for heatmap
        for i, row_name in enumerate(col_names):
            for j, col_name in enumerate(col_names):
                corr_val = convert_to_json_serializable(corr_matrix.iloc[i, j])
                corr_values.append([i, j, corr_val])
        
        # Create correlation heatmap chart
        heatmap_chart = {
            'data': [{
                'type': 'heatmap',
                'z': [[convert_to_json_serializable(corr_matrix.iloc[i, j]) for j in range(len(col_names))] for i in range(len(col_names))],
                'x': col_names,
                'y': col_names,
                'colorscale': 'RdBu',
                'zmid': 0,
                'zmin': -1,
                'zmax': 1,
                'showscale': True,
                'hovertemplate': 'Correlation: %{z:.3f}<br>%{y} vs %{x}<extra></extra>'
            }],
            'layout': {
                'title': {'text': 'Correlation Matrix - Strength of Relationships Between Variables'},
                'xaxis': {'title': {'text': 'Variables'}},
                'yaxis': {'title': {'text': 'Variables'}},
                'width': 600,
                'height': 500
            }
        }
        
        charts.append({
            'type': 'correlation',
            'title': 'Correlation Matrix',
            'chart': heatmap_chart
        })
        
        # Create correlation bar chart (absolute values with first numeric column)
        if len(numeric_cols) > 2:
            # Use first numeric column for correlation analysis
            first_numeric_col = numeric_cols[0]
            correlations_with_first = corr_matrix[first_numeric_col].drop(first_numeric_col).abs().sort_values(ascending=True)
            
            bar_chart = {
                'data': [{
                    'type': 'bar',
                    'x': [convert_to_json_serializable(val) for val in correlations_with_first.values],
                    'y': correlations_with_first.index.tolist(),
                    'orientation': 'h',
                    'marker': {'color': 'lightblue'},
                    'hovertemplate': 'Correlation: %{x:.3f}<br>Variable: %{y}<extra></extra>'
                }],
                'layout': {
                    'title': {'text': f'Correlation Strength with {first_numeric_col}'},
                    'xaxis': {'title': {'text': 'Absolute Correlation'}},
                    'yaxis': {'title': {'text': 'Variables'}},
                    'height': 400
                }
            }
            
            charts.append({
                'type': 'correlation_bar',
                'title': f'Variable Correlations with {first_numeric_col}',
                'chart': bar_chart
            })
    
    # 4. PREDICTIVE MODELING - 2nd column as dependent, rest as independent
    if len(numeric_cols) > 2:
        try:
            # Prepare data for modeling
            # Assuming 2nd column is dependent variable, rest are independent
            dependent_col = df_clean.columns[1] if len(df_clean.columns) > 1 else None
            
            if dependent_col and dependent_col in numeric_cols:
                # Get independent variables (numeric columns excluding the dependent variable)
                independent_cols = [col for col in numeric_cols if col != dependent_col]
                
                if len(independent_cols) > 0:
                    # Prepare modeling data
                    model_data = df_clean[[dependent_col] + independent_cols].dropna()
                    
                    if len(model_data) > 10:  # Need sufficient data for modeling
                        X = model_data[independent_cols]
                        y = model_data[dependent_col]
                        
                        # Split data for training and testing
                        if len(model_data) > 30:
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        else:
                            # For small datasets, use all data for training
                            X_train, X_test, y_train, y_test = X, X, y, y
                        
                        # Try both Linear Regression and Random Forest
                        models = {
                            'Linear Regression': LinearRegression(),
                            'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42)
                        }
                        
                        best_model = None
                        best_score = -np.inf
                        best_model_name = ''
                        best_predictions = None
                        
                        model_results = {}
                        
                        for model_name, model in models.items():
                            # Train model
                            model.fit(X_train, y_train)
                            
                            # Make predictions
                            y_pred = model.predict(X_test)
                            
                            # Calculate metrics
                            r2 = r2_score(y_test, y_pred)
                            mae = mean_absolute_error(y_test, y_pred)
                            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                            
                            model_results[model_name] = {
                                'r2_score': convert_to_json_serializable(r2),
                                'mae': convert_to_json_serializable(mae),
                                'rmse': convert_to_json_serializable(rmse)
                            }
                            
                            # Keep track of best model
                            # Prefer Linear Regression if performance is very close (within 1%) for interpretability
                            if (r2 > best_score) or (model_name == 'Linear Regression' and abs(r2 - best_score) < 0.01):
                                best_score = r2
                                best_model = model
                                best_model_name = model_name
                                best_predictions = y_pred
                        
                        # Generate predictions for the entire dataset
                        full_predictions = best_model.predict(X)
                        
                        # Create actual vs predicted plot
                        # Use only the data that was used for modeling to ensure alignment
                        model_data_with_x = df_clean.loc[model_data.index, [first_col, dependent_col]]
                        
                        x_values = [convert_to_json_serializable(x) for x in model_data_with_x[first_col]]
                        actual_values = [convert_to_json_serializable(y) for y in model_data[dependent_col]]
                        predicted_values = [convert_to_json_serializable(p) for p in full_predictions]
                        
                        prediction_chart = {
                            'data': [
                                {
                                    'x': x_values,
                                    'y': actual_values,
                                    'type': 'scatter',
                                    'mode': 'lines+markers',
                                    'name': f'Actual {dependent_col}',
                                    'line': {'color': 'blue', 'width': 2},
                                    'marker': {'size': 4}
                                },
                                {
                                    'x': x_values,
                                    'y': predicted_values,
                                    'type': 'scatter',
                                    'mode': 'lines+markers',
                                    'name': f'Predicted {dependent_col} ({best_model_name})',
                                    'line': {'color': 'red', 'width': 2, 'dash': 'dash'},
                                    'marker': {'size': 4}
                                }
                            ],
                            'layout': {
                                'title': {'text': f'Actual vs Predicted {dependent_col}'},
                                'xaxis': {'title': {'text': first_col}},
                                'yaxis': {'title': {'text': dependent_col}},
                                'hovermode': 'x unified',
                                'legend': {
                                    'orientation': 'h',
                                    'yanchor': 'bottom',
                                    'y': 1.02,
                                    'xanchor': 'right',
                                    'x': 1
                                }
                            }
                        }
                        
                        charts.append({
                            'type': 'prediction',
                            'title': f'Predictive Model: Actual vs Predicted {dependent_col}',
                            'chart': prediction_chart
                        })
                        
                        # Extract model formula (works best with Linear Regression)
                        model_formula = None
                        formula_explanation = None
                        
                        if best_model_name == 'Linear Regression':
                            # Get coefficients and intercept
                            intercept = best_model.intercept_
                            coefficients = best_model.coef_
                            
                            # Build formula string
                            formula_parts = [f"{convert_to_json_serializable(intercept):.4f}"]
                            
                            for i, (coef, var) in enumerate(zip(coefficients, independent_cols)):
                                coef_val = convert_to_json_serializable(coef)
                                if coef_val >= 0:
                                    formula_parts.append(f" + {coef_val:.4f} × {var}")
                                else:
                                    formula_parts.append(f" - {abs(coef_val):.4f} × {var}")
                            
                            model_formula = f"{dependent_col} = " + "".join(formula_parts)
                            
                            # Create explanation
                            formula_explanation = f"Linear regression formula where {dependent_col} is predicted using:"
                            for coef, var in zip(coefficients, independent_cols):
                                coef_val = convert_to_json_serializable(coef)
                                impact = "increases" if coef_val > 0 else "decreases"
                                formula_explanation += f"\n• {var}: coefficient {coef_val:.4f} (each unit {impact} {dependent_col} by {abs(coef_val):.4f})"
                                
                        elif best_model_name == 'Random Forest':
                            # For Random Forest, show feature importance
                            feature_importance = best_model.feature_importances_
                            
                            formula_explanation = f"Random Forest model (non-linear) - Feature Importance for predicting {dependent_col}:"
                            importance_pairs = list(zip(independent_cols, feature_importance))
                            importance_pairs.sort(key=lambda x: x[1], reverse=True)
                            
                            for var, importance in importance_pairs:
                                importance_val = convert_to_json_serializable(importance)
                                formula_explanation += f"\n• {var}: {importance_val:.1%} importance"
                            
                            model_formula = f"{dependent_col} = RandomForest({', '.join(independent_cols)})"
                        
                        # Add model performance information to analysis
                        analysis['model_results'] = {
                            'dependent_variable': dependent_col,
                            'independent_variables': independent_cols,
                            'best_model': best_model_name,
                            'best_r2_score': convert_to_json_serializable(best_score),
                            'model_comparison': model_results,
                            'data_points': len(model_data),
                            'model_formula': model_formula,
                            'formula_explanation': formula_explanation
                        }
                        
        except Exception as e:
            # If modeling fails, add error info but don't break the analysis
            analysis['model_error'] = str(e)
    
    analysis['charts'] = charts
    analysis['first_column_info'] = {
        'name': first_col,
        'is_date': is_date_column,
        'type': str(df_clean[first_col].dtype)
    }
    return analysis

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read Excel file
            if filename.endswith('.xlsx'):
                df = pd.read_excel(filepath, engine='openpyxl')
            else:
                df = pd.read_excel(filepath)
            
            # Analyze the data
            analysis = analyze_dataframe(df)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'filename': filename,
                'analysis': analysis
            })
            
        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type. Please upload .xlsx or .xls files only.'}), 400

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)