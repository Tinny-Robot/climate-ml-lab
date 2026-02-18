import json
from flask import Flask, render_template, request, redirect, url_for, Response
import pandas as pd
import datetime as dt
import io

app = Flask(__name__)

forecast_df = None
try:
    forecast_df = pd.read_csv('samp/daily_data_model_41.csv')
    forecast_df['date'] = pd.to_datetime(forecast_df['date'])
    forecast_df.set_index('date', inplace=True)
except FileNotFoundError:
    print("ERROR: Forecast data file not found.")

@app.route('/')
def home():
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    # --- Data for Dashboard ---
    processed_df = None
    summary_stats = None
    historical_chart_data = None
    data_quality_metrics = None
    correlation_data = None
    monthly_avg_data = None
    monthly_precipitation_data = None

    try:
        processed_df = pd.read_csv('data/processed/clean_weather.csv', index_col='date', parse_dates=True)
        
        # 1. Processed Data Summary
        summary_stats = processed_df.describe().loc[['mean', 'std', 'min', 'max']].to_dict()
        
        # Prepare data for historical trends chart (last 30 days)
        historical_data = processed_df.last('30D')
        # force change to date to current dates for charting purposes
        current_date = dt.datetime.now().date()
        historical_data.index = pd.date_range(start=current_date - pd.Timedelta(days=len(historical_data)-1), periods=len(historical_data), freq='D')

        historical_chart_data = {
            'dates': historical_data.index.strftime('%Y-%m-%d').tolist(),
            'temperatures': historical_data['temperature_c'].tolist(),
            'humidity': historical_data['humidity_pct'].tolist(),
            'wind_speed': historical_data['wind_speed_mps'].tolist()
        }

        # 2. Data Quality Metrics (Missing Values)
        missing_values = processed_df.isnull().sum()
        total_rows = len(processed_df)
        data_quality_metrics = {}
        for col, count in missing_values.items():
            data_quality_metrics[col] = (count / total_rows) * 100 # Percentage

        # 3. Correlation Matrix
        numeric_cols = ['temperature_c', 'humidity_pct', 'wind_speed_mps', 'precipitation_mm', 'surface_pressure_hpa']
        corr_matrix = processed_df[numeric_cols].corr()
        correlation_data = corr_matrix.to_dict('index') # { 'col_A': {'col_A': 1.0, 'col_B': 0.5}, ... }

        # 4. Monthly Averages
        monthly_avg_df = processed_df.groupby(processed_df.index.month).mean(numeric_only=True)
        monthly_avg_data = {
            'months': list(range(1, 13)),
            'temperature_c': monthly_avg_df['temperature_c'].tolist(),
            'humidity_pct': monthly_avg_df['humidity_pct'].tolist(),
            'wind_speed_mps': monthly_avg_df['wind_speed_mps'].tolist(),
            'precipitation_mm': monthly_avg_df['precipitation_mm'].tolist(),
            'surface_pressure_hpa': monthly_avg_df['surface_pressure_hpa'].tolist(),
        }

        # 5. Monthly Precipitation Data (for Box Plot)
        monthly_precipitation_data_raw = []
        for month in range(1, 13):
            month_data = processed_df[processed_df.index.month == month]['precipitation_mm'].dropna().tolist()
            monthly_precipitation_data_raw.append(month_data)
        monthly_precipitation_data = monthly_precipitation_data_raw
        
        # 6. Precipitation Distribution (for Pie Chart)
        rainy_days = processed_df[processed_df['precipitation_mm'] > 0].shape[0]
        dry_days = processed_df[processed_df['precipitation_mm'] == 0].shape[0]
        total_days = processed_df.shape[0]

        # Calculate percentages
        rainy_percent = (rainy_days / total_days) * 100 if total_days > 0 else 0
        dry_percent = (dry_days / total_days) * 100 if total_days > 0 else 0
        
        precipitation_distribution = {
            'labels': ['Rainy Days', 'Dry Days'],
            'data': [round(rainy_percent, 2), round(dry_percent, 2)]
        }

    except FileNotFoundError:
        print("ERROR: Processed data file not found at data/processed/clean_weather.csv.")
        
    # 7. Model Evaluation Metrics (existing)
    model_details = {}
    model_comparison_data = None
    try:
        with open('reports/evaluation_metrics.json') as f:
            metrics = json.load(f)
        
        model_names = []
        mae_values = []
        rmse_values = []
        r2_values = []

        for model_name, model_metric_data in metrics.items():
            model_names.append(model_name)
            mae_values.append(model_metric_data.get('MAE'))
            rmse_values.append(model_metric_data.get('RMSE'))
            r2_values.append(model_metric_data.get('R2'))
        
        model_comparison_data = {
            'models': model_names,
            'mae': mae_values,
            'rmse': rmse_values,
            'r2': r2_values
        }
        model_details = metrics

    except FileNotFoundError:
        print("ERROR: Model evaluation metrics file not found at reports/evaluation_metrics.json.")

    return render_template(
        'dashboard.html', 
        model_comparison=model_comparison_data,
        summary_stats=summary_stats,
        historical_data=historical_chart_data,
        model_details=model_details,
        data_quality_metrics=data_quality_metrics,
        correlation_data=correlation_data,
        monthly_avg_data=monthly_avg_data,
        monthly_precipitation_data=monthly_precipitation_data,
        precipitation_distribution=precipitation_distribution
    )

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    forecast_data = None
    error = None
    start_date_str = ''
    end_date_str = ''

    if forecast_df is None:
        error = "Forecast data not loaded. Please make sure 'samp/daily_data_model_41.csv' exists."
        return render_template('predict.html', error=error)

    if request.method == 'POST':
        try:
            start_date_str = request.form['start_date']
            end_date_str = request.form['end_date']

            # Pandas can handle string indexing on a DatetimeIndex.
            # This avoids timezone comparison issues.
            filtered_df = forecast_df.loc[start_date_str:end_date_str]
            
            if filtered_df.empty:
                # Check if the reason for empty is that start is after end
                try:
                    start_date = dt.datetime.strptime(start_date_str, '%Y-%m-%d')
                    end_date = dt.datetime.strptime(end_date_str, '%Y-%m-%d')
                    if start_date > end_date:
                        error = "Start date cannot be after end date."
                    else:
                        error = "No data available for the selected date range."
                except ValueError:
                    error = "Invalid date format."
            else:
                # Convert dataframe to a list of dictionaries for the template
                forecast_data = filtered_df.reset_index().to_dict('records')

        except Exception as e:
            error = f"An error occurred: {e}"

    return render_template(
        'predict.html', 
        forecast_data=forecast_data, 
        error=error,
        start_date=start_date_str,
        end_date=end_date_str
    )

@app.route('/export')
def export():
    try:
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')

        if not start_date_str or not end_date_str:
            return "Error: Please provide a start and end date.", 400

        # Filter the dataframe for the selected date range
        filtered_df = forecast_df.loc[start_date_str:end_date_str].reset_index()

        # Excel does not support timezone-aware datetimes.
        # Make the 'date' column timezone-unaware before exporting.
        filtered_df['date'] = filtered_df['date'].dt.tz_localize(None)
        
        # Create an in-memory Excel file
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            filtered_df.to_excel(writer, index=False, sheet_name='Forecast')
        
        output.seek(0)

        # Create a filename
        filename = f"forecast_{start_date_str}_to_{end_date_str}.xlsx"

        return Response(
            output,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment;filename={filename}"}
        )
    except Exception as e:
        return str(e)


if __name__ == '__main__':
    app.run(debug=True)
