# ==================== COMPLETE ARIMAX IMPLEMENTATION - SOL 15 MENIT ====================
# FIXED VERSION - WITH GUARANTEED DIFFERENCING FOR 15-MINUTE DATA
print("="*80)
print("ARIMAX PREDICTION FOR SOL 15 MENIT - DATA 2 TAHUN TERAKHIR (FORCED DIFFERENCING)")
print("="*80)

# ==================== 1. IMPORTS & SETUP ====================
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Statsmodels
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# Scikit-learn for metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Google Cloud
from google.oauth2 import service_account
from google.cloud import bigquery

print("✅ Libraries loaded")

# ==================== 2. LOAD DATA (LAST 2 YEARS) ====================
print("\n" + "="*80)
print("STEP 1: LOADING DATA FROM BIGQUERY (2 YEARS)")
print("="*80)

# Credentials - UPDATED untuk GitHub Actions
import os

# Cek apakah running di GitHub Actions
if os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
    # Running di GitHub Actions - gunakan env variable
    SERVICE_ACCOUNT = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT)
else:
    # Running lokal - gunakan path lokal
    SERVICE_ACCOUNT = "D:\KULIAH NOVIA\SMT 5 NOVIA\Time Series Analysis\arimax\time-series-analysis-480002-e7649b18ed82.json"
    creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT)

client = bigquery.Client(credentials=creds, project=creds.project_id)

# Calculate date range for last 2 years
end_date = datetime.now()
start_date = end_date - timedelta(days=2*365)  # Approximately 2 years

# Format dates untuk DATETIME
start_date_str = start_date.strftime('%Y-%m-%d')

# Query data for last 2 years
query = f"""
    SELECT 
        datetime,
        open,
        high,
        low,
        close,
        volume
    FROM `time-series-analysis-480002.SOL.SOL_15menit`
    WHERE datetime >= '{start_date_str}'
    ORDER BY datetime
"""

df = client.query(query).to_dataframe()
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

print(f"✅ Data loaded: {len(df)} rows from last 2 years")
print(f"📅 Date range: {df['datetime'].min()} to {df['datetime'].max()}")
print(f"📊 Data frequency: 15 minutes")
print(f"📈 Total data points: {len(df)}")
print(f"💰 Last close price: ${df['close'].iloc[-1]:.4f}")

# ==================== 3. FORCED DIFFERENCING ====================
print("\n" + "="*80)
print("STEP 2: FORCED DIFFERENCING FOR 15-MINUTE DATA")
print("="*80)

# Plot 1: Original data
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.plot(df['close'].values, 'b-', linewidth=1, alpha=0.7)
plt.title('Original Close Price (15-minute data)')
plt.xlabel('Time (15-minute intervals)')
plt.ylabel('Price ($)')
plt.grid(True, alpha=0.3)

# Y = target variable (close price)
Y_original = df['close'].values

print("🔍 For 15-minute high-frequency data, applying systematic differencing...")
print("ℹ️  Note: Always differencing for high-frequency data regardless of ADF test")

# Langkah 1: First differencing (always)
Y_diff1 = np.diff(Y_original)
print(f"\n1. First Differencing Applied:")
print(f"   Original data points: {len(Y_original)}")
print(f"   After 1st diff: {len(Y_diff1)}")

# Check ACF of first differenced data
from statsmodels.tsa.stattools import acf as sm_acf
acf_diff1 = sm_acf(Y_diff1, nlags=20, fft=True)
print(f"   ACF at lag 1: {acf_diff1[1]:.3f}")
print(f"   ACF at lag 4 (1 hour): {acf_diff1[4]:.3f}")
print(f"   ACF at lag 8 (2 hours): {acf_diff1[8]:.3f}")

# Decision: If ACF at lag 1 is still high (> 0.2), apply second differencing
if abs(acf_diff1[1]) > 0.2:
    print(f"\n⚠️  ACF at lag 1 is still high ({acf_diff1[1]:.3f}), applying second differencing...")
    
    Y_diff2 = np.diff(Y_diff1)
    acf_diff2 = sm_acf(Y_diff2, nlags=20, fft=True)
    
    print(f"\n2. Second Differencing Applied:")
    print(f"   After 2nd diff: {len(Y_diff2)}")
    print(f"   ACF at lag 1: {acf_diff2[1]:.3f}")
    print(f"   ACF at lag 4 (1 hour): {acf_diff2[4]:.3f}")
    
    # Check variance to avoid over-differencing
    var_diff1 = np.var(Y_diff1)
    var_diff2 = np.var(Y_diff2)
    variance_ratio = var_diff2 / var_diff1 if var_diff1 > 0 else 1.0
    
    print(f"   Variance ratio (diff2/diff1): {variance_ratio:.3f}")
    
    # Decision: Use second differencing if variance doesn't increase too much
    if variance_ratio < 1.5:
        d = 2
        Y = Y_diff2
        print(f"\n✅ Using d={d} (second differencing)")
        print(f"   Reason: ACF improved and variance increase acceptable")
    else:
        d = 1
        Y = Y_diff1
        print(f"\n✅ Using d={d} (first differencing only)")
        print(f"   Reason: Variance increase too high ({variance_ratio:.3f}x)")
else:
    d = 1
    Y = Y_diff1
    print(f"\n✅ Using d={d} (first differencing)")
    print(f"   Reason: ACF at lag 1 is acceptable ({acf_diff1[1]:.3f})")

print(f"\n📊 Final differencing setup:")
print(f"  Differencing order: d={d}")
print(f"  Final data points: {len(Y)}")
print(f"  Data reduction: {len(Y_original) - len(Y)} points lost")

# Plot 2: Differenced data
plt.subplot(2, 2, 2)
plt.plot(Y, 'r-', linewidth=1, alpha=0.7)
plt.title(f'Differenced Series (d={d})')
plt.xlabel('Time (15-minute intervals)')
plt.ylabel('Price Difference')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)

# ==================== 4. ACF & PACF ANALYSIS ====================
print("\n" + "="*80)
print("STEP 3: ACF & PACF ANALYSIS AFTER DIFFERENCING")
print("="*80)

# Plot 3: ACF of differenced data
plt.subplot(2, 2, 3)
plot_acf(Y, lags=60, ax=plt.gca(), alpha=0.01)  # 99% confidence
plt.title(f'ACF Plot After Differencing (d={d})')
plt.xlabel('Lag (15-minute intervals)')
plt.ylabel('Autocorrelation')
plt.grid(True, alpha=0.3)

# Plot 4: PACF of differenced data
plt.subplot(2, 2, 4)
plot_pacf(Y, lags=60, ax=plt.gca(), alpha=0.01, method='ywm')
plt.title(f'PACF Plot After Differencing (d={d})')
plt.xlabel('Lag (15-minute intervals)')
plt.ylabel('Partial Autocorrelation')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Analyze ACF/PACF patterns
print("\n🔍 Analyzing ACF/PACF patterns after differencing...")

acf_values = sm_acf(Y, nlags=40, fft=True)
pacf_values = sm_acf(Y, nlags=40, fft=False)

conf_bound = 2.58 / np.sqrt(len(Y))  # 99% confidence

print(f"  99% Confidence bound: ±{conf_bound:.3f}")

# Find significant lags
significant_acf = []
significant_pacf = []

for lag in range(1, 41):
    if abs(acf_values[lag]) > conf_bound:
        significant_acf.append(lag)
    if abs(pacf_values[lag]) > conf_bound:
        significant_pacf.append(lag)

print(f"\n  Significant ACF lags (q candidates): {significant_acf[:10]}")
print(f"  Significant PACF lags (p candidates): {significant_pacf[:10]}")

# Check for seasonal patterns
print(f"\n🔍 Checking for seasonal patterns (15-minute intervals):")
seasonal_lags = [4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96]  # Hourly and multi-hour patterns
seasonal_patterns = []

for lag in seasonal_lags:
    if lag < len(acf_values) and abs(acf_values[lag]) > conf_bound:
        seasonal_patterns.append((lag, acf_values[lag]))

if seasonal_patterns:
    print(f"  Seasonal patterns found at:")
    for lag, val in seasonal_patterns[:5]:
        hours = lag / 4
        print(f"    Lag {lag} ({hours} hours): ACF = {val:.3f}")
else:
    print(f"  No strong seasonal patterns detected")

# Determine p and q based on patterns
print(f"\n🔍 Determining ARIMA orders:")
if len(significant_pacf) == 0 and len(significant_acf) == 0:
    p = 0
    q = 0
    print(f"  Pattern: White noise → ARIMA(0,{d},0)")
elif len(significant_pacf) > 0 and len(significant_acf) == 0:
    # AR process
    p = min(3, len(significant_pacf))
    q = 0
    print(f"  Pattern: AR process → ARIMA({p},{d},0)")
elif len(significant_acf) > 0 and len(significant_pacf) == 0:
    # MA process
    p = 0
    q = min(3, len(significant_acf))
    print(f"  Pattern: MA process → ARIMA(0,{d},{q})")
else:
    # ARMA process
    p = min(2, len(significant_pacf))
    q = min(2, len(significant_acf))
    print(f"  Pattern: ARMA process → ARIMA({p},{d},{q})")

# If ACF at lag 1 is still significant, increase q
if 1 in significant_acf and q < 2:
    q = max(q, 1)
    print(f"  Adjusted: ACF at lag 1 significant, setting q={q}")

# If PACF at lag 1 is still significant, increase p
if 1 in significant_pacf and p < 2:
    p = max(p, 1)
    print(f"  Adjusted: PACF at lag 1 significant, setting p={p}")

print(f"\n💡 Selected orders: p={p}, d={d}, q={q}")

# ==================== 5. PREPARE EXOGENOUS VARIABLES ====================
print("\n" + "="*80)
print("STEP 4: PREPARE EXOGENOUS VARIABLES")
print("="*80)

# Prepare 4 exogenous variables: open, high, low, volume from previous period (t-1)
X_vars = []

for col in ['open', 'high', 'low', 'volume']:
    # Create lagged version (t-1)
    lagged = df[col].values.copy()
    lagged = np.roll(lagged, 1)
    lagged[0] = np.nan
    X_vars.append(lagged)

X = np.column_stack(X_vars)

# Align Y and X properly
# Karena Y sudah di-differencing, kita mulai dari index yang sesuai
if d == 1:
    Y_aligned = Y  # Y sudah mulai dari index 1
    X_aligned = X[1:]  # Skip baris pertama (NaN)
elif d == 2:
    Y_aligned = Y  # Y sudah mulai dari index 2
    X_aligned = X[2:]  # Skip 2 baris pertama
else:
    Y_aligned = Y
    X_aligned = X[1:]  # Default: skip baris pertama

# Ensure same length
min_len = min(len(Y_aligned), len(X_aligned))
Y_aligned = Y_aligned[:min_len]
X_aligned = X_aligned[:min_len]

print(f"✅ Data aligned:")
print(f"  Y shape: {Y_aligned.shape}")
print(f"  X shape: {X_aligned.shape}")
print(f"  Note: X contains lagged (t-1) values of open, high, low, volume")

# ==================== 6. TRAIN-TEST SPLIT ====================
print("\n" + "="*80)
print("STEP 5: TRAIN-TEST SPLIT")
print("="*80)

train_size = int(len(Y_aligned) * 0.8)

Y_train = Y_aligned[:train_size]
Y_test = Y_aligned[train_size:]

X_train = X_aligned[:train_size]
X_test = X_aligned[train_size:]

print(f"📊 Train-Test Split:")
print(f"  Training samples: {len(Y_train)} ({train_size/len(Y_aligned)*100:.1f}%)")
print(f"  Testing samples:  {len(Y_test)} ({(1-train_size/len(Y_aligned))*100:.1f}%)")
print(f"  Total aligned samples: {len(Y_aligned)}")

# ==================== 7. MODEL TRAINING ====================
print("\n" + "="*80)
print("STEP 6: ARIMAX MODEL TRAINING")
print("="*80)

# Model candidates - berdasarkan analisis ACF/PACF
candidate_models = [
    (p, 0, q),      # Hasil dari analisis
    (1, 0, 1),      # ARMA(1,1) baseline
    (2, 0, 2),      # ARMA(2,2) untuk autocorrelation tinggi
    (0, 0, 1),      # MA(1)
    (1, 0, 0),      # AR(1)
]

# Jika p atau q adalah 0, tambahkan beberapa kombinasi
if p == 0 or q == 0:
    candidate_models.extend([(1, 0, 2), (2, 0, 1)])

models_results = []

for order in candidate_models:
    p_test, d_test, q_test = order
    print(f"\n  Trying ARIMAX({p_test},{0},{q_test})...")
    
    try:
        model = SARIMAX(
            endog=Y_train,
            exog=X_train,
            order=(p_test, 0, q_test),
            trend='c',
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        fit_result = model.fit(disp=False, maxiter=200)
        
        # Check residuals
        residuals = fit_result.resid
        if len(residuals) > 10:
            lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
            lb_pvalue = lb_test['lb_pvalue'].values[0]
        else:
            lb_pvalue = 1.0
        
        # Calculate ACF of residuals
        if len(residuals) > 1:
            resid_acf = abs(sm_acf(residuals, nlags=1, fft=True)[1])
        else:
            resid_acf = 0
        
        models_results.append({
            'order': order,
            'model': fit_result,
            'aic': fit_result.aic,
            'bic': fit_result.bic,
            'lb_pvalue': lb_pvalue,
            'residuals_white_noise': lb_pvalue > 0.05,
            'resid_acf_lag1': resid_acf
        })
        
        status = "✅" if lb_pvalue > 0.05 else "⚠️"
        print(f"    {status} Success - AIC: {fit_result.aic:.2f}, BIC: {fit_result.bic:.2f}")
        print(f"      Residual test p-value: {lb_pvalue:.4f}")
        
    except Exception as e:
        print(f"    ❌ Failed: {str(e)[:50]}")

# Select best model
if models_results:
    # Sort by: 1. White noise residuals, 2. AIC, 3. Simple model
    models_results.sort(key=lambda x: (
        not x['residuals_white_noise'],
        x['aic'],
        sum(x['order'])  # Prefer simpler models
    ))
    
    print(f"\n🏆 MODEL COMPARISON:")
    for i, result in enumerate(models_results[:5], 1):
        wn_status = "✓" if result['residuals_white_noise'] else "✗"
        print(f"{i}. ARIMAX{result['order']}: AIC={result['aic']:.2f}, White Noise={wn_status}")
    
    best_result = models_results[0]
    best_fit = best_result['model']
    best_order = best_result['order']
    
    print(f"\n✅ Best model: ARIMAX{best_order}")
    print(f"   AIC: {best_result['aic']:.2f}")
    print(f"   BIC: {best_result['bic']:.2f}")
    print(f"   Total differencing: d={d} (manual)")
    print(f"   Residuals white noise: {'YES' if best_result['residuals_white_noise'] else 'NO'}")
    
else:
    print("❌ No models trained successfully!")
    raise Exception("Model training failed")

# ==================== 8. FORECASTING & EVALUATION ====================
print("\n" + "="*80)
print("STEP 7: FORECASTING & EVALUATION")
print("="*80)

print("🔮 Forecasting on test data...")

# Forecast on test data
try:
    forecast_obj = best_fit.get_forecast(steps=len(Y_test), exog=X_test)
    pred_mean = forecast_obj.predicted_mean
    
    if hasattr(pred_mean, 'values'):
        pred_mean = pred_mean.values
    pred_mean = np.array(pred_mean).flatten()
    
    # Get confidence intervals
    conf_int = forecast_obj.conf_int()
    if hasattr(conf_int, 'values'):
        conf_int = conf_int.values
    
    print(f"✅ Forecasting completed")
    
except Exception as e:
    print(f"❌ Forecasting error: {e}")
    print("Using simple prediction...")
    pred_mean = np.zeros(len(Y_test))
    conf_int = None

# Calculate metrics
r2 = r2_score(Y_test, pred_mean)
rmse = np.sqrt(mean_squared_error(Y_test, pred_mean))
mae = mean_absolute_error(Y_test, pred_mean)

# Reconstruct prices for interpretation
if d == 1:
    # For d=1, reconstruct prices
    last_train_price = Y_original[train_size] if train_size < len(Y_original) else Y_original[-1]
    Y_test_actual_prices = np.cumsum(Y_test) + last_train_price
    pred_prices = np.cumsum(pred_mean) + last_train_price
    
    # Calculate MAPE
    mask = Y_test_actual_prices != 0
    if np.any(mask):
        mape = np.mean(np.abs((Y_test_actual_prices[mask] - pred_prices[mask]) / Y_test_actual_prices[mask])) * 100
    else:
        mape = np.nan
elif d == 2:
    # For d=2, reconstruct with double cumsum
    last_train_price = Y_original[train_size] if train_size < len(Y_original) else Y_original[-1]
    second_last_price = Y_original[train_size-1] if train_size-1 >= 0 else Y_original[0]
    
    Y_test_actual_prices = np.cumsum(np.cumsum(Y_test)) + last_train_price + second_last_price
    pred_prices = np.cumsum(np.cumsum(pred_mean)) + last_train_price + second_last_price
    
    mask = Y_test_actual_prices != 0
    if np.any(mask):
        mape = np.mean(np.abs((Y_test_actual_prices[mask] - pred_prices[mask]) / Y_test_actual_prices[mask])) * 100
    else:
        mape = np.nan
else:
    mape = np.nan

print(f"\n📈 EVALUATION METRICS:")
print(f"  R-squared (R²): {r2:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE: {mae:.4f}")
if not np.isnan(mape):
    print(f"  MAPE (on reconstructed prices): {mape:.2f}%")
    accuracy = max(0, 100 - mape)
    print(f"  Accuracy: {accuracy:.2f}%")

# ==================== 9. 8-STEP AHEAD FORECAST ====================
print("\n" + "="*80)
print("STEP 8: 8-STEP (2-HOUR) AHEAD FORECAST")
print("="*80)

# Prepare for 8 steps ahead (2 hours)
forecast_steps = 8

# Use last available exogenous data
if len(X_test) > 0:
    last_exog = X_test[-1:].reshape(1, -1)
else:
    last_exog = X_train[-1:].reshape(1, -1)

future_exog = np.repeat(last_exog, forecast_steps, axis=0)

try:
    future_forecast = best_fit.get_forecast(steps=forecast_steps, exog=future_exog)
    future_pred_diff = future_forecast.predicted_mean
    
    if hasattr(future_pred_diff, 'values'):
        future_pred_diff = future_pred_diff.values.flatten()
    else:
        future_pred_diff = np.array(future_pred_diff).flatten()
    
    # Reconstruct to original price scale
    last_price = df['close'].iloc[-1]
    
    if d == 0:
        future_pred_prices = future_pred_diff
    elif d == 1:
        future_pred_prices = last_price + np.cumsum(future_pred_diff)
    else:  # d == 2
        future_pred_prices = last_price + np.cumsum(np.cumsum(future_pred_diff))
    
    # Generate forecast timestamps
    last_date = df['datetime'].iloc[-1]
    forecast_dates = [last_date + timedelta(minutes=15*(i+1)) for i in range(forecast_steps)]
    
    print(f"\n🔮 2-HOUR AHEAD FORECAST (8 steps of 15 minutes):")
    print(f"  Current price: ${last_price:.4f}")
    print(f"  Current time: {last_date}")
    print(f"  Forecast period: {forecast_dates[0]} to {forecast_dates[-1]}")
    
    print(f"\n  Detailed forecast:")
    for i in range(forecast_steps):
        forecast_price = future_pred_prices[i]
        if i == 0:
            change_pct = ((forecast_price - last_price) / last_price * 100)
            prev_price = last_price
        else:
            change_pct = ((forecast_price - future_pred_prices[i-1]) / future_pred_prices[i-1] * 100)
            prev_price = future_pred_prices[i-1]
        
        time_str = forecast_dates[i].strftime('%H:%M')
        print(f"  Step {i+1:2d} ({time_str}): ${forecast_price:.4f} ({change_pct:+.3f}%)")
    
    print(f"\n📊 Forecast Summary:")
    print(f"  Average forecast: ${np.mean(future_pred_prices):.4f}")
    print(f"  Minimum forecast: ${np.min(future_pred_prices):.4f}")
    print(f"  Maximum forecast: ${np.max(future_pred_prices):.4f}")
    print(f"  Total 2-hour change: {((future_pred_prices[-1] - last_price) / last_price * 100):+.3f}%")
    
except Exception as e:
    print(f"❌ Future forecast error: {e}")
    # Simple fallback
    last_price = df['close'].iloc[-1]
    future_pred_prices = [last_price] * forecast_steps
    forecast_dates = [df['datetime'].iloc[-1] + timedelta(minutes=15*(i+1)) for i in range(forecast_steps)]
    print(f"  Using simple forecast: constant ${last_price:.4f}")

# ==================== 11. SAVE TO BIGQUERY ====================
print("\n" + "="*80)
print("STEP 10: SAVING ALL DATA TO BIGQUERY (NO DUPLICATES)")
print("="*80)

def save_to_bigquery():
    """Save ALL data to BigQuery - semua data (train, test, forecast)"""
    
    PROJECT_ID = "time-series-analysis-480002"
    PREDICTION_DATASET = "PREDIKSI"
    
    # Ensure dataset exists
    dataset_id = f"{PROJECT_ID}.{PREDICTION_DATASET}"
    try:
        client.get_dataset(dataset_id)
        print(f"✅ Dataset {PREDICTION_DATASET} exists")
    except:
        print(f"📁 Creating dataset {PREDICTION_DATASET}...")
        dataset = bigquery.Dataset(dataset_id)
        dataset.location = "US"
        client.create_dataset(dataset)
        print(f"✅ Dataset created")
    
    # Table name
    table_name = "arimax_15menit"
    table_id = f"{dataset_id}.{table_name}"
    
    # FULL SCHEMA (lengkap)
    schema = [
        bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("price_actual", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("price_predicted", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("lower_ci", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("upper_ci", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("data_type", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("model_type", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("timeframe", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("training_date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("mape", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("accuracy", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("mse", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("rmse", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("mae", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("aic", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("bic", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("residual_mean", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("residual_std", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("data_years", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("forecast_minute", "INTEGER", mode="NULLABLE")
    ]
    
    # ===== CLEAR TABLE DENGAN CREATE OR REPLACE =====
    print(f"\n🔄 Clearing table {table_name}...")
    
    try:
        # Gunakan CREATE OR REPLACE TABLE WHERE FALSE untuk menghapus data tanpa delete table
        clear_query = f"""
            CREATE OR REPLACE TABLE `{table_id}` AS
            SELECT *
            FROM `{table_id}`
            WHERE FALSE
        """
        
        clear_job = client.query(clear_query)
        clear_job.result()
        print(f"🗑️  Table {table_name} cleared (all old data removed)")
        
    except Exception as e:
        # Jika table tidak ada, buat baru
        print(f"ℹ️  Table doesn't exist, creating new one: {e}")
        try:
            table = bigquery.Table(table_id, schema=schema)
            table = client.create_table(table)
            print(f"📊 New table {table_name} created")
        except Exception as create_error:
            print(f"❌ Failed to create table: {create_error}")
            return False
    
    # ===== PREPARE ALL DATA (TRAIN + TEST + FORECAST) =====
    print(f"\n📊 Preparing ALL data for upload...")
    
    current_time = datetime.now(pytz.timezone('Asia/Jakarta'))
    current_time_utc = current_time.astimezone(pytz.UTC)
    training_date = current_time_utc.date()
    
    records = []
    
    # Reconstruct prices based on differencing order
    if d == 0:
        # Data tanpa differencing
        train_prices = Y_original[:len(Y_train)]
        test_actual_prices = Y_original[len(Y_train):len(Y_train)+len(Y_test)]
        pred_reconstructed = pred_mean
        train_dates = df['datetime'].iloc[:len(train_prices)].tolist()
        test_dates = df['datetime'].iloc[len(train_prices):len(train_prices)+len(test_actual_prices)].tolist()
        
    elif d == 1:
        # Data dengan first differencing
        train_prices = Y_original[1:1+len(Y_train)]
        last_train_idx = 1 + len(Y_train)
        test_actual_prices = Y_original[last_train_idx:last_train_idx+len(Y_test)]
        last_train_price = Y_original[last_train_idx-1] if last_train_idx-1 < len(Y_original) else Y_original[-1]
        pred_reconstructed = last_train_price + np.cumsum(pred_mean)
        train_dates = df['datetime'].iloc[1:1+len(train_prices)].tolist()
        test_dates = df['datetime'].iloc[last_train_idx:last_train_idx+len(test_actual_prices)].tolist()
        
    else:  # d == 2
        # Data dengan second differencing
        train_prices = Y_original[2:2+len(Y_train)]
        last_train_idx = 2 + len(Y_train)
        test_actual_prices = Y_original[last_train_idx:last_train_idx+len(Y_test)]
        last_train_price = Y_original[last_train_idx-1] if last_train_idx-1 < len(Y_original) else Y_original[-1]
        second_last_price = Y_original[last_train_idx-2] if last_train_idx-2 >= 0 else Y_original[0]
        pred_reconstructed = last_train_price + second_last_price + np.cumsum(np.cumsum(pred_mean))
        train_dates = df['datetime'].iloc[2:2+len(train_prices)].tolist()
        test_dates = df['datetime'].iloc[last_train_idx:last_train_idx+len(test_actual_prices)].tolist()
    
    print(f"  Data reconstruction completed:")
    print(f"    Train samples: {len(train_prices):,}")
    print(f"    Test actual samples: {len(test_actual_prices):,}")
    print(f"    Test predicted samples: {len(pred_reconstructed):,}")
    print(f"    Forecast samples: {len(future_pred_prices)}")
    
    # ===== TIDAK ADA SAMPLING - SEMUA DATA DISIMPAN =====
    print(f"\n  Sampling settings:")
    print(f"    Train: ALL points (no sampling)")
    print(f"    Test: ALL points (no sampling)")
    print(f"    Forecast: ALL points (no sampling)")
    
    # ===== 1. TRAIN_ACTUAL DATA (SEMUA) =====
    print(f"\n  Collecting TRAIN_ACTUAL data...")
    train_count = 0
    for i in range(len(train_prices)):
        if i < len(train_dates):
            timestamp = train_dates[i]
            timestamp_utc = timestamp.tz_localize('UTC') if timestamp.tz is None else timestamp.astimezone(pytz.UTC)
            
            records.append({
                "timestamp": timestamp_utc,
                "price_actual": float(train_prices[i]),
                "price_predicted": None,
                "lower_ci": None,
                "upper_ci": None,
                "data_type": "TRAIN_ACTUAL",
                "model_type": "ARIMAX",
                "timeframe": "15menit",
                "training_date": training_date,
                "created_at": current_time_utc,
                "mape": float(mape) if not np.isnan(mape) else None,
                "accuracy": float(accuracy) if not np.isnan(mape) else None,
                "mse": float(rmse**2),
                "rmse": float(rmse),
                "mae": float(mae),
                "aic": float(best_fit.aic),
                "bic": float(best_fit.bic),
                "residual_mean": float(residuals.mean()) if 'residuals' in locals() and len(residuals) > 0 else None,
                "residual_std": float(residuals.std()) if 'residuals' in locals() and len(residuals) > 0 else None,
                "data_years": 2.0,
                "forecast_minute": None
            })
            train_count += 1
    
    # ===== 2. TEST_ACTUAL DATA (SEMUA) =====
    print(f"  Collecting TEST_ACTUAL data...")
    test_actual_count = 0
    for i in range(len(test_actual_prices)):
        if i < len(test_dates):
            timestamp = test_dates[i]
            timestamp_utc = timestamp.tz_localize('UTC') if timestamp.tz is None else timestamp.astimezone(pytz.UTC)
            
            records.append({
                "timestamp": timestamp_utc,
                "price_actual": float(test_actual_prices[i]),
                "price_predicted": None,
                "lower_ci": None,
                "upper_ci": None,
                "data_type": "TEST_ACTUAL",
                "model_type": "ARIMAX",
                "timeframe": "15menit",
                "training_date": training_date,
                "created_at": current_time_utc,
                "mape": float(mape) if not np.isnan(mape) else None,
                "accuracy": float(accuracy) if not np.isnan(mape) else None,
                "mse": float(rmse**2),
                "rmse": float(rmse),
                "mae": float(mae),
                "aic": float(best_fit.aic),
                "bic": float(best_fit.bic),
                "residual_mean": float(residuals.mean()) if 'residuals' in locals() and len(residuals) > 0 else None,
                "residual_std": float(residuals.std()) if 'residuals' in locals() and len(residuals) > 0 else None,
                "data_years": 2.0,
                "forecast_minute": None
            })
            test_actual_count += 1
    
    # ===== 3. TEST_PREDICTION DATA (SEMUA) =====
    print(f"  Collecting TEST_PREDICTION data...")
    test_pred_count = 0
    for i in range(len(pred_reconstructed)):
        if i < len(test_dates):
            timestamp = test_dates[i]
            timestamp_utc = timestamp.tz_localize('UTC') if timestamp.tz is None else timestamp.astimezone(pytz.UTC)
            
            # Calculate confidence interval untuk 15-menit
            ci_lower = pred_reconstructed[i] * 0.985
            ci_upper = pred_reconstructed[i] * 1.015
            
            # Get actual price if available
            actual_price = float(test_actual_prices[i]) if i < len(test_actual_prices) else None
            
            records.append({
                "timestamp": timestamp_utc,
                "price_actual": actual_price,
                "price_predicted": float(pred_reconstructed[i]),
                "lower_ci": float(ci_lower),
                "upper_ci": float(ci_upper),
                "data_type": "TEST_PREDICTION",
                "model_type": "ARIMAX",
                "timeframe": "15menit",
                "training_date": training_date,
                "created_at": current_time_utc,
                "mape": float(mape) if not np.isnan(mape) else None,
                "accuracy": float(accuracy) if not np.isnan(mape) else None,
                "mse": float(rmse**2),
                "rmse": float(rmse),
                "mae": float(mae),
                "aic": float(best_fit.aic),
                "bic": float(best_fit.bic),
                "residual_mean": float(residuals.mean()) if 'residuals' in locals() and len(residuals) > 0 else None,
                "residual_std": float(residuals.std()) if 'residuals' in locals() and len(residuals) > 0 else None,
                "data_years": 2.0,
                "forecast_minute": None
            })
            test_pred_count += 1
    
    # ===== 4. FORECAST DATA (SEMUA) =====
    print(f"  Collecting FORECAST data...")
    forecast_count = 0
    for i in range(len(future_pred_prices)):
        forecast_date = forecast_dates[i] if i < len(forecast_dates) else forecast_dates[-1] + timedelta(minutes=15*(i-len(forecast_dates)+1))
        forecast_date_utc = forecast_date.tz_localize('UTC') if forecast_date.tz is None else forecast_date.astimezone(pytz.UTC)
        
        # Calculate confidence interval untuk 15-menit
        ci_lower = future_pred_prices[i] * 0.985
        ci_upper = future_pred_prices[i] * 1.015
        
        records.append({
            "timestamp": forecast_date_utc,
            "price_actual": None,
            "price_predicted": float(future_pred_prices[i]),
            "lower_ci": float(ci_lower),
            "upper_ci": float(ci_upper),
            "data_type": "FORECAST",
            "model_type": "ARIMAX",
            "timeframe": "15menit",
            "training_date": training_date,
            "created_at": current_time_utc,
            "mape": float(mape) if not np.isnan(mape) else None,
            "accuracy": float(accuracy) if not np.isnan(mape) else None,
            "mse": float(rmse**2),
            "rmse": float(rmse),
            "mae": float(mae),
            "aic": float(best_fit.aic),
            "bic": float(best_fit.bic),
            "residual_mean": float(residuals.mean()) if 'residuals' in locals() and len(residuals) > 0 else None,
            "residual_std": float(residuals.std()) if 'residuals' in locals() and len(residuals) > 0 else None,
            "data_years": 2.0,
            "forecast_minute": i+1
        })
        forecast_count += 1
    
    # Summary
    print(f"\n📦 Data preparation completed:")
    print(f"  TRAIN_ACTUAL: {train_count:,} records")
    print(f"  TEST_ACTUAL: {test_actual_count:,} records")
    print(f"  TEST_PREDICTION: {test_pred_count:,} records")
    print(f"  FORECAST: {forecast_count:,} records")
    print(f"  TOTAL: {len(records):,} records")
    
    # Estimate size
    estimated_size_mb = len(records) * 0.0001  # Approx 0.1KB per record
    print(f"  Estimated size: ~{estimated_size_mb:.2f} MB")
    
    # ===== UPLOAD TO BIGQUERY =====
    print(f"\n⏳ Uploading data to BigQuery...")
    
    df_upload = pd.DataFrame(records)
    
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND
    )
    
    try:
        # Upload data
        job = client.load_table_from_dataframe(df_upload, table_id, job_config=job_config)
        job.result()
        
        # Verify upload
        count_query = f"SELECT COUNT(*) as cnt FROM `{table_id}`"
        count_result = client.query(count_query).to_dataframe()
        
        print(f"\n✅ ALL data successfully saved to BigQuery!")
        print(f"   Table: {table_id}")
        print(f"   Total rows in table: {count_result['cnt'].iloc[0]:,}")
        print(f"   Expected rows: {len(records):,}")
        
        # Check if counts match
        if count_result['cnt'].iloc[0] == len(records):
            print(f"   ✅ Upload verification: SUCCESS (counts match)")
        else:
            print(f"   ⚠️  Upload verification: COUNTS DON'T MATCH")
            print(f"      Expected: {len(records):,}, Got: {count_result['cnt'].iloc[0]:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving to BigQuery: {e}")
        
        # Try batch upload if single upload fails
        print(f"🔄 Trying batch upload...")
        try:
            # Upload in smaller batches
            batch_size = 5000
            for i in range(0, len(records), batch_size):
                batch = records[i:i+batch_size]
                df_batch = pd.DataFrame(batch)
                
                if i == 0:
                    job_config = bigquery.LoadJobConfig(
                        schema=schema,
                        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
                    )
                else:
                    job_config = bigquery.LoadJobConfig(
                        schema=schema,
                        write_disposition=bigquery.WriteDisposition.WRITE_APPEND
                    )
                
                job = client.load_table_from_dataframe(df_batch, table_id, job_config=job_config)
                job.result()
                print(f"   Batch {i//batch_size + 1} uploaded: {len(batch):,} records")
            
            print(f"✅ Batch upload successful!")
            return True
            
        except Exception as e2:
            print(f"❌ Batch upload also failed: {e2}")
            return False

# Execute save function
save_success = save_to_bigquery()

# ==================== 11. FINAL SUMMARY ====================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print(f"\n📊 MODEL SPECIFICATION:")
print(f"  Model: ARIMAX{best_order} with d={d} differencing")
print(f"  Data period: Last 2 years")
print(f"  Data frequency: 15 minutes")
print(f"  Total data points: {len(df)}")
print(f"  Training samples: {len(Y_train)}")
print(f"  Testing samples: {len(Y_test)}")

print(f"\n📈 MODEL PERFORMANCE:")
print(f"  AIC: {best_fit.aic:.2f}")
print(f"  BIC: {best_fit.bic:.2f}")
print(f"  R²: {r2:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE: {mae:.4f}")
if not np.isnan(mape):
    print(f"  MAPE: {mape:.2f}%")
    print(f"  Accuracy: {accuracy:.2f}%")

print(f"\n🔮 FUTURE FORECAST (2-HOUR AHEAD):")
print(f"  Current price: ${last_price:.4f}")
print(f"  Forecast range: ${future_pred_prices[0]:.4f} to ${future_pred_prices[-1]:.4f}")
print(f"  Total change: {((future_pred_prices[-1] - last_price) / last_price * 100):+.3f}%")
print(f"  Next forecast (15 min): ${future_pred_prices[0]:.4f}")

print(f"\n💾 DATA STORAGE:")
print(f"  Status: {'✅ Successfully saved to BigQuery' if save_success else '❌ Failed to save'}")
print(f"  Table: arimax_15menit in PREDIKSI dataset")
print(f"  Contents: 8-step forecasts + current price")

print("\n" + "="*80)
print("ARIMAX ANALYSIS FOR 15-MENIT COMPLETE! (WITH GUARANTEED DIFFERENCING)")
print("="*80)