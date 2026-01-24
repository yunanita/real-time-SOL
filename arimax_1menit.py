# ==================== COMPLETE ARIMAX IMPLEMENTATION - SOL 1 MENIT ====================
# FIXED VERSION - USING LAST 1 YEAR DATA, 15-STEP AHEAD FORECAST
print("="*80)
print("ARIMAX PREDICTION FOR SOL 1 MENIT - DATA 1 TAHUN TERAKHIR")
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

# ==================== 2. LOAD DATA (LAST 1 YEAR) ====================
print("\n" + "="*80)
print("STEP 1: LOADING DATA FROM BIGQUERY (1 YEAR)")
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
# Calculate date range for last 1 year
end_date = datetime.now()
start_date = end_date - timedelta(days=365)  # Approximately 1 year

# Format dates untuk DATETIME
start_date_str = start_date.strftime('%Y-%m-%d')

# Query data for last 1 year
query = f"""
    SELECT 
        datetime,
        open,
        high,
        low,
        close,
        volume
    FROM `time-series-analysis-480002.SOL.SOL_1menit`
    WHERE datetime >= '{start_date_str}'
    ORDER BY datetime
"""

df = client.query(query).to_dataframe()
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

print(f"✅ Data loaded: {len(df):,} rows from last 1 year")
print(f"📅 Date range: {df['datetime'].min()} to {df['datetime'].max()}")
print(f"📊 Data frequency: 1 minute")
print(f"📈 Total data points: {len(df):,}")
print(f"💰 Last close price: ${df['close'].iloc[-1]:.4f}")

# Quick data check
print(f"\n📋 Data quality check:")
print(f"  Missing values in close: {df['close'].isna().sum():,}")
print(f"  Zero values in close: {(df['close'] == 0).sum():,}")
print(f"  Min price: ${df['close'].min():.4f}")
print(f"  Max price: ${df['close'].max():.4f}")
print(f"  Mean price: ${df['close'].mean():.4f}")

# ==================== 3. STATIONARITY TEST & DIFFERENCING ====================
print("\n" + "="*80)
print("STEP 2: STATIONARITY TESTING & DIFFERENCING")
print("="*80)

# Plot 1: Y sebelum differencing (first 5000 points for clarity)
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
sample_points = min(5000, len(df))
plt.plot(df['close'].values[:sample_points], 'b-', linewidth=0.5, alpha=0.7)
plt.title(f'Y (Close Price) - First {sample_points:,} points (1-minute data)')
plt.xlabel('Time (1-minute intervals)')
plt.ylabel('Price ($)')
plt.grid(True, alpha=0.3)

# Y = target variable (close price)
Y_original = df['close'].values

# Test stationarity on a sample for speed
sample_size = min(10000, len(Y_original))
if sample_size > 1000:
    print(f"🔍 Testing stationarity on sample of {sample_size:,} points...")
    adf_result = adfuller(Y_original[:sample_size], autolag='AIC')
else:
    adf_result = adfuller(Y_original, autolag='AIC')

print(f"ADF Test results:")
print(f"  ADF Statistic: {adf_result[0]:.6f}")
print(f"  p-value: {adf_result[1]:.6f}")
print(f"  Is stationary? {'YES' if adf_result[1] <= 0.05 else 'NO'}")

# Untuk data 1-menit, selalu gunakan differencing
print("\n🔍 For ultra high-frequency data (1-minute), applying systematic differencing...")

# Langkah 1: Selalu lakukan first differencing
Y_diff = np.diff(Y_original)

# Langkah 2: Check variance dan ACF untuk menentukan是否需要 second differencing
from statsmodels.tsa.stattools import acf as sm_acf
acf_diff1 = sm_acf(Y_diff, nlags=5, fft=True)
var_diff1 = np.var(Y_diff)

print(f"\n1. First Differencing:")
print(f"   Data points: {len(Y_diff):,}")
print(f"   ACF at lag 1: {acf_diff1[1]:.3f}")
print(f"   Variance: {var_diff1:.6f}")

# Decision: Jika ACF di lag 1 masih tinggi (> 0.3), pertimbangkan second differencing
if abs(acf_diff1[1]) > 0.3:
    print(f"   ⚠️  ACF at lag 1 is high ({acf_diff1[1]:.3f}), considering second differencing...")
    
    Y_diff2 = np.diff(Y_diff)
    acf_diff2 = sm_acf(Y_diff2, nlags=5, fft=True)
    var_diff2 = np.var(Y_diff2)
    
    print(f"\n2. Second Differencing:")
    print(f"   Data points: {len(Y_diff2):,}")
    print(f"   ACF at lag 1: {acf_diff2[1]:.3f}")
    print(f"   Variance: {var_diff2:.6f}")
    print(f"   Variance ratio (diff2/diff1): {var_diff2/var_diff1:.3f}")
    
    # Decision rule: Jika variance tidak meningkat terlalu banyak (< 2x), gunakan d=2
    if var_diff2/var_diff1 < 2.0:
        d = 2
        Y = Y_diff2
        print(f"\n✅ Using d={d} (second differencing)")
        print(f"   Reason: ACF improved ({acf_diff1[1]:.3f} → {acf_diff2[1]:.3f}) and variance increase acceptable ({var_diff2/var_diff1:.3f}x)")
    else:
        d = 1
        Y = Y_diff
        print(f"\n✅ Using d={d} (first differencing)")
        print(f"   Reason: Variance increase too high ({var_diff2/var_diff1:.3f}x)")
else:
    d = 1
    Y = Y_diff
    print(f"\n✅ Using d={d} (first differencing)")
    print(f"   Reason: ACF at lag 1 is acceptable ({acf_diff1[1]:.3f})")

print(f"\n📊 Final data for modeling:")
print(f"  Differencing order (d): {d}")
print(f"  Y shape: {Y.shape}")
print(f"  Total points: {len(Y):,}")

# Plot 2: Y setelah differencing (first 5000 points)
plt.subplot(2, 2, 2)
sample_points_y = min(5000, len(Y))
plt.plot(Y[:sample_points_y], 'r-', linewidth=0.5, alpha=0.7)
plt.title(f'Y (Close Price) - After Differencing (d={d}, first {sample_points_y:,} points)')
plt.xlabel('Time (1-minute intervals)')
plt.ylabel('Difference' if d > 0 else 'Price ($)')
plt.grid(True, alpha=0.3)
if d > 0:
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)

# ==================== 4. ACF & PACF ANALYSIS ====================
print("\n" + "="*80)
print("STEP 3: ACF & PACF ANALYSIS")
print("="*80)

# Use a subset for ACF/PACF analysis to avoid memory issues
acf_sample_size = min(5000, len(Y))

# Plot 3: ACF
plt.subplot(2, 2, 3)
plot_acf(Y[:acf_sample_size], lags=100, ax=plt.gca(), alpha=0.01)  # 99% confidence
plt.title(f'ACF Plot (100 lags, first {acf_sample_size:,} points)')
plt.xlabel('Lag (1-minute intervals)')
plt.ylabel('Autocorrelation')
plt.grid(True, alpha=0.3)

# Plot 4: PACF
plt.subplot(2, 2, 4)
plot_pacf(Y[:acf_sample_size], lags=100, ax=plt.gca(), alpha=0.01, method='ywm')
plt.title(f'PACF Plot (100 lags, first {acf_sample_size:,} points)')
plt.xlabel('Lag (1-minute intervals)')
plt.ylabel('Partial Autocorrelation')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Determine p and q from ACF/PACF
print("\n🔍 Determining p and q from ACF/PACF patterns...")

from statsmodels.tsa.stattools import acf, pacf

acf_values = acf(Y[:acf_sample_size], nlags=60, fft=True)
pacf_values = pacf(Y[:acf_sample_size], nlags=60, method='ywm')

conf_bound = 2.58 / np.sqrt(acf_sample_size)  # 99% confidence untuk data 1-menit

print(f"  99% Confidence bound: ±{conf_bound:.3f}")
print(f"  (Using strict thresholds for 1-minute data)")

# Find significant lags dengan threshold tinggi
significant_acf = []
significant_pacf = []

for lag in range(1, 61):
    if abs(acf_values[lag]) > conf_bound * 1.5:  # Threshold lebih tinggi
        significant_acf.append(lag)
    if abs(pacf_values[lag]) > conf_bound * 1.5:
        significant_pacf.append(lag)

print(f"  Significant ACF lags (q candidates): {significant_acf[:15]}")
print(f"  Significant PACF lags (p candidates): {significant_pacf[:15]}")

# Check for micro-seasonal patterns (1, 5, 10, 15, 30, 60 minutes)
print(f"\n🔍 Checking for micro-seasonal patterns (1-minute data)...")
micro_seasonal_lags = [1, 2, 3, 4, 5, 10, 15, 30, 60, 120, 180, 240, 300]
micro_seasonal_acf = []

for lag in micro_seasonal_lags:
    if lag < len(acf_values):
        if abs(acf_values[lag]) > conf_bound:
            micro_seasonal_acf.append((lag, acf_values[lag]))

if micro_seasonal_acf:
    print(f"  Significant micro-seasonal ACF lags: {[(lag, round(val, 3)) for lag, val in micro_seasonal_acf[:10]]}")
else:
    print(f"  No strong micro-seasonal patterns detected")

# Select p and q based on patterns
if len(significant_pacf) > 3 and len(significant_acf) > 3:
    # High autocorrelation in both
    p = 3
    q = 3
    print(f"  Pattern: High autocorrelation in both ACF & PACF → ARMA({p},{q})")
elif len(significant_pacf) > len(significant_acf):
    # More PACF significant
    p = 2
    q = 1
    print(f"  Pattern: More PACF significant → AR({p}) with MA({q})")
elif len(significant_acf) > len(significant_pacf):
    # More ACF significant
    p = 1
    q = 2
    print(f"  Pattern: More ACF significant → MA({q}) with AR({p})")
else:
    # Default untuk data 1-menit
    p = 2
    q = 2
    print(f"  Pattern: Default for 1-minute data → ARMA({p},{q})")

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
if d > 0:
    Y_aligned = Y  # Y sudah mulai dari index 1 (karena differencing)
    X_aligned = X[1:]  # Skip baris pertama (NaN) untuk alignment dengan Y
else:
    Y_aligned = Y  # Y mulai dari index 0
    X_aligned = X[1:]  # Skip baris pertama (NaN)

# Ensure same length
min_len = min(len(Y_aligned), len(X_aligned))
Y_aligned = Y_aligned[:min_len]
X_aligned = X_aligned[:min_len]

print(f"✅ Data aligned:")
print(f"  Y shape: {Y_aligned.shape}")
print(f"  X shape: {X_aligned.shape}")
print(f"  Alignment: X(t-1) untuk memprediksi Y(t)")

# ==================== 6. TRAIN-TEST SPLIT ====================
print("\n" + "="*80)
print("STEP 5: TRAIN-TEST SPLIT")
print("="*80)

# Untuk data 1-menit
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

# Untuk data 1-menit, gunakan subset untuk model training
train_subset_size = min(20000, len(Y_train))
train_subset = Y_train[:train_subset_size]
X_train_subset = X_train[:train_subset_size]

print(f"  Using subset of {train_subset_size:,} points for model training...")

# Model candidates untuk data 1-menit
candidate_models = [
    (2, 0, 2),    # ARMA(2,2) - baseline
    (1, 0, 1),    # ARMA(1,1) - simple
    (3, 0, 3),    # ARMA(3,3) - untuk autocorrelation tinggi
    (1, 0, 2),    # ARMA(1,2)
    (2, 0, 1),    # ARMA(2,1)
]

models_results = []

for order in candidate_models:
    p_test, d_test, q_test = order
    print(f"\n  Trying ARIMAX({p_test},{0},{q_test}) on subset...")
    
    try:
        model = SARIMAX(
            endog=train_subset,
            exog=X_train_subset,
            order=(p_test, 0, q_test),
            trend='c',
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        fit_result = model.fit(disp=False, maxiter=100)
        
        # Check residuals
        residuals = fit_result.resid
        if len(residuals) > 10:
            lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
            lb_pvalue = lb_test['lb_pvalue'].values[0]
        else:
            lb_pvalue = 1.0
        
        models_results.append({
            'order': order,
            'model': fit_result,
            'aic': fit_result.aic,
            'bic': fit_result.bic,
            'lb_pvalue': lb_pvalue,
            'residuals_white_noise': lb_pvalue > 0.05 if not np.isnan(lb_pvalue) else False
        })
        
        status = "✅" if lb_pvalue > 0.05 else "⚠️"
        print(f"    {status} Success - AIC: {fit_result.aic:.2f}, BIC: {fit_result.bic:.2f}")
        
    except Exception as e:
        print(f"    ❌ Failed: {str(e)[:50]}")

# Select best model
if models_results:
    # Prioritize models with white noise residuals
    models_results.sort(key=lambda x: (
        not x['residuals_white_noise'],  # False first
        x['aic']
    ))
    
    print(f"\n🏆 MODEL COMPARISON:")
    for i, result in enumerate(models_results[:5], 1):
        wn_status = "✓" if result['residuals_white_noise'] else "✗"
        print(f"{i}. ARIMAX{result['order']}: AIC={result['aic']:.2f}, White Noise={wn_status}")
    
    best_result = models_results[0]
    best_order = best_result['order']
    
    print(f"\n✅ Best model from subset: ARIMAX{best_order}")
    
    # Train final model on larger subset
    print(f"\n🔄 Training final model on larger dataset...")
    final_train_size = min(50000, len(Y_train))
    Y_train_final = Y_train[:final_train_size]
    X_train_final = X_train[:final_train_size]
    
    try:
        final_model = SARIMAX(
            endog=Y_train_final,
            exog=X_train_final,
            order=best_order,
            trend='c',
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        best_fit = final_model.fit(disp=False, maxiter=150)
        print(f"  ✅ Final model trained on {final_train_size:,} points")
        print(f"  Final AIC: {best_fit.aic:.2f}")
        
    except Exception as e:
        print(f"  ❌ Failed to train final model: {e}")
        print(f"  Using subset model instead")
        best_fit = best_result['model']
    
else:
    print("❌ No models trained successfully!")
    # Fallback to simple model
    print("🔄 Trying simple ARMA(1,1) as fallback...")
    try:
        simple_model = SARIMAX(
            endog=Y_train[:10000],
            exog=X_train[:10000],
            order=(1, 0, 1),
            trend='c',
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        best_fit = simple_model.fit(disp=False, maxiter=100)
        best_order = (1, 0, 1)
        print(f"  ✅ Simple model trained successfully")
    except:
        raise Exception("Model training failed completely")

# ==================== 8. FORECASTING & EVALUATION ====================
print("\n" + "="*80)
print("STEP 7: FORECASTING & EVALUATION")
print("="*80)

print("🔮 Forecasting on test data...")

# Forecast on reasonable test size
test_forecast_size = min(1000, len(Y_test))
Y_test_forecast = Y_test[:test_forecast_size]
X_test_forecast = X_test[:test_forecast_size]

try:
    forecast_obj = best_fit.get_forecast(steps=test_forecast_size, exog=X_test_forecast)
    pred_mean = forecast_obj.predicted_mean
    
    if hasattr(pred_mean, 'values'):
        pred_mean = pred_mean.values
    pred_mean = np.array(pred_mean).flatten()
    
    conf_int = forecast_obj.conf_int()
    if hasattr(conf_int, 'values'):
        conf_int = conf_int.values
    
    print(f"✅ Forecasting completed for {test_forecast_size:,} test points")
    
except Exception as e:
    print(f"❌ Forecasting error: {e}")
    print("Using simple prediction...")
    pred_mean = np.zeros(test_forecast_size)
    conf_int = None

# Calculate metrics
r2 = r2_score(Y_test_forecast, pred_mean) if len(Y_test_forecast) == len(pred_mean) else -1
rmse = np.sqrt(mean_squared_error(Y_test_forecast, pred_mean)) if len(Y_test_forecast) == len(pred_mean) else np.nan
mae = mean_absolute_error(Y_test_forecast, pred_mean) if len(Y_test_forecast) == len(pred_mean) else np.nan

# Reconstruct test actual prices
if d == 1:
    # For d=1, reconstruct prices
    last_train_idx = train_size + d
    last_train_price = Y_original[last_train_idx - 1] if last_train_idx - 1 < len(Y_original) else Y_original[-1]
    Y_test_actual_prices = last_train_price + np.cumsum(Y_test_forecast)
    pred_prices = last_train_price + np.cumsum(pred_mean)
    
elif d == 2:
    # For d=2, reconstruct with double cumsum
    last_train_idx = train_size + d
    last_train_price = Y_original[last_train_idx - 1] if last_train_idx - 1 < len(Y_original) else Y_original[-1]
    second_last_price = Y_original[last_train_idx - 2] if last_train_idx - 2 >= 0 else Y_original[0]
    
    Y_test_actual_prices = last_train_price + second_last_price + np.cumsum(np.cumsum(Y_test_forecast))
    pred_prices = last_train_price + second_last_price + np.cumsum(np.cumsum(pred_mean))
    
else:  # d == 0
    Y_test_actual_prices = Y_test_forecast
    pred_prices = pred_mean

# Calculate MAPE on actual prices
mask = Y_test_actual_prices != 0
if np.any(mask):
    mape = np.mean(np.abs((Y_test_actual_prices[mask] - pred_prices[mask]) / Y_test_actual_prices[mask])) * 100
    accuracy = max(0, 100 - mape)
else:
    mape = np.nan
    accuracy = np.nan

# Get residuals for later use
residuals = best_fit.resid if hasattr(best_fit, 'resid') else np.array([])

print(f"\n📈 EVALUATION METRICS:")
print(f"  R-squared (R²): {r2:.4f}")
print(f"  RMSE (differences): {rmse:.6f}")
print(f"  MAE (differences): {mae:.6f}")
if not np.isnan(mape):
    print(f"  MAPE (prices): {mape:.2f}%")
    print(f"  Accuracy (prices): {accuracy:.2f}%")
else:
    print(f"  MAPE: Unable to calculate")

# ==================== 9. 15-STEP AHEAD FORECAST ====================
print("\n" + "="*80)
print("STEP 8: 15-STEP (15-MINUTE) AHEAD FORECAST")
print("="*80)

# Prepare for 15 steps ahead
forecast_steps = 15

# Use last available exogenous data
if len(X_test) > 0:
    last_exog = X_test[-1:].reshape(1, -1)
else:
    last_exog = X_train[-1:].reshape(1, -1)

future_exog = np.repeat(last_exog, forecast_steps, axis=0)

# Forecast
try:
    future_forecast = best_fit.get_forecast(steps=forecast_steps, exog=future_exog)
    future_pred_diff = future_forecast.predicted_mean
    
    if hasattr(future_pred_diff, 'values'):
        future_pred_diff = future_pred_diff.values.flatten()
    else:
        future_pred_diff = np.array(future_pred_diff).flatten()
    
    # Reconstruct prices
    last_price = df['close'].iloc[-1]
    
    if d == 0:
        future_pred_prices = future_pred_diff
    elif d == 1:
        future_pred_prices = last_price + np.cumsum(future_pred_diff)
    else:
        future_pred_prices = last_price + np.cumsum(np.cumsum(future_pred_diff))
    
    # Generate timestamps
    last_date = df['datetime'].iloc[-1]
    forecast_dates = [last_date + timedelta(minutes=(i+1)) for i in range(forecast_steps)]
    
    print(f"\n🔮 15-MINUTE AHEAD FORECAST:")
    print(f"  Current price: ${last_price:.4f}")
    print(f"  Forecast period: {forecast_dates[0]} to {forecast_dates[-1]}")
    
    print(f"\n  Forecast summary:")
    print(f"  First (1 min): ${future_pred_prices[0]:.4f}")
    print(f"  Last (15 min): ${future_pred_prices[-1]:.4f}")
    print(f"  Total change: {((future_pred_prices[-1] - last_price) / last_price * 100):+.4f}%")
    
except Exception as e:
    print(f"❌ Future forecast error: {e}")
    # Simple fallback
    last_price = df['close'].iloc[-1]
    future_pred_prices = [last_price] * forecast_steps
    forecast_dates = [df['datetime'].iloc[-1] + timedelta(minutes=(i+1)) for i in range(forecast_steps)]


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
    table_name = "arimax_1menit"
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
                "timeframe": "1menit",
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
                "data_years": 1.0,
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
                "timeframe": "1menit",
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
                "data_years": 1.0,
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
            
            # Calculate confidence interval
            ci_lower = pred_reconstructed[i] * 0.995
            ci_upper = pred_reconstructed[i] * 1.005
            
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
                "timeframe": "1menit",
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
                "data_years": 1.0,
                "forecast_minute": None
            })
            test_pred_count += 1
    
    # ===== 4. FORECAST DATA (SEMUA) =====
    print(f"  Collecting FORECAST data...")
    forecast_count = 0
    for i in range(len(future_pred_prices)):
        forecast_date = forecast_dates[i] if i < len(forecast_dates) else forecast_dates[-1] + timedelta(minutes=(i-len(forecast_dates)+1))
        forecast_date_utc = forecast_date.tz_localize('UTC') if forecast_date.tz is None else forecast_date.astimezone(pytz.UTC)
        
        # Calculate confidence interval
        ci_lower = future_pred_prices[i] * 0.995
        ci_upper = future_pred_prices[i] * 1.005
        
        records.append({
            "timestamp": forecast_date_utc,
            "price_actual": None,
            "price_predicted": float(future_pred_prices[i]),
            "lower_ci": float(ci_lower),
            "upper_ci": float(ci_upper),
            "data_type": "FORECAST",
            "model_type": "ARIMAX",
            "timeframe": "1menit",
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
            "data_years": 1.0,
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
print(f"  Model: ARIMAX{best_order}")
print(f"  Data period: Last 1 year")
print(f"  Data frequency: 1 minute")
print(f"  Total data points: {len(df):,}")
print(f"  Differencing order: d={d}")

print(f"\n📈 MODEL PERFORMANCE:")
print(f"  AIC: {best_fit.aic:.2f}")
print(f"  R² (on test): {r2:.4f}")
print(f"  RMSE: {rmse:.6f}")

print(f"\n🔮 FUTURE FORECAST (15-MINUTE AHEAD):")
print(f"  Current price: ${last_price:.4f}")
print(f"  Forecast range: {future_pred_prices[0]:.4f} to {future_pred_prices[-1]:.4f}")
print(f"  Total change: {((future_pred_prices[-1] - last_price) / last_price * 100):+.4f}%")

print(f"\n💾 DATA STORAGE:")
print(f"  Status: {'✅ Successfully saved to BigQuery' if save_success else '❌ Failed to save'}")
print(f"  Table: arimax_1menit in PREDIKSI dataset")

print("\n" + "="*80)
print("ARIMAX ANALYSIS FOR 1-MENIT COMPLETE!")
print("="*80)
