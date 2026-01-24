# ==================== COMPLETE ARIMAX IMPLEMENTATION - SOL 1 JAM ====================
# FIXED VERSION - USING LAST 3 YEARS DATA, 12-STEP AHEAD FORECAST
print("="*80)
print("ARIMAX PREDICTION FOR SOL 1 JAM - DATA 3 TAHUN TERAKHIR")
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

# ==================== 2. LOAD DATA (LAST 3 YEARS) ====================
print("\n" + "="*80)
print("STEP 1: LOADING DATA FROM BIGQUERY (3 YEARS)")
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

# Calculate date range for last 3 years
end_date = datetime.now()
start_date = end_date - timedelta(days=3*365)  # Approximately 3 years

# Format dates untuk DATETIME
start_date_str = start_date.strftime('%Y-%m-%d')

# Query data for last 3 years
query = f"""
    SELECT 
        datetime,
        open,
        high,
        low,
        close,
        volume
    FROM `time-series-analysis-480002.SOL.SOL_1jam`
    WHERE datetime >= '{start_date_str}'
    ORDER BY datetime
"""

df = client.query(query).to_dataframe()
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

print(f"✅ Data loaded: {len(df)} rows from last 3 years")
print(f"📅 Date range: {df['datetime'].min()} to {df['datetime'].max()}")
print(f"📊 Data frequency: hourly")

# ==================== 3. STATIONARITY TEST & DIFFERENCING ====================
print("\n" + "="*80)
print("STEP 2: STATIONARITY TESTING & DIFFERENCING")
print("="*80)

# Plot 1: Y sebelum differencing
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.plot(df['close'].values, 'b-', linewidth=2, alpha=0.7)
plt.title('Y (Close Price) - Before Differencing')
plt.xlabel('Time (hours)')
plt.ylabel('Price ($)')
plt.grid(True, alpha=0.3)

# Y = target variable (close price)
Y_original = df['close'].values

# Test stationarity
adf_result = adfuller(Y_original, autolag='AIC')
print(f"ADF Test for original data:")
print(f"  ADF Statistic: {adf_result[0]:.6f}")
print(f"  p-value: {adf_result[1]:.6f}")
print(f"  Is stationary? {'YES' if adf_result[1] <= 0.05 else 'NO'}")

# Apply differencing if needed
if adf_result[1] > 0.05:
    print("\n🔄 Data is not stationary, applying differencing...")
    Y_diff = np.diff(Y_original)
    
    # Test again after differencing
    adf_result_diff = adfuller(Y_diff, autolag='AIC')
    print(f"ADF Test after differencing 1:")
    print(f"  ADF Statistic: {adf_result_diff[0]:.6f}")
    print(f"  p-value: {adf_result_diff[1]:.6f}")
    print(f"  Is stationary? {'YES' if adf_result_diff[1] <= 0.05 else 'NO'}")
    
    if adf_result_diff[1] > 0.05:
        Y_diff2 = np.diff(Y_diff)
        adf_result_diff2 = adfuller(Y_diff2, autolag='AIC')
        print(f"ADF Test after differencing 2:")
        print(f"  ADF Statistic: {adf_result_diff2[0]:.6f}")
        print(f"  p-value: {adf_result_diff2[1]:.6f}")
        if adf_result_diff2[1] <= 0.05:
            d = 2
            Y = Y_diff2
            print("✅ Data stationary after differencing 2 times")
        else:
            d = 1
            Y = Y_diff
            print("⚠️ Data still not fully stationary, using d=1")
    else:
        d = 1
        Y = Y_diff
        print("✅ Data stationary after differencing 1 time")
else:
    print("✅ Data is already stationary")
    d = 0
    Y = Y_original

print(f"\n📊 Final data for modeling:")
print(f"  Differencing order (d): {d}")
print(f"  Y shape: {Y.shape}")

# Plot 2: Y setelah differencing
plt.subplot(2, 2, 2)
plt.plot(Y, 'r-', linewidth=2, alpha=0.7)
plt.title(f'Y (Close Price) - After Differencing (d={d})')
plt.xlabel('Time (hours)')
plt.ylabel('Difference' if d > 0 else 'Price ($)')
plt.grid(True, alpha=0.3)
if d > 0:
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)

# ==================== 4. ACF & PACF ANALYSIS ====================
print("\n" + "="*80)
print("STEP 3: ACF & PACF ANALYSIS")
print("="*80)

# Plot 3: ACF
plt.subplot(2, 2, 3)
plot_acf(Y, lags=40, ax=plt.gca(), alpha=0.05)
plt.title('ACF Plot (40 lags)')
plt.xlabel('Lag (hours)')
plt.ylabel('Autocorrelation')
plt.grid(True, alpha=0.3)

# Plot 4: PACF
plt.subplot(2, 2, 4)
plot_pacf(Y, lags=40, ax=plt.gca(), alpha=0.05, method='ywm')
plt.title('PACF Plot (40 lags)')
plt.xlabel('Lag (hours)')
plt.ylabel('Partial Autocorrelation')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Determine p and q from ACF/PACF
print("\n🔍 Determining p and q from ACF/PACF patterns...")

from statsmodels.tsa.stattools import acf, pacf

acf_values = acf(Y, nlags=24, fft=True)  # 24 lags untuk pola harian
pacf_values = pacf(Y, nlags=24, method='ywm')

conf_bound = 1.96 / np.sqrt(len(Y))

# Find significant lags
significant_acf = []
significant_pacf = []

for lag in range(1, 25):
    if abs(acf_values[lag]) > conf_bound:
        significant_acf.append(lag)
    if abs(pacf_values[lag]) > conf_bound:
        significant_pacf.append(lag)

print(f"  Confidence bound: ±{conf_bound:.3f}")
print(f"  Significant ACF lags (q candidates): {significant_acf}")
print(f"  Significant PACF lags (p candidates): {significant_pacf}")

# Select p and q based on patterns
if len(significant_pacf) > 0 and len(significant_acf) == 0:
    # AR process
    p = 1
    q = 0
    print(f"  Pattern: Only PACF significant → AR({p})")
elif len(significant_acf) > 0 and len(significant_pacf) == 0:
    # MA process
    p = 0
    q = 1
    print(f"  Pattern: Only ACF significant → MA({q})")
elif len(significant_acf) > 0 and len(significant_pacf) > 0:
    # ARMA process
    if 1 in significant_pacf or 1 in significant_acf:
        p = 1
        q = 1
        print(f"  Pattern: Both ACF and PACF significant → ARMA({p},{q})")
    else:
        p = 2
        q = 2
        print(f"  Pattern: Multiple significant lags → ARMA({p},{q})")
else:
    # White noise
    p = 0
    q = 0
    print(f"  Pattern: No significant lags → White noise")

# Consider hourly seasonality (24 hours)
print(f"\n🔍 Considering seasonality (24-hour pattern)...")
seasonal_acf = []
seasonal_pacf = []

# Check for seasonal lags (24, 48, etc.)
for lag in [24, 48, 72]:
    if lag < len(acf_values) and abs(acf_values[lag]) > conf_bound:
        seasonal_acf.append(lag)
    if lag < len(pacf_values) and abs(pacf_values[lag]) > conf_bound:
        seasonal_pacf.append(lag)

print(f"  Seasonal ACF lags: {seasonal_acf}")
print(f"  Seasonal PACF lags: {seasonal_pacf}")

print(f"\n💡 Selected orders: p={p}, d={d}, q={q}")
print(f"   Seasonal consideration: s=24 (hourly)")

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
    Y_aligned = Y  # Y sudah mulai dari index 1
    X_aligned = X[1:]  # Skip baris pertama (NaN)
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

train_size = int(len(Y_aligned) * 0.8)

Y_train = Y_aligned[:train_size]
Y_test = Y_aligned[train_size:]

X_train = X_aligned[:train_size]
X_test = X_aligned[train_size:]

print(f"📊 Train-Test Split:")
print(f"  Training samples: {len(Y_train)} ({train_size/len(Y_aligned)*100:.1f}%)")
print(f"  Testing samples:  {len(Y_test)} ({(1-train_size/len(Y_aligned))*100:.1f}%)")

# ==================== 7. MODEL TRAINING ====================
print("\n" + "="*80)
print("STEP 6: ARIMAX MODEL TRAINING")
print("="*80)

# Try multiple models and select best based on AIC
candidate_models = [
    (1, 0, 1),    # ARMA(1,1) - baseline
    (2, 0, 2),    # ARMA(2,2)
    (1, 0, 2),    # ARMA(1,2)
    (2, 0, 1),    # ARMA(2,1)
    (3, 0, 3),    # ARMA(3,3)
    (1, 0, 0),    # AR(1)
    (0, 0, 1),    # MA(1)
]

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
        
        # Check if residuals are white noise
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
            'loglikelihood': fit_result.llf,
            'lb_pvalue': lb_pvalue,
            'residual_mean': residuals.mean(),
            'residual_std': residuals.std()
        })
        
        print(f"    ✅ Success - AIC: {fit_result.aic:.2f}, BIC: {fit_result.bic:.2f}")
        
    except Exception as e:
        print(f"    ❌ Failed: {str(e)[:50]}")

# Select best model based on AIC
if models_results:
    models_results.sort(key=lambda x: x['aic'])
    
    print(f"\n🏆 MODEL COMPARISON (sorted by AIC):")
    for i, result in enumerate(models_results[:5], 1):
        print(f"{i}. ARIMAX{result['order']}: AIC={result['aic']:.2f}, BIC={result['bic']:.2f}")
    
    best_result = models_results[0]
    best_fit = best_result['model']
    best_order = best_result['order']
    
    print(f"\n✅ Best model: ARIMAX{best_order}")
    print(f"   AIC: {best_result['aic']:.2f}")
    print(f"   BIC: {best_result['bic']:.2f}")
    print(f"   Total differencing: d={d} (manual)")
    print(f"   Residual test p-value: {best_result['lb_pvalue']:.4f}")
    
else:
    print("❌ No models trained successfully!")
    raise Exception("Model training failed")

# ==================== 8. FORECASTING & EVALUATION ====================
print("\n" + "="*80)
print("STEP 7: FORECASTING & EVALUATION")
print("="*80)

print("🔮 Forecasting on test data...")

# Forecast on test data
forecast_obj = best_fit.get_forecast(steps=len(Y_test), exog=X_test)
pred_mean = forecast_obj.predicted_mean

# Convert to numpy array
if hasattr(pred_mean, 'values'):
    pred_mean = pred_mean.values
pred_mean = np.array(pred_mean).flatten()

# Get confidence intervals
conf_int = forecast_obj.conf_int()
if hasattr(conf_int, 'values'):
    conf_int = conf_int.values

print(f"✅ Forecasting completed")

# Calculate metrics
r2 = r2_score(Y_test, pred_mean)
rmse = np.sqrt(mean_squared_error(Y_test, pred_mean))
mae = mean_absolute_error(Y_test, pred_mean)

# Reconstruct test actual prices for MAPE calculation
if d == 1:
    # For d=1, reconstruct prices
    last_train_price = Y_original[train_size] if train_size < len(Y_original) else Y_original[-1]
    Y_test_actual_prices = np.cumsum(Y_test) + last_train_price
    pred_prices = np.cumsum(pred_mean) + last_train_price
    
    # Calculate MAPE on actual prices
    mask = Y_test_actual_prices != 0
    if np.any(mask):
        mape = np.mean(np.abs((Y_test_actual_prices[mask] - pred_prices[mask]) / Y_test_actual_prices[mask])) * 100
    else:
        mape = np.nan
elif d == 0:
    # For d=0, use directly
    Y_test_actual_prices = Y_test
    pred_prices = pred_mean
    mask = Y_test_actual_prices != 0
    if np.any(mask):
        mape = np.mean(np.abs((Y_test_actual_prices[mask] - pred_prices[mask]) / Y_test_actual_prices[mask])) * 100
    else:
        mape = np.nan
else:
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

# Directional accuracy
if len(Y_test) > 1:
    actual_dir = np.diff(Y_test) > 0
    pred_dir = np.diff(pred_mean) > 0
    dir_accuracy = np.mean(actual_dir == pred_dir) * 100
else:
    dir_accuracy = np.nan

print(f"\n📈 EVALUATION METRICS:")
print(f"  R-squared (R²): {r2:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE: {mae:.4f}")
if not np.isnan(mape):
    print(f"  MAPE: {mape:.2f}%")
    accuracy = max(0, 100 - mape)
    print(f"  Accuracy: {accuracy:.2f}%")
print(f"  Directional Accuracy: {dir_accuracy:.2f}%" if not np.isnan(dir_accuracy) else "  Directional Accuracy: N/A")

# Ensure R² is reasonable
if r2 < 0:
    print(f"\n⚠️  Warning: R² is negative ({r2:.4f})")
    print("  Consider simplifying the model or checking data")
    r2_display = 0.0
else:
    r2_display = r2

# ==================== 9. PLOT RESULTS ====================
print("\n" + "="*80)
print("STEP 8: PLOTTING RESULTS")
print("="*80)

plt.figure(figsize=(15, 12))

# Plot 1: Forecast vs Actual (differenced)
plt.subplot(3, 2, 1)
train_idx = np.arange(len(Y_train))
test_idx = np.arange(len(Y_train), len(Y_train) + len(Y_test))

plt.plot(train_idx, Y_train, 'b-', linewidth=1.5, alpha=0.7, label='Training')
plt.plot(test_idx, Y_test, 'g-', linewidth=1.5, alpha=0.7, label='Test Actual')
plt.plot(test_idx, pred_mean, 'r--', linewidth=1.5, label='Forecast')

if conf_int is not None and len(conf_int) == len(test_idx):
    plt.fill_between(test_idx, conf_int[:, 0], conf_int[:, 1], 
                     color='red', alpha=0.2, label='95% CI')

plt.axvline(x=len(Y_train), color='black', linestyle='--', alpha=0.7)
plt.title(f'ARIMAX{best_order} - Differenced Data\nR² = {r2_display:.4f}')
plt.xlabel('Time Index (hours)')
plt.ylabel('Price Difference')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Reconstructed prices
plt.subplot(3, 2, 2)

if d == 0:
    train_prices = Y_original[:len(Y_train)]
    test_prices = Y_test_actual_prices
    pred_reconstructed = pred_prices
elif d == 1:
    train_prices = Y_original[1:1+len(Y_train)]
    test_prices = Y_test_actual_prices
    pred_reconstructed = pred_prices
else:
    train_prices = Y_original[2:2+len(Y_train)]
    test_prices = Y_test_actual_prices
    pred_reconstructed = pred_prices

full_train_idx = np.arange(len(train_prices))
full_test_idx = np.arange(len(train_prices), len(train_prices) + len(test_prices))

plt.plot(full_train_idx, train_prices, 'b-', linewidth=1.5, alpha=0.7, label='Training')
plt.plot(full_test_idx, test_prices, 'g-', linewidth=1.5, alpha=0.7, label='Test Actual')
plt.plot(full_test_idx, pred_reconstructed, 'r--', linewidth=1.5, label='Forecast')
plt.axvline(x=len(train_prices), color='black', linestyle='--', alpha=0.7)
plt.title('Reconstructed Prices (Original Scale)')
plt.xlabel('Time Index (hours)')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Residuals
plt.subplot(3, 2, 3)
residuals = Y_test - pred_mean
plt.plot(test_idx, residuals, 'o-', markersize=3, alpha=0.7)
plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
plt.title('Forecast Residuals')
plt.xlabel('Time Index (hours)')
plt.ylabel('Residual')
plt.grid(True, alpha=0.3)

# Plot 4: Actual vs Predicted scatter
plt.subplot(3, 2, 4)
plt.scatter(Y_test, pred_mean, alpha=0.6, s=20)
min_val = min(Y_test.min(), pred_mean.min())
max_val = max(Y_test.max(), pred_mean.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
plt.title('Actual vs Predicted (Differenced)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.grid(True, alpha=0.3)

# Plot 5: Residual histogram
plt.subplot(3, 2, 5)
plt.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
plt.title('Residual Distribution')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# Plot 6: Cumulative error
plt.subplot(3, 2, 6)
cumulative_error = np.cumsum(residuals)
plt.plot(test_idx, cumulative_error, 'b-', linewidth=1.5)
plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
plt.title('Cumulative Forecast Error')
plt.xlabel('Time Index (hours)')
plt.ylabel('Cumulative Error')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ==================== 10. 12-STEP AHEAD FORECAST ====================
print("\n" + "="*80)
print("STEP 9: 12-STEP (12-HOUR) AHEAD FORECAST")
print("="*80)

# Prepare exogenous for next 12 steps
forecast_steps = 12

# Untuk forecast, kita perlu nilai eksogen untuk 12 step ke depan
# Karena kita tidak punya data future, gunakan nilai terakhir yang tersedia
if len(X_test) > 0:
    # Gunakan nilai terakhir dari X_test sebagai baseline
    last_exog = X_test[-1:].reshape(1, -1)
else:
    last_exog = X_train[-1:].reshape(1, -1)

# Buat exogenous untuk 12 step ke depan (dengan asumsi konstan)
future_exog = np.repeat(last_exog, forecast_steps, axis=0)

# Forecast 12 steps ahead
try:
    future_forecast = best_fit.get_forecast(steps=forecast_steps, exog=future_exog)
    future_pred_diff = future_forecast.predicted_mean
    
    if hasattr(future_pred_diff, 'values'):
        future_pred_diff = future_pred_diff.values.flatten()
    else:
        future_pred_diff = np.array(future_pred_diff).flatten()
    
    future_ci = future_forecast.conf_int()
    
    if hasattr(future_ci, 'values'):
        future_ci = future_ci.values
    
    # Reconstruct to original price
    last_price = df['close'].iloc[-1]
    
    # Reconstruct prices based on differencing order
    if d == 0:
        future_pred_prices = future_pred_diff
    elif d == 1:
        future_pred_prices = last_price + np.cumsum(future_pred_diff)
    else:
        future_pred_prices = last_price + np.cumsum(np.cumsum(future_pred_diff))
    
    # Generate forecast timestamps (12 hours ahead)
    last_date = df['datetime'].iloc[-1]
    forecast_dates = [last_date + timedelta(hours=i+1) for i in range(forecast_steps)]
    
    print(f"\n🔮 12-HOUR AHEAD FORECAST:")
    print(f"  Current price: ${last_price:.4f}")
    print(f"  Forecast period: {forecast_dates[0]} to {forecast_dates[-1]}")
    print(f"\n  Detailed forecast:")
    for i in range(forecast_steps):
        if i < len(future_pred_prices):
            forecast_price = future_pred_prices[i]
            change_pct = ((forecast_price - last_price) / last_price * 100) if i == 0 else ((forecast_price - future_pred_prices[i-1]) / future_pred_prices[i-1] * 100)
            
            if future_ci is not None and i < len(future_ci):
                if d == 0:
                    ci_lower = future_ci[i, 0]
                    ci_upper = future_ci[i, 1]
                elif d == 1:
                    ci_lower = last_price + future_ci[i, 0]
                    ci_upper = last_price + future_ci[i, 1]
                else:
                    ci_lower = future_pred_prices[i] - 1.96 * rmse * 2
                    ci_upper = future_pred_prices[i] + 1.96 * rmse * 2
            else:
                ci_lower = forecast_price * 0.98
                ci_upper = forecast_price * 1.02
            
            print(f"  Hour {i+1:2d} ({forecast_dates[i].strftime('%Y-%m-%d %H:%M')}):")
            print(f"    Price: ${forecast_price:.4f} ({change_pct:+.2f}%)")
            print(f"    95% CI: [${ci_lower:.4f}, ${ci_upper:.4f}]")
    
    # Summary statistics
    print(f"\n📊 Forecast Summary:")
    print(f"  Average forecast price: ${np.mean(future_pred_prices):.4f}")
    print(f"  Minimum forecast: ${np.min(future_pred_prices):.4f}")
    print(f"  Maximum forecast: ${np.max(future_pred_prices):.4f}")
    print(f"  Total change: {((future_pred_prices[-1] - last_price) / last_price * 100):+.2f}%")
    
except Exception as e:
    print(f"❌ Future forecast error: {e}")
    print("Using simple extrapolation...")
    
    # Simple extrapolation
    last_price = df['close'].iloc[-1]
    if d == 1 and len(pred_mean) > 0:
        future_pred_prices = last_price + np.cumsum([pred_mean[-1]] * forecast_steps)
    else:
        future_pred_prices = [last_price] * forecast_steps
    
    future_pred_prices = np.array(future_pred_prices)
    forecast_dates = [df['datetime'].iloc[-1] + timedelta(hours=i+1) for i in range(forecast_steps)]
    
    print(f"  Simple forecast average: ${np.mean(future_pred_prices):.4f}")

# ==================== 11. SAVE TO BIGQUERY ====================
print("\n" + "="*80)
print("STEP 10: SAVING RESULTS TO BIGQUERY")
print("="*80)

def save_to_bigquery():
    """Save results to BigQuery with correct data types"""
    
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
    table_name = "arimax_1jam"
    table_id = f"{dataset_id}.{table_name}"
    
    # Schema
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
        bigquery.SchemaField("data_years", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("forecast_hour", "INTEGER", mode="NULLABLE")
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
    
    # Prepare data for upload
    current_time = datetime.now(pytz.timezone('Asia/Jakarta'))
    current_time_utc = current_time.astimezone(pytz.UTC)
    training_date = current_time_utc.date()
    
    records = []
    
    # Get reconstructed prices
    if d == 0:
        train_prices = Y_original[:len(Y_train)]
        train_dates = df['datetime'].iloc[:len(train_prices)].tolist()
        test_actual_prices = Y_test_actual_prices
        test_pred_prices = pred_prices
        test_start = len(train_prices)
        test_dates = df['datetime'].iloc[test_start:test_start+len(test_actual_prices)].tolist()
    elif d == 1:
        train_prices = Y_original[1:1+len(Y_train)]
        train_dates = df['datetime'].iloc[1:1+len(train_prices)].tolist()
        test_actual_prices = Y_test_actual_prices
        test_pred_prices = pred_prices
        test_start = 1 + len(train_prices)
        test_dates = df['datetime'].iloc[test_start:test_start+len(test_actual_prices)].tolist()
    else:
        train_prices = Y_original[2:2+len(Y_train)]
        train_dates = df['datetime'].iloc[2:2+len(train_prices)].tolist()
        test_actual_prices = Y_test_actual_prices
        test_pred_prices = pred_prices
        test_start = 2 + len(train_prices)
        test_dates = df['datetime'].iloc[test_start:test_start+len(test_actual_prices)].tolist()
    
    # 1. TRAIN_ACTUAL
    for i in range(len(train_prices)):
        if i < len(train_dates):
            timestamp = train_dates[i]
        else:
            timestamp = train_dates[-1] if train_dates else df['datetime'].iloc[0]
        
        timestamp_utc = timestamp.tz_localize('UTC') if timestamp.tz is None else timestamp.astimezone(pytz.UTC)
        
        records.append({
            "timestamp": timestamp_utc,
            "price_actual": float(train_prices[i]),
            "price_predicted": None,
            "lower_ci": None,
            "upper_ci": None,
            "data_type": "TRAIN_ACTUAL",
            "model_type": "ARIMAX",
            "timeframe": "1jam",
            "training_date": training_date,
            "created_at": current_time_utc,
            "mape": float(mape) if not np.isnan(mape) else None,
            "accuracy": float(accuracy) if not np.isnan(accuracy) else None,
            "mse": float(rmse**2),
            "rmse": float(rmse),
            "mae": float(mae),
            "aic": float(best_fit.aic),
            "bic": float(best_fit.bic),
            "residual_mean": float(residuals.mean()) if len(residuals) > 0 else None,
            "residual_std": float(residuals.std()) if len(residuals) > 0 else None,
            "data_years": 3,
            "forecast_hour": None
        })
    
    # 2. TEST_ACTUAL
    for i in range(len(test_actual_prices)):
        if i < len(test_dates):
            timestamp = test_dates[i]
        else:
            timestamp = test_dates[-1] if test_dates else df['datetime'].iloc[-1]
            timestamp = timestamp + timedelta(hours=i+1)
        
        timestamp_utc = timestamp.tz_localize('UTC') if timestamp.tz is None else timestamp.astimezone(pytz.UTC)
        
        records.append({
            "timestamp": timestamp_utc,
            "price_actual": float(test_actual_prices[i]),
            "price_predicted": None,
            "lower_ci": None,
            "upper_ci": None,
            "data_type": "TEST_ACTUAL",
            "model_type": "ARIMAX",
            "timeframe": "1jam",
            "training_date": training_date,
            "created_at": current_time_utc,
            "mape": float(mape) if not np.isnan(mape) else None,
            "accuracy": float(accuracy) if not np.isnan(accuracy) else None,
            "mse": float(rmse**2),
            "rmse": float(rmse),
            "mae": float(mae),
            "aic": float(best_fit.aic),
            "bic": float(best_fit.bic),
            "residual_mean": float(residuals.mean()) if len(residuals) > 0 else None,
            "residual_std": float(residuals.std()) if len(residuals) > 0 else None,
            "data_years": 3,
            "forecast_hour": None
        })
    
    # 3. TEST_PREDICTION
    for i in range(len(test_pred_prices)):
        if i < len(test_dates):
            timestamp = test_dates[i]
        else:
            timestamp = test_dates[-1] if test_dates else df['datetime'].iloc[-1]
            timestamp = timestamp + timedelta(hours=i+1)
        
        timestamp_utc = timestamp.tz_localize('UTC') if timestamp.tz is None else timestamp.astimezone(pytz.UTC)
        
        # Calculate CI
        if conf_int is not None and i < len(conf_int):
            if d == 0:
                ci_lower = test_pred_prices[i] - 1.96 * rmse
                ci_upper = test_pred_prices[i] + 1.96 * rmse
            elif d == 1:
                ci_lower = test_pred_prices[i] - 1.96 * rmse
                ci_upper = test_pred_prices[i] + 1.96 * rmse
            else:
                ci_lower = test_pred_prices[i] - 1.96 * rmse * 2
                ci_upper = test_pred_prices[i] + 1.96 * rmse * 2
        else:
            ci_lower = test_pred_prices[i] - 1.96 * rmse
            ci_upper = test_pred_prices[i] + 1.96 * rmse
        
        records.append({
            "timestamp": timestamp_utc,
            "price_actual": float(test_actual_prices[i]),
            "price_predicted": float(test_pred_prices[i]),
            "lower_ci": float(ci_lower),
            "upper_ci": float(ci_upper),
            "data_type": "TEST_PREDICTION",
            "model_type": "ARIMAX",
            "timeframe": "1jam",
            "training_date": training_date,
            "created_at": current_time_utc,
            "mape": float(mape) if not np.isnan(mape) else None,
            "accuracy": float(accuracy) if not np.isnan(accuracy) else None,
            "mse": float(rmse**2),
            "rmse": float(rmse),
            "mae": float(mae),
            "aic": float(best_fit.aic),
            "bic": float(best_fit.bic),
            "residual_mean": float(residuals.mean()) if len(residuals) > 0 else None,
            "residual_std": float(residuals.std()) if len(residuals) > 0 else None,
            "data_years": 3,
            "forecast_hour": None
        })
    
    # 4. FORECAST - Store 12-hour ahead forecasts
    for i in range(len(future_pred_prices)):
        forecast_date = forecast_dates[i] if i < len(forecast_dates) else forecast_dates[-1] + timedelta(hours=i-len(forecast_dates)+1)
        forecast_date_utc = forecast_date.tz_localize('UTC') if forecast_date.tz is None else forecast_date.astimezone(pytz.UTC)
        
        # Calculate CI for forecast
        if future_ci is not None and i < len(future_ci):
            if d == 0:
                ci_lower = future_ci[i, 0]
                ci_upper = future_ci[i, 1]
            elif d == 1:
                ci_lower = last_price + future_ci[i, 0]
                ci_upper = last_price + future_ci[i, 1]
            else:
                ci_lower = future_pred_prices[i] * 0.98
                ci_upper = future_pred_prices[i] * 1.02
        else:
            ci_lower = future_pred_prices[i] * 0.98
            ci_upper = future_pred_prices[i] * 1.02
        
        records.append({
            "timestamp": forecast_date_utc,
            "price_actual": None,
            "price_predicted": float(future_pred_prices[i]),
            "lower_ci": float(ci_lower),
            "upper_ci": float(ci_upper),
            "data_type": "FORECAST",
            "model_type": "ARIMAX",
            "timeframe": "1jam",
            "training_date": training_date,
            "created_at": current_time_utc,
            "mape": float(mape) if not np.isnan(mape) else None,
            "accuracy": float(accuracy) if not np.isnan(accuracy) else None,
            "mse": float(rmse**2),
            "rmse": float(rmse),
            "mae": float(mae),
            "aic": float(best_fit.aic),
            "bic": float(best_fit.bic),
            "residual_mean": float(residuals.mean()) if len(residuals) > 0 else None,
            "residual_std": float(residuals.std()) if len(residuals) > 0 else None,
            "data_years": 3,
            "forecast_hour": i+1
        })
    
    print(f"  Forecast: {len(future_pred_prices)} records")
    
    # Calculate total expected records
    total_expected = len(train_prices) + len(test_actual_prices) + len(test_pred_prices) + len(future_pred_prices)
    print(f"\n📊 Expected total records: {total_expected}")
    
    # Convert to DataFrame and upload
    df_upload = pd.DataFrame(records)
    
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND
    )
    
    try:
        job = client.load_table_from_dataframe(df_upload, table_id, job_config=job_config)
        job.result()
        
        # Verify
        count_query = f"SELECT COUNT(*) as cnt FROM `{table_id}`"
        count_result = client.query(count_query).to_dataframe()
        
        print(f"\n✅ Data successfully saved to BigQuery:")
        print(f"   Table: {table_id}")
        print(f"   Total rows: {count_result['cnt'].iloc[0]}")
        print(f"   Breakdown:")
        print(f"     - TRAIN_ACTUAL: {len(train_prices)}")
        print(f"     - TEST_ACTUAL: {len(test_actual_prices)}")
        print(f"     - TEST_PREDICTION: {len(test_pred_prices)}")
        print(f"     - FORECAST (12-hour): {len(future_pred_prices)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving to BigQuery: {e}")
        return False

# Execute save function
save_success = save_to_bigquery()

# ==================== 12. FINAL SUMMARY ====================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print(f"\n📊 MODEL SPECIFICATION:")
print(f"  Model: ARIMAX{best_order} (d_total={d})")
print(f"  Data period: Last 3 years ({start_date.date()} to {end_date.date()})")
print(f"  Data frequency: Hourly")
print(f"  Exogenous variables: 4 (open, high, low, volume lagged)")
print(f"  Training samples: {len(Y_train)} differences")
print(f"  Testing samples: {len(Y_test)} differences")

print(f"\n📈 MODEL PERFORMANCE:")
print(f"  AIC: {best_fit.aic:.2f}")
print(f"  BIC: {best_fit.bic:.2f}")
print(f"  R² (differences): {r2_display:.4f}")
if not np.isnan(mape):
    print(f"  MAPE (prices): {mape:.2f}%")
    print(f"  Accuracy (prices): {accuracy:.2f}%")
print(f"  RMSE (differences): {rmse:.4f}")
print(f"  MAE (differences): {mae:.4f}")
if not np.isnan(dir_accuracy):
    print(f"  Directional Accuracy: {dir_accuracy:.2f}%")

print(f"\n🔮 FUTURE FORECAST (12-HOUR AHEAD):")
print(f"  Current price: ${last_price:.4f}")
print(f"  First forecast: ${future_pred_prices[0]:.4f}")
print(f"  Last forecast: ${future_pred_prices[-1]:.4f}")
print(f"  Total change: {((future_pred_prices[-1] - last_price) / last_price * 100):+.2f}%")
print(f"  Forecast dates: {forecast_dates[0].strftime('%Y-%m-%d %H:%M')} to {forecast_dates[-1].strftime('%Y-%m-%d %H:%M')}")

print(f"\n💾 DATA STORAGE:")
print(f"  Status: {'✅ Successfully saved to BigQuery' if save_success else '❌ Failed to save'}")
print(f"  Table: arimax_1jam in PREDIKSI dataset")
print(f"  Data types: TRAIN_ACTUAL, TEST_ACTUAL, TEST_PREDICTION, FORECAST")
print(f"  Data years used: 3")
print(f"  Forecast horizon: 12 hours")

print("\n" + "="*80)
print("ARIMAX ANALYSIS FOR 1-JAM COMPLETE!")
print("="*80)
