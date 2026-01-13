# ========================
# IMPORT
# ========================
import ccxt
import pandas as pd
import time
import os
import json
from datetime import datetime, timezone
from google.oauth2 import service_account
from pandas_gbq import read_gbq, to_gbq


# ========================
# 1. AUTH & SETUP BIGQUERY
# ========================
def get_credentials():
    """
    Mendapatkan credentials dari:
    1. Environment variable GCP_CREDENTIALS (untuk GitHub Actions)
    2. File JSON lokal (untuk development lokal)
    """
    gcp_creds_json = os.environ.get("GCP_CREDENTIALS")
    
    if gcp_creds_json:
        creds_dict = json.loads(gcp_creds_json)
        credentials = service_account.Credentials.from_service_account_info(creds_dict)
        print("✅ Using credentials from environment variable")
        return credentials
    
    local_file = "time-series-analysis-480002-e7649b18ed82.json"
    if os.path.exists(local_file):
        credentials = service_account.Credentials.from_service_account_file(local_file)
        print("✅ Using credentials from local file")
        return credentials
    
    raise Exception("❌ No credentials found! Set GCP_CREDENTIALS env var or provide JSON file.")


credentials = get_credentials()
project_id = "time-series-analysis-480002"
dataset_id = "SOL"

# Mapping interval ke nama tabel BigQuery
interval_table_map = {
    "1m": "SOL_1menit",
    "15m": "SOL_15menit",
    "1h": "SOL_1jam",
    "1d": "SOL_1hari",
    "1M": "SOL_1bulan"
}

# SOL listing date di KuCoin (Agustus 2021)
SOL_LISTING_DATE_1M = int(datetime(2021, 8, 4, tzinfo=timezone.utc).timestamp() * 1000)
SOL_LISTING_DATE_ALL = int(datetime(2021, 8, 1, tzinfo=timezone.utc).timestamp() * 1000)


# ========================
# 2. CCXT EXCHANGE SETUP
# ========================
exchange = ccxt.kucoin({"enableRateLimit": True})


# ========================
# 3. FETCH HISTORICAL DATA (DARI AWAL LISTING)
# ========================
def fetch_historical(symbol, timeframe, since):
    """
    Fetch SEMUA data historis dari 'since' sampai sekarang.
    Menggunakan pagination untuk ambil jutaan data.
    """
    all_data = []
    limit = 1500
    batch_count = 0

    print(f"   🔄 Fetching {timeframe} from {datetime.utcfromtimestamp(since/1000)}...")
    
    while True:
        try:
            data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        except Exception as e:
            print(f"   ❌ Error fetching {timeframe}: {e}")
            break

        if not data:
            break

        batch_count += 1
        all_data.extend(data)
        
        last_timestamp = data[-1][0]
        
        # Progress log setiap 100 batch
        if batch_count % 100 == 0:
            last_dt = datetime.utcfromtimestamp(last_timestamp / 1000)
            print(f"   📦 {timeframe} Batch {batch_count}: {len(all_data):,} rows | Last: {last_dt}")

        since = last_timestamp + 1
        time.sleep(exchange.rateLimit / 1000)

    print(f"   ✅ {timeframe} DONE: {batch_count} batches, {len(all_data):,} total rows")
    
    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.drop_duplicates(subset="timestamp", keep="last")
        df = df.sort_values("timestamp").reset_index(drop=True)

    return df


# ========================
# 4. FETCH UPDATE DATA (INCREMENTAL)
# ========================
def fetch_update(symbol, timeframe, since):
    """
    Fetch data update dari timestamp terakhir di BigQuery.
    Biasanya hanya beberapa rows (data baru sejak run terakhir).
    """
    all_data = []
    limit = 1500

    print(f"   🔄 Updating {timeframe} from {datetime.utcfromtimestamp(since/1000)}...")
    
    while True:
        try:
            data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        except Exception as e:
            print(f"   ❌ Error fetching {timeframe}: {e}")
            break

        if not data:
            break

        all_data.extend(data)
        last_timestamp = data[-1][0]
        since = last_timestamp + 1
        time.sleep(exchange.rateLimit / 1000)

    print(f"   ✅ {timeframe} Update: {len(all_data)} new rows")
    
    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.drop_duplicates(subset="timestamp", keep="last")
        df = df.sort_values("timestamp").reset_index(drop=True)

    return df


# ========================
# 5. CHECK TABLE STATUS
# ========================
def get_table_row_count(interval):
    """Cek jumlah rows di tabel BigQuery."""
    table = interval_table_map[interval]
    query = f"SELECT COUNT(*) as cnt FROM `{project_id}.{dataset_id}.{table}`"
    
    try:
        df = read_gbq(query, project_id=project_id, credentials=credentials)
        return int(df.loc[0, "cnt"]) if not df.empty else 0
    except Exception:
        return 0


def get_last_timestamp(interval):
    """Ambil timestamp terakhir di tabel BigQuery."""
    table = interval_table_map[interval]
    query = f"SELECT MAX(timestamp) AS last_ts FROM `{project_id}.{dataset_id}.{table}`"
    
    try:
        df = read_gbq(query, project_id=project_id, credentials=credentials)
        if df.empty or pd.isna(df.loc[0, "last_ts"]):
            return None
        return int(df.loc[0, "last_ts"])
    except Exception:
        return None


# ========================
# 6. UPLOAD TO BIGQUERY (WITH DEDUP)
# ========================
def upload_df(df, interval):
    """Upload dataframe ke BigQuery dengan deduplikasi."""
    table = interval_table_map[interval]

    if df.empty:
        print(f"   ⚠️ {interval}: No data to upload")
        return 0

    # Deduplicate
    df = df.drop_duplicates(subset="timestamp", keep="last").copy()
    
    min_ts = df["timestamp"].min()
    max_ts = df["timestamp"].max()

    # Cek existing timestamps di BigQuery
    try:
        query = f"""
            SELECT timestamp FROM `{project_id}.{dataset_id}.{table}`
            WHERE timestamp BETWEEN {min_ts} AND {max_ts}
        """
        existing_df = read_gbq(query, project_id=project_id, credentials=credentials)
        existing = set(existing_df["timestamp"].tolist()) if not existing_df.empty else set()
    except Exception:
        existing = set()

    # Filter hanya yang baru
    df_new = df[~df["timestamp"].isin(existing)].copy()

    if df_new.empty:
        print(f"   ⏸️ {interval}: All data already exists, skip")
        return 0

    # Upload
    print(f"   📤 {interval}: Uploading {len(df_new):,} rows...")
    to_gbq(
        df_new,
        f"{dataset_id}.{table}",
        project_id=project_id,
        credentials=credentials,
        if_exists="append",
        api_method="load_csv"
    )

    print(f"   ✅ {interval}: {len(df_new):,} rows uploaded")
    return len(df_new)


# ========================
# 7. MAIN PIPELINE
# ========================
def run_pipeline():
    """
    Pipeline utama:
    - Jika tabel KOSONG → fetch SEMUA data historis dari awal listing (2021)
    - Jika tabel ADA DATA → fetch hanya data UPDATE (dari timestamp terakhir)
    """
    print(f"\n{'='*70}")
    print(f"🚀 SOL DATA PIPELINE - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"{'='*70}")
    
    start_time = time.time()
    total_uploaded = 0
    summary = {}
    
    # Cek status semua tabel
    print("\n📋 TABLE STATUS:")
    tables_empty = {}
    for tf in interval_table_map:
        count = get_table_row_count(tf)
        tables_empty[tf] = (count == 0)
        status = "❌ EMPTY" if count == 0 else f"✅ {count:,} rows"
        print(f"   {tf}: {status}")
    
    # Process setiap interval
    for tf in interval_table_map:
        print(f"\n{'─'*50}")
        print(f"📊 Processing {tf}...")
        
        if tables_empty[tf]:
            # TABEL KOSONG → Fetch semua data historis
            print(f"   Mode: HISTORICAL (full fetch from listing date)")
            if tf == "1m":
                since = SOL_LISTING_DATE_1M
            else:
                since = SOL_LISTING_DATE_ALL
            
            df = fetch_historical("SOL/USDT", tf, since)
        else:
            # TABEL ADA DATA → Fetch update saja
            print(f"   Mode: UPDATE (incremental)")
            last_ts = get_last_timestamp(tf)
            since = last_ts + 1
            df = fetch_update("SOL/USDT", tf, since)
        
        # Upload ke BigQuery
        if not df.empty:
            uploaded = upload_df(df, tf)
            summary[tf] = {"fetched": len(df), "uploaded": uploaded}
            total_uploaded += uploaded
        else:
            summary[tf] = {"fetched": 0, "uploaded": 0}
    
    # Final summary
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"📊 FINAL SUMMARY:")
    for tf, stats in summary.items():
        status = "✅" if stats["uploaded"] > 0 else "⏸️"
        print(f"   {status} {tf}: {stats['uploaded']:,} uploaded (fetched: {stats['fetched']:,})")
    print(f"{'='*70}")
    print(f"⏱️ Total time: {elapsed:.2f}s")
    print(f"📈 Total uploaded: {total_uploaded:,} rows")
    print(f"{'='*70}\n")


# ========================
# RUN
# ========================
if __name__ == "__main__":
    run_pipeline()