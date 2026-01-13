# ========================
# IMPORT
# ========================
import ccxt
import pandas as pd
import time
import os
import json
from datetime import datetime, timedelta
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
    # Cek apakah ada environment variable (GitHub Actions)
    gcp_creds_json = os.environ.get("GCP_CREDENTIALS")
    
    if gcp_creds_json:
        # Parse JSON dari environment variable
        creds_dict = json.loads(gcp_creds_json)
        credentials = service_account.Credentials.from_service_account_info(creds_dict)
        print("✅ Using credentials from environment variable")
        return credentials
    
    # Fallback ke file lokal (untuk development)
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


# ========================
# 2. CCXT EXCHANGE SETUP
# ========================
exchange = ccxt.kucoin({"enableRateLimit": True})


# ========================
# 3. FETCH DATA
# ========================
def fetch_interval(symbol, timeframe, since=None):
    """
    Fetch OHLCV data dari exchange dengan pagination.
    Akan terus fetch sampai SEMUA data dari 'since' sampai sekarang.
    Jika since=None, fetch dari awal listing (bisa jutaan rows).
    """
    all_data = []
    limit = 1500  # Max per request dari KuCoin
    seen_timestamps = set()  # Track timestamp untuk hindari duplikat
    batch_count = 0

    print(f"   🔄 Starting fetch {timeframe}...")
    
    while True:
        try:
            data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        except Exception as e:
            print(f"   ❌ Error fetching {timeframe}: {e}")
            break

        if not data:
            print(f"   📭 No more data for {timeframe}")
            break

        batch_count += 1
        
        # Filter duplikat saat fetch (layer 1)
        new_in_batch = 0
        for candle in data:
            ts = candle[0]
            if ts not in seen_timestamps:
                seen_timestamps.add(ts)
                all_data.append(candle)
                new_in_batch += 1

        last_timestamp = data[-1][0]
        last_dt = datetime.utcfromtimestamp(last_timestamp / 1000)
        
        # Progress log setiap 10 batch
        if batch_count % 10 == 0 or len(data) < limit:
            print(f"   📦 {timeframe} Batch {batch_count}: +{new_in_batch} rows | Total: {len(all_data)} | Last: {last_dt}")

        # Jika data yang didapat kurang dari limit, berarti sudah sampai akhir
        if len(data) < limit:
            print(f"   ✅ {timeframe} COMPLETE: {batch_count} batches, {len(all_data)} total rows")
            break

        # Update 'since' untuk fetch batch berikutnya
        since = last_timestamp + 1
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        # Layer 2: Drop duplicates (safety net)
        before_dedup = len(df)
        df = df.drop_duplicates(subset="timestamp", keep="last")
        after_dedup = len(df)
        if before_dedup != after_dedup:
            print(f"   ⚠️ Removed {before_dedup - after_dedup} duplicates during fetch")
        df = df.sort_values("timestamp").reset_index(drop=True)

    return df


# ========================
# 4. CHECK RAW DATA
# ========================
def check_raw_data(df, name=""):
    print(f"\n===== CHECK RAW DATA: {name} =====")
    print("Total rows:", len(df))
    
    if "timestamp" in df.columns:
        dup_count = df.duplicated("timestamp").sum()
        print(f"Duplicated timestamp: {dup_count}", "⚠️ WARNING!" if dup_count > 0 else "✅")
        print("Min:", df["timestamp"].min())
        print("Max:", df["timestamp"].max())

    print("=" * 60)


# ========================
# 4.5 CHECK DUPLICATES IN BIGQUERY (VERIFICATION)
# ========================
def check_duplicates_in_bigquery():
    """
    Fungsi untuk mengecek apakah ada duplikat di BigQuery.
    Jalankan ini untuk verifikasi data.
    """
    print("\n" + "="*60)
    print("🔍 CHECKING DUPLICATES IN BIGQUERY...")
    print("="*60)
    
    has_duplicates = False
    
    for tf, table in interval_table_map.items():
        query = f"""
            SELECT timestamp, COUNT(*) as cnt
            FROM `{project_id}.{dataset_id}.{table}`
            GROUP BY timestamp
            HAVING cnt > 1
            LIMIT 10
        """
        
        try:
            df = read_gbq(query, project_id=project_id, credentials=credentials)
            if not df.empty:
                print(f"❌ {tf} ({table}): FOUND {len(df)} duplicate timestamps!")
                print(df.head())
                has_duplicates = True
            else:
                print(f"✅ {tf} ({table}): No duplicates")
        except Exception as e:
            print(f"⚠️ {tf} ({table}): Cannot check - {e}")
    
    if has_duplicates:
        print("\n⚠️ DUPLICATES FOUND! Run cleanup if needed.")
    else:
        print("\n✅ ALL TABLES ARE CLEAN - No duplicates!")
    
    print("="*60 + "\n")
    return not has_duplicates


# ========================
# 5. GET EXISTING TIMESTAMPS (FIX UTAMA)
# ========================
def get_existing_timestamps(interval, start_ts=None, end_ts=None):
    table = interval_table_map[interval]

    where_clause = ""
    if start_ts is not None and end_ts is not None:
        where_clause = f"WHERE timestamp BETWEEN {start_ts} AND {end_ts}"
    elif start_ts is not None:
        where_clause = f"WHERE timestamp >= {start_ts}"

    query = f"""
        SELECT timestamp
        FROM `{project_id}.{dataset_id}.{table}`
        {where_clause}
    """

    try:
        df = read_gbq(query, project_id=project_id, credentials=credentials)
        return set(df["timestamp"].tolist()) if not df.empty else set()
    except Exception:
        print(f"⚠️ Table {table} belum ada / belum bisa diakses. Dianggap kosong.")
        return set()


# ========================
# 6. UPLOAD (DEDUP) - TRIPLE PROTECTION
# ========================
def upload_df(df, interval, mode="append"):
    """
    Upload dataframe ke BigQuery dengan TRIPLE deduplikasi:
    1. Drop duplicates dalam dataframe sendiri
    2. Filter timestamp yang sudah ada di BigQuery
    3. Final validation sebelum upload
    """
    table = interval_table_map[interval]

    if df.empty:
        print(f"⚠️ {interval} kosong, skip.")
        return 0

    # LAYER 1: Pastikan tidak ada duplikat dalam dataframe
    df = df.drop_duplicates(subset="timestamp", keep="last").copy()
    
    min_ts = df["timestamp"].min()
    max_ts = df["timestamp"].max()

    # LAYER 2: Cek timestamp yang sudah ada di BigQuery
    print(f"   🔍 {interval}: Checking existing data in BigQuery...")
    existing = get_existing_timestamps(interval, min_ts, max_ts)
    
    # Filter hanya data yang BELUM ada di BigQuery
    df_new = df[~df["timestamp"].isin(existing)].copy()

    if df_new.empty:
        print(f"   ⏸️ {interval}: Semua {len(df)} rows sudah ada di BigQuery, skip upload.")
        return 0

    # LAYER 3: Final validation - pastikan df_new tidak ada duplikat
    final_count = len(df_new)
    df_new = df_new.drop_duplicates(subset="timestamp", keep="last")
    if len(df_new) != final_count:
        print(f"   ⚠️ Removed {final_count - len(df_new)} duplicates in final check")

    # Upload ke BigQuery
    print(f"   📤 {interval}: Uploading {len(df_new)} new rows...")
    to_gbq(
        df_new,
        f"{dataset_id}.{table}",
        project_id=project_id,
        credentials=credentials,
        if_exists="append",
        api_method="load_csv"
    )

    print(f"   ✅ {interval} → {table}: {len(df_new)} rows UPLOADED")
    print(f"      (Fetched: {len(df)}, Already in DB: {len(existing)}, New: {len(df_new)})")
    return len(df_new)


# ========================
# 7. GET LAST TIMESTAMP
# ========================
def get_last_timestamp(interval):
    table = interval_table_map[interval]
    query = f"""
        SELECT MAX(timestamp) AS last_ts
        FROM `{project_id}.{dataset_id}.{table}`
    """

    try:
        df = read_gbq(query, project_id=project_id, credentials=credentials)
        if df.empty or pd.isna(df.loc[0, "last_ts"]):
            return None
        return int(df.loc[0, "last_ts"])
    except Exception:
        return None


# ========================
# 8. FETCH UPDATE (OPTIMIZED)
# ========================
def fetch_sol_updates():
    """
    Fetch data terbaru untuk semua interval.
    Mengambil dari last_timestamp yang ada di BigQuery + 1.
    Jika tabel kosong, akan fetch dari awal (sejak listing).
    """
    result = {}
    for tf in interval_table_map:
        last_ts = get_last_timestamp(tf)
        
        if last_ts:
            # Jika sudah ada data, ambil dari timestamp terakhir + 1
            since = last_ts + 1
            print(f"📊 {tf}: Fetching updates since {datetime.utcfromtimestamp(since/1000)}")
        else:
            # Jika tabel kosong, fetch dari awal (None = dari awal listing)
            since = None
            print(f"📊 {tf}: Table empty, fetching historical data from beginning...")
        
        result[tf] = fetch_interval("SOL/USDT", tf, since)
    
    return result


# ========================
# 9. SINGLE RUN FOR GITHUB ACTIONS
# ========================
def run_single_update():
    """
    Fungsi utama untuk GitHub Actions.
    Dijalankan sekali setiap cron trigger (misal setiap 15 menit).
    
    LOGIKA:
    - Run pertama: Tabel kosong → fetch semua data historis dari awal listing
    - Run berikutnya: Ada data → fetch hanya data baru setelah timestamp terakhir
    """
    print(f"\n{'='*60}")
    print(f"🚀 SOL DATA FETCH - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"{'='*60}")
    
    # Cek status tabel sebelum fetch
    print("\n📋 STATUS TABEL BIGQUERY:")
    is_first_run = True
    for tf in interval_table_map:
        last_ts = get_last_timestamp(tf)
        if last_ts:
            is_first_run = False
            last_dt = datetime.utcfromtimestamp(last_ts / 1000)
            print(f"   {tf}: Last data = {last_dt}")
        else:
            print(f"   {tf}: KOSONG (akan fetch historis)")
    
    if is_first_run:
        print("\n⚠️ FIRST RUN DETECTED - Akan fetch SEMUA data historis!")
        print("   Ini mungkin memakan waktu beberapa menit...\n")
    else:
        print("\n✅ UPDATE MODE - Hanya fetch data baru\n")
    
    start_time = time.time()
    total_uploaded = 0
    upload_summary = {}
    
    # Fetch data untuk semua interval
    data = fetch_sol_updates()
    
    # Upload masing-masing interval
    for tf, df in data.items():
        if not df.empty:
            check_raw_data(df, tf)
            rows_fetched = len(df)
            uploaded = upload_df(df, tf)
            upload_summary[tf] = {"fetched": rows_fetched, "uploaded": uploaded}
            total_uploaded += uploaded
        else:
            print(f"⚠️ {tf}: No new data fetched")
            upload_summary[tf] = {"fetched": 0, "uploaded": 0}
    
    elapsed = time.time() - start_time
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 UPLOAD SUMMARY:")
    for tf, stats in upload_summary.items():
        status = "✅" if stats["uploaded"] > 0 else "⏸️"
        print(f"   {status} {tf}: {stats['uploaded']} uploaded (fetched: {stats['fetched']})")
    print(f"{'='*60}")
    print(f"✅ COMPLETED in {elapsed:.2f}s")
    print(f"📈 Total rows uploaded: {total_uploaded}")
    print(f"{'='*60}\n")
    
    # Verifikasi tidak ada duplikat (opsional, uncomment jika mau cek)
    # check_duplicates_in_bigquery()


# ========================
# RUN
# ========================
if __name__ == "__main__":
    # Single run untuk GitHub Actions (bukan loop)
    run_single_update()