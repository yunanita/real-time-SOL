# Real-Time SOL Data Pipeline 📊

Analisis Real Time Prediksi Harga Solana - Otomatis fetch data menggunakan GitHub Actions.

## 🎯 Deskripsi

Project ini mengambil data historis dan real-time harga Solana (SOL/USDT) dari exchange KuCoin menggunakan library CCXT, lalu menyimpannya ke Google BigQuery. Proses berjalan otomatis setiap **15 menit** menggunakan GitHub Actions.

### Data yang dikumpulkan:

| Interval | Tabel BigQuery |
| -------- | -------------- |
| 1 menit  | SOL_1menit     |
| 15 menit | SOL_15menit    |
| 1 jam    | SOL_1jam       |
| 1 hari   | SOL_1hari      |
| 1 bulan  | SOL_1bulan     |

---

## 🚀 Cara Setup GitHub Actions

### 1. Fork/Clone Repository ini

### 2. Setup Google Cloud Credentials di GitHub Secrets

1. Buka repository di GitHub
2. Pergi ke **Settings** → **Secrets and variables** → **Actions**
3. Klik **New repository secret**
4. Isi:

   - **Name**: `GCP_CREDENTIALS`
   - **Secret**: Copy-paste **SELURUH ISI** file `time-series-analysis-480002-e7649b18ed82.json`

   ⚠️ **PENTING**: Copy seluruh isi JSON termasuk kurung kurawal `{ }`, seperti:

   ```json
   {
     "type": "service_account",
     "project_id": "time-series-analysis-480002",
     "private_key_id": "...",
     "private_key": "-----BEGIN PRIVATE KEY-----\n...",
     ...
   }
   ```

5. Klik **Add secret**

### 3. Enable GitHub Actions

1. Pergi ke tab **Actions** di repository
2. Klik **I understand my workflows, go ahead and enable them**
3. Workflow akan berjalan otomatis setiap 15 menit

### 4. (Opsional) Jalankan Manual

Untuk test atau menjalankan manual:

1. Pergi ke tab **Actions**
2. Pilih workflow **Fetch SOL Data**
3. Klik **Run workflow** → **Run workflow**

---

## 📁 Struktur File

```
├── .github/
│   └── workflows/
│       └── fetch-data.yml    # GitHub Actions workflow (cron setiap 15 menit)
├── fetch.py                   # Script utama untuk fetch data
├── requirements.txt           # Python dependencies
├── README.md                  # Dokumentasi ini
└── *.json                     # (JANGAN COMMIT!) Credentials file - simpan di GitHub Secrets
```

---

## ⚙️ Cara Kerja

1. **GitHub Actions** trigger workflow setiap 15 menit
2. Script membaca **credentials dari environment variable** `GCP_CREDENTIALS`
3. Untuk setiap interval (5m, 15m, 1h, 1d, 1M):
   - Cek timestamp terakhir di BigQuery
   - Fetch data baru dari exchange (setelah timestamp terakhir)
   - Upload data baru ke BigQuery (dengan deduplikasi)
4. Script selesai, tidak ada loop infinite

---

## 🔒 Keamanan

- **JANGAN** commit file JSON credentials ke repository
- Selalu gunakan **GitHub Secrets** untuk menyimpan credentials
- Tambahkan `*.json` ke `.gitignore`

---

## 🛠️ Development Lokal

Jika ingin menjalankan di lokal untuk testing:

```bash
# Install dependencies
pip install -r requirements.txt

# Jalankan script (pastikan file JSON ada di folder yang sama)
python fetch.py
```

Script akan otomatis mendeteksi file JSON lokal jika environment variable tidak ada.

---

## 📝 Catatan Penting

1. **GitHub Actions Free Tier**: 2000 menit/bulan untuk repository private, unlimited untuk public
2. **Cron tidak selalu tepat waktu**: GitHub Actions cron bisa delay 5-15 menit saat traffic tinggi
3. **Data tidak duplikat**: Script sudah handle deduplikasi berdasarkan timestamp

---

## 📊 BigQuery Setup

Pastikan sudah membuat dataset `SOL` di BigQuery dengan tabel-tabel:

- `SOL_1menit`
- `SOL_15menit`
- `SOL_1jam`
- `SOL_1hari`
- `SOL_1bulan`

Tabel akan otomatis terisi saat script pertama kali dijalankan.
