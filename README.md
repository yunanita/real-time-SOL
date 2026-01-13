# 📈 Real-Time Solana (SOL) Price Data Pipeline

Automated data pipeline untuk mengumpulkan data harga Solana (SOL/USDT) secara real-time dari exchange KuCoin dan menyimpannya ke Google BigQuery.

## 🎯 Deskripsi Project

Project ini merupakan tugas UAS mata kuliah **Analisis Runtun Waktu** yang bertujuan untuk:

- Mengumpulkan data historis harga SOL dari awal listing (Agustus 2021) sampai sekarang
- Melakukan update data secara otomatis setiap 15 menit menggunakan GitHub Actions
- Menyimpan data ke Google BigQuery untuk analisis time series lebih lanjut

## 📊 Data yang Dikumpulkan

| Interval | Tabel BigQuery | Deskripsi                             |
| -------- | -------------- | ------------------------------------- |
| 1 menit  | `SOL_1menit`   | Data candle per menit (~2+ juta rows) |
| 15 menit | `SOL_15menit`  | Data candle per 15 menit              |
| 1 jam    | `SOL_1jam`     | Data candle per jam                   |
| 1 hari   | `SOL_1hari`    | Data candle harian                    |
| 1 bulan  | `SOL_1bulan`   | Data candle bulanan                   |

### Kolom Data (OHLCV)

- `timestamp` - Unix timestamp dalam milliseconds
- `datetime` - Tanggal dan waktu (UTC)
- `open` - Harga pembukaan
- `high` - Harga tertinggi
- `low` - Harga terendah
- `close` - Harga penutupan
- `volume` - Volume perdagangan

## ⚙️ Cara Kerja Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  RUN PERTAMA (Tabel Kosong)                                 │
│  ─────────────────────────────────────────────────────────  │
│  • Mendeteksi tabel BigQuery kosong                         │
│  • Fetch SEMUA data historis dari listing (Aug 2021)        │
│  • Upload jutaan rows ke BigQuery                           │
│  • Waktu: ~10-30 menit                                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  RUN SELANJUTNYA (Tabel Ada Data)                           │
│  ─────────────────────────────────────────────────────────  │
│  • Cek timestamp terakhir di BigQuery                       │
│  • Fetch hanya data BARU (incremental update)               │
│  • Upload dengan deduplikasi                                │
│  • Waktu: ~30 detik                                         │
└─────────────────────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

- **Data Source**: KuCoin Exchange via CCXT library
- **Storage**: Google BigQuery
- **Automation**: GitHub Actions (Cron setiap 15 menit)
- **Language**: Python 3.11

## 📁 Struktur Project

```
real-time-SOL/
├── .github/
│   └── workflows/
│       └── fetch-data.yml    # GitHub Actions workflow
├── fetch.py                   # Script utama pipeline
├── requirements.txt           # Python dependencies
├── README.md                  # Dokumentasi (file ini)
└── .gitignore                 # Ignore credentials
```

## 🔒 Keamanan

- Credentials Google Cloud disimpan sebagai GitHub Secrets
- File JSON credentials tidak di-commit ke repository

## 📈 BigQuery Schema

```sql
-- Project: time-series-analysis-480002
-- Dataset: SOL
-- Tables: SOL_1menit, SOL_15menit, SOL_1jam, SOL_1hari, SOL_1bulan

SELECT * FROM `time-series-analysis-480002.SOL.SOL_1menit`
ORDER BY timestamp DESC
LIMIT 100
```

## 👩‍🎓 Author

**Novia** - Tugas UAS Analisis Runtun Waktu (Semester 5)

## 📄 License

Project ini dibuat untuk keperluan akademis.
