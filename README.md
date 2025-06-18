# MLOps Serving & Monitoring

---
Proyek ini mendemonstrasikan implementasi sistem Machine Learning Operations (MLOps) yang lengkap, mencakup modeling dan tracking menggunakan MLFlow dan Dagshub, serving model ML melalui Docker dan Flask sebagai proxy API, serta monitoring dan alerting menggunakan Prometheus dan Grafana.

---
## Struktur Proyek

```
SMSML_Nama-siswa.zip
├── Eksperimen_SML_rtaufik27.txt
├── Membangun_model
    ├── modelling.py
    ├── modelling_tuning.py (skilled/advanced)
    ├── namadataset_preprocessing (bisa berupa file atau folder)
    ├── screenshoot_dashboard.jpg
    ├── screenshoot_artifak.jpg
    ├── requirements.txt
    ├── DagsHub.txt (berisikan tautan DagsHub jika menerapkan advanced)
├── Workflow-CI.txt
├── Monitoring dan Logging
    ├── 1.bukti_serving
    ├── 2.prometheus.yml
    ├── 3.prometheus_exporter.py
    ├── 4.bukti monitoring Prometheus (folder)
        └── 1.monitoring_<metriks>
        └── 2.monitoring_<metriks>
    ├── 5.bukti monitoring Grafana (folder)
        └── 1.monitoring_<metriks>
        └── 2.monitoring_<metriks>
    ├── 6.bukti alerting Grafana (folder)
        └── 1.rules_<metriks>
        └── 2.notifikasi_<metriks>
    ├── 7.Inference.py
```
---
## Komponen Proyek

* **Model ML:** Model Machine Learning yang telah dilatih (dalam kasus ini, untuk klasifikasi attrition).
* **MLflow:** Digunakan untuk melayani (serving) model ML.
* **Dagshub:** Sama seperti MLFLow, tetapi disimpan dalam cloud.
* **Docker:** Meng-*containerize* model server, memastikan lingkungan yang konsisten dan *portable*.
* **Python Flask:** Aplikasi web ringan yang berfungsi sebagai:
    * API *proxy* untuk menerima permintaan inferensi dan meneruskannya ke model server.
    * *Exporter* metrik Prometheus menggunakan `prometheus_client`.
* **Prometheus:** Sistem *monitoring* dan *alerting* yang bertugas mengumpulkan metrik kinerja aplikasi dari Flask.
* **Grafana:** Platform visualisasi data dan *dashboarding* yang terintegrasi dengan Prometheus untuk menampilkan metrik secara interaktif, serta mengelola dan mengirimkan notifikasi *alert* via email.

Cara Menyiapkan & Menjalankan Monitoring Model
Ikuti langkah-langkah di bawah ini. Disarankan untuk menggunakan terminal terpisah untuk setiap komponen.

1. Menjalankan Model Server (Docker Container)
Pastikan Docker Desktop (macOS/Windows) atau Docker Engine (Linux) Anda berjalan.
```
docker run -p 5005:8080 your_docker_username/attrition-model:latest
```
(Ganti your_docker_username/attrition-model:latest dengan nama image Docker model Anda.)
Anda akan melihat log dari model server MLflow yang menandakan bahwa model sedang dimuat dan siap menerima permintaan.

2. Menjalankan Aplikasi Flask (Prometheus Exporter)
Navigasikan ke direktori tempat prometheus_exporter.py berada dan jalankan:
```
python prometheus_exporter.py
```
Aplikasi Flask akan dimulai dan mendengarkan permintaan di http://127.0.0.1:8000. Endpoint metrik akan tersedia di http://127.0.0.1:8000/metrics.

3. Menjalankan Prometheus
Pastikan file konfigurasi prometheus.yml Anda telah diatur dengan benar untuk men-scrape metrik dari aplikasi Flask Anda, kemudian jalankan Prometheus:
```
/path/to/your/prometheus --config.file=prometheus.yml
```
(Ganti /path/to/your/prometheus dengan jalur biner Prometheus Anda.)
Prometheus UI akan tersedia di http://localhost:9090. Anda dapat memverifikasi status scrape di http://localhost:9090/targets.

4. Menjalankan Grafana
Pastikan file konfigurasi grafana.ini Anda telah dimodifikasi untuk mengaktifkan dan mengonfigurasi bagian [smtp] agar notifikasi email dapat dikirim. Pastikan enabled = true dan detail SMTP lainnya sudah benar.
```
# Bagian dari /opt/homebrew/etc/grafana/grafana.ini
[smtp]
enabled = true
host = smtp.gmail.com:587
user = your-email@gmail.com
password = """your app password"""  # Gunakan App Password jika dari Gmail/Outlook
from_address = your-email@gmail.com   # Sangat disarankan sama dengan 'user'
from_name = Grafana Alerts
skip_verify = false
starttls_policy = AlwaysStartTLS
```
Jalankan Grafana (jika menggunakan Homebrew di macOS dan menjalankannya secara foreground):
```
/opt/homebrew/opt/grafana/bin/grafana server --config /opt/homebrew/etc/grafana/grafana.ini --homepath /opt/homebrew/opt/grafana/share/grafana --packaging=brew cfg:default.paths.logs=/opt/homebrew/var/log/grafana cfg:default.paths.data=/opt/homebrew/var/lib/grafana cfg:default.paths.plugins=/opt/homebrew/var/lib/grafana/plugin
```
Grafana UI akan tersedia di http://localhost:3000.

5. Cara Melakukan Inferensi (Mengirim Permintaan)
Gunakan skrip inference.py untuk mengirim sejumlah permintaan ke API model Anda. Anda dapat menyesuaikan NUM_REQUESTS_TO_SEND dalam skrip ini untuk mensimulasikan beban yang berbeda.
```
python inference.py
```

6. Monitoring & Alerting
Setelah semua komponen berjalan dan Anda mengirim permintaan inferensi, Anda dapat memantau sistem:
- Prometheus UI: Kunjungi http://localhost:9090/graph untuk menjelajahi metrik mentah yang dikumpulkan.
- Grafana Dashboard: Akses http://localhost:3000. Tambahkan Prometheus sebagai sumber data (http://localhost:9090). 
Metrik-metrik yang bisa dilihat:
- Total Request per Detik (RPS): sum(rate(http_requests_total{job="ml_model_exporter"}[1m]))
- Latensi API Model (P95): histogram_quantile(0.95, sum by(le) (rate(http_request_duration_seconds_bucket{job="ml_model_exporter"}[5m])))
- Penggunaan CPU Proses Aplikasi: rate(process_cpu_seconds_total{job="ml_model_exporter"}[5m])
- Penggunaan RAM Proses Aplikasi (dalam MB): avg_over_time(process_resident_memory_bytes{job="ml_model_exporter"}[5m]) / (1024 * 1024)
- Kesehatan Target: up{job="ml_model_exporter"}
- Jumlah File Descriptors yang Dibuka: avg_over_time(process_open_fds{job="ml_model_exporter"}[5m])
- Jumlah Thread Proses: avg_over_time(process_threads{job="ml_model_exporter"}[5m])

Grafana Alerts: Konfigurasikan aturan alert di Grafana (misal, jika total request melebihi threshold tertentu, atau jika latensi memburuk) untuk menerima notifikasi melalui email. Pastikan contact point email sudah diuji dan berfungsi.