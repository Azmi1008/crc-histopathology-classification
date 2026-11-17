# CRC Histopathology Classification — ResNet-50 Ensemble

Aplikasi demonstrasi penelitian untuk klasifikasi tingkat diferensiasi kanker kolorektal berdasarkan citra histopatologi menggunakan **ResNet-50 5-Fold Ensemble, Soft Voting, dan Ensemble Grad-CAM**.

Repository ini berisi source code aplikasi, dataset CRC-HGD-v1, manifest SHA-256, dan gambar sampel untuk pengujian.

## Fitur Aplikasi

- Upload citra histopatologi.
- Validasi gambar menggunakan SHA-256 terhadap manifest dataset.
- Menampilkan sampel gambar Grade I, Grade II, dan Grade III.
- Preprocessing gambar: RGB, Resize 224×224, ToTensor, dan normalisasi ImageNet.
- Menggunakan lima model ResNet-50 hasil Stratified 5-Fold Cross Validation.
- Soft Voting dengan rata-rata probabilitas dari lima model.
- Menampilkan prediksi kelas dan probabilitas masing-masing kelas.
- Visualisasi Ensemble Grad-CAM.

## Kelas Klasifikasi

| Label | Keterangan |
|---|---|
| Grade I | Well Differentiated |
| Grade II | Moderately Differentiated |
| Grade III | Poorly Differentiated |

## Hasil Eksperimen S3

Skenario S3 menggunakan Oversampling dan Data Augmentation dengan Stratified 5-Fold Cross Validation.

| Metrik | Hasil |
|---|---:|
| Accuracy | 90.53% |
| Macro Precision | 91.08% |
| Macro Recall | 90.00% |
| Macro F1-Score | 90.50% |

Hasil tersebut merupakan hasil evaluasi eksperimen penelitian, bukan jaminan performa pada data eksternal.

## Arsitektur Aplikasi

1. Pengguna mengunggah citra histopatologi.
2. Aplikasi memvalidasi SHA-256 gambar terhadap manifest dataset.
3. Gambar yang valid diproses dengan preprocessing 224×224 dan normalisasi ImageNet.
4. Lima model ResNet-50 menghasilkan probabilitas untuk tiga kelas.
5. Soft Voting menghitung rata-rata probabilitas kelima model.
6. Kelas dengan probabilitas tertinggi menjadi hasil prediksi.
7. Ensemble Grad-CAM menggabungkan visualisasi dari kelima model untuk kelas hasil prediksi.

## Struktur Repository

```text
crc-histopathology-classification/
├── app.py
├── model_utils.py
├── requirements.txt
├── README.md
├── .gitignore
├── dataset_hashes.json
├── generate_manifest.py
├── test_model_loading.py
├── dataset/
├── sample_images/
│   ├── Grade_I/
│   ├── Grade_II/
│   └── Grade_III/
└── .streamlit/
    └── config.toml
```

## Model Repository

Lima checkpoint ResNet-50 disimpan di Hugging Face Model Repository:

[Miikaelathen/crc-resnet50-ensemble](https://huggingface.co/Miikaelathen/crc-resnet50-ensemble)

Nama checkpoint:

```text
resnet50_oversampling_aug_30epoch_fold_1.pth
resnet50_oversampling_aug_30epoch_fold_2.pth
resnet50_oversampling_aug_30epoch_fold_3.pth
resnet50_oversampling_aug_30epoch_fold_4.pth
resnet50_oversampling_aug_30epoch_fold_5.pth
```

Aplikasi mengunduh checkpoint dari Hugging Face saat dibutuhkan. Repository model harus dapat diakses oleh aplikasi.

## Instalasi dan Menjalankan Aplikasi

### 1. Clone repository

```bash
git clone https://github.com/Azmi1008/crc-histopathology-classification.git
cd crc-histopathology-classification
```

### 2. Buat virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Jalankan aplikasi

```bash
streamlit run app.py
```

## Konfigurasi Hugging Face

Repository model default:

```text
Miikaelathen/crc-resnet50-ensemble
```

Aplikasi menggunakan repository tersebut sebagai konfigurasi default. Jika ingin menggantinya, atur environment variable `HF_REPO_ID`.

## Deployment

Aplikasi dirancang untuk dijalankan menggunakan Streamlit Community Cloud.

1. Hubungkan akun GitHub ke Streamlit Community Cloud.
2. Pilih repository `Azmi1008/crc-histopathology-classification`.
3. Pilih branch `main`.
4. Tentukan file utama `app.py`.
5. Deploy aplikasi.

Perhatikan kebutuhan memori dan waktu startup karena aplikasi menggunakan lima checkpoint model dengan ukuran total unduhan yang besar.

## Batasan Penelitian

- Aplikasi merupakan prototipe penelitian dan demonstrasi akademik, bukan alat diagnosis medis.
- Prediksi model tidak menggantikan pemeriksaan dan keputusan tenaga medis.
- Grad-CAM menunjukkan area yang berkontribusi terhadap respons model, bukan segmentasi jaringan atau bukti kausal.
- Performa pada citra eksternal, laboratorium lain, dan perangkat pemindaian berbeda belum dapat diasumsikan sama dengan hasil evaluasi internal.
- Validasi SHA-256 hanya menerima file yang byte-nya identik dengan gambar dalam manifest. Gambar yang diubah, dikompresi ulang, atau di-resize dapat ditolak.

## Lisensi

Hak penggunaan dan redistribusi dataset serta checkpoint mengikuti ketentuan lisensi dan pemilik masing-masing sumber. Penggunaan source code aplikasi tidak otomatis memberikan hak redistribusi terhadap dataset atau model.
