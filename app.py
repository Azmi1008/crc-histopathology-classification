import os
import json
import random
import hashlib
from io import BytesIO
from pathlib import Path

import streamlit as st
import pandas as pd
from PIL import Image

from model_utils import (
    CLASS_NAMES,
    IMAGE_SIZE,
    predict_ensemble,
    make_ensemble_visualization,
)


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CRC-HGD Classification",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).parent

MANIFEST_PATH = BASE_DIR / "dataset_hashes.json"

SAMPLE_DIR = BASE_DIR / "sample_images"

HF_REPO_ID = os.getenv(
    "HF_REPO_ID",
    "Miikaelathen/crc-resnet50-ensemble"
)

SAMPLE_CLASSES = {
    "Grade I": SAMPLE_DIR / "Grade_I",
    "Grade II": SAMPLE_DIR / "Grade_II",
    "Grade III": SAMPLE_DIR / "Grade_III",
}

SAMPLE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg",
    ".tif", ".tiff", ".bmp"
}


# ============================================================
# 3. DATASET MANIFEST - SHA-256 VALIDATION
# ============================================================

@st.cache_data
def load_dataset_hashes(manifest_path):

    path = Path(manifest_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Manifest tidak ditemukan: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        hashes = json.load(file)

    if not isinstance(hashes, list) or not hashes:
        raise ValueError(
            "Format manifest tidak valid atau kosong."
        )

    for item in hashes:

        if (
            not isinstance(item, str)
            or len(item) != 64
            or any(
                c not in "0123456789abcdef"
                for c in item.lower()
            )
        ):
            raise ValueError(
                "Manifest mengandung hash SHA-256 yang tidak valid."
            )

    return set(hashes)


def calculate_sha256(image_bytes):

    return hashlib.sha256(image_bytes).hexdigest()


def validate_dataset_image(image_bytes, dataset_hashes):

    image_hash = calculate_sha256(image_bytes)

    is_registered = image_hash in dataset_hashes

    return is_registered, image_hash


try:

    DATASET_HASHES = load_dataset_hashes(
        str(MANIFEST_PATH)
    )

except Exception as exc:

    st.error("Gagal memuat manifest dataset.")
    st.exception(exc)
    st.stop()


# ============================================================
# 4. RANDOM DATASET SAMPLES
# ============================================================

def get_random_samples(dataset_hashes):

    samples = {}

    for class_name, class_dir in SAMPLE_CLASSES.items():

        if not class_dir.exists():

            samples[class_name] = []
            continue

        files = [
            file_path
            for file_path in class_dir.iterdir()
            if (
                file_path.is_file()
                and file_path.suffix.lower()
                in SAMPLE_EXTENSIONS
            )
        ]

        valid_files = []

        for file_path in files:

            try:

                file_bytes = file_path.read_bytes()

                file_hash = calculate_sha256(file_bytes)

                if file_hash in dataset_hashes:
                    valid_files.append(file_path)

            except Exception:
                continue

        selected = random.sample(
            valid_files,
            min(2, len(valid_files))
        )

        samples[class_name] = selected

    return samples


if "dataset_samples" not in st.session_state:

    st.session_state.dataset_samples = get_random_samples(
        DATASET_HASHES
    )


# ============================================================
# 5. SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🔬 CRC-HGD")

    st.caption(
        "Histopathology Classification System"
    )

    st.divider()

    st.subheader("Informasi Model")

    st.markdown(
        """
        **Arsitektur**

        ResNet-50

        **Strategi Training**

        Oversampling + Data Augmentation

        **Validasi**

        Stratified 5-Fold

        **Metode Ensemble**

        Soft Voting

        **Interpretabilitas**

        Ensemble Grad-CAM
        """
    )

    st.divider()

    st.subheader("Informasi Dataset")

    st.metric(
        "Jumlah Gambar Terdaftar",
        f"{len(DATASET_HASHES):,}"
    )

    st.caption("Dataset: CRC-HGD-v1")

    st.info(
        "Aplikasi hanya menerima gambar asli yang "
        "terdaftar dalam dataset penelitian."
    )

    st.divider()

    st.subheader("Preprocessing")

    st.write("Resize: 224 × 224 piksel")

    st.write(
        "Normalisasi: ImageNet Mean dan Standard Deviation"
    )

    st.divider()

    st.caption(
        "Academic Research Prototype"
    )


# ============================================================
# 6. MAIN HEADER
# ============================================================

st.title("Klasifikasi Histopatologi Kanker Kolorektal")

st.subheader(
    "CRC-HGD Classification System"
)

st.write(
    """
    Sistem klasifikasi tingkat diferensiasi kanker kolorektal
    berdasarkan citra histopatologi menggunakan ResNet-50
    dengan pendekatan 5-Fold Ensemble.
    """
)

st.warning(
    """
    Aplikasi ini merupakan prototipe penelitian akademik.
    Hasil klasifikasi bukan diagnosis medis dan tidak boleh
    digunakan sebagai dasar pengambilan keputusan klinis.
    """
)


# ============================================================
# 7. OVERVIEW METRICS
# ============================================================

st.subheader("Ringkasan Sistem")

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    st.metric("Jumlah Kelas", "3")

with metric2:
    st.metric("Model Ensemble", "5")

with metric3:
    st.metric("Resolusi Input", "224 × 224")

with metric4:
    st.metric(
        "Jumlah Dataset",
        f"{len(DATASET_HASHES):,}"
    )


st.divider()


# ============================================================
# 8. TABS
# ============================================================

tab_samples, tab_classification, tab_about = st.tabs(
    [
        "📁 Sampel Dataset",
        "🔬 Klasifikasi",
        "ℹ️ Informasi Model",
    ]
)


# ============================================================
# 9. TAB DATASET SAMPLES
# ============================================================

with tab_samples:

    st.header("Sampel Dataset CRC-HGD-v1")

    st.write(
        """
        Belum memiliki gambar dataset?
        Unduh sampel gambar asli berikut untuk mencoba
        fitur klasifikasi aplikasi.
        """
    )

    st.info(
        "Tersedia maksimal dua sampel acak dari setiap kelas. "
        "Setiap sampel telah diverifikasi menggunakan SHA-256."
    )

    if st.button(
        "🔄 Tampilkan Sampel Acak Lain",
        use_container_width=True
    ):

        st.session_state.dataset_samples = get_random_samples(
            DATASET_HASHES
        )

        st.rerun()

    for class_name, sample_files in (
        st.session_state.dataset_samples.items()
    ):

        st.subheader(class_name)

        if not sample_files:

            st.warning(
                f"Tidak tersedia sampel untuk {class_name}. "
                "Periksa folder sample_images."
            )

            continue

        columns = st.columns(2)

        for column, sample_path in zip(
            columns,
            sample_files
        ):

            with column:

                try:

                    sample_bytes = sample_path.read_bytes()

                    sample_image = Image.open(
                        BytesIO(sample_bytes)
                    ).convert("RGB")

                    with st.container(border=True):

                        st.image(
                            sample_image,
                            caption=sample_path.name,
                            use_container_width=True
                        )

                        st.download_button(
                            label="⬇ Unduh Sampel",
                            data=sample_bytes,
                            file_name=sample_path.name,
                            mime="application/octet-stream",
                            key=(
                                f"download_"
                                f"{class_name}_"
                                f"{sample_path.name}"
                            ),
                            use_container_width=True
                        )

                except Exception as exc:

                    st.error(
                        f"Gagal membaca sampel: {exc}"
                    )

    st.divider()

    st.caption(
        """
        Catatan: Sampel merupakan gambar asli dataset.
        """
    )


# ============================================================
# 10. TAB CLASSIFICATION WORKSPACE
# ============================================================

with tab_classification:

    st.header("Klasifikasi Citra Histopatologi")

    st.write(
        """
        Unggah gambar histopatologi yang terdaftar dalam
        dataset CRC-HGD-v1 untuk menjalankan klasifikasi.
        """
    )

    uploaded_file = st.file_uploader(
        "Unggah Gambar Histopatologi",
        type=[
            "png", "jpg", "jpeg",
            "tif", "tiff", "bmp"
        ],
        help=(
            "Hanya gambar asli yang terdaftar dalam "
            "dataset CRC-HGD-v1 yang dapat diklasifikasikan."
        ),
        key="classification_upload"
    )

    if uploaded_file is not None:

        image_bytes = uploaded_file.getvalue()

        # -----------------------------------------------
        # SHA-256 VALIDATION
        # -----------------------------------------------

        is_registered, image_hash = validate_dataset_image(
            image_bytes,
            DATASET_HASHES
        )

        st.subheader("Validasi Dataset")

        if not is_registered:

            st.error(
                "AKSES DITOLAK — Gambar tidak terdaftar "
                "dalam dataset CRC-HGD-v1."
            )

            st.warning(
                """
                Proses klasifikasi dibatalkan.
                Gunakan gambar asli dari dataset atau
                unduh sampel melalui menu Sampel Dataset.
                """
            )

            with st.expander("Detail Validasi"):

                st.write(
                    "Nama file:",
                    uploaded_file.name
                )

                st.write(
                    "Status:",
                    "Tidak Terdaftar"
                )

                st.code(image_hash)

        else:

            st.success(
                "VALIDASI BERHASIL — Gambar terdaftar "
                "dalam dataset CRC-HGD-v1."
            )

            with st.expander("Detail Validasi"):

                st.write(
                    "Nama file:",
                    uploaded_file.name
                )

                st.write(
                    "Status:",
                    "Terdaftar"
                )

                st.code(image_hash)

            # -------------------------------------------
            # OPEN ORIGINAL IMAGE
            # -------------------------------------------

            try:

                image = Image.open(
                    BytesIO(image_bytes)
                ).convert("RGB")

            except Exception as exc:

                st.error(
                    f"Gambar tidak dapat dibaca: {exc}"
                )

                image = None

            if image is not None:

                preview_col, action_col = st.columns(2)

                with preview_col:

                    st.subheader("Gambar Input")

                    st.image(
                        image,
                        caption=uploaded_file.name,
                        use_container_width=True
                    )

                    st.caption(
                        f"Ukuran asli: {image.width} × "
                        f"{image.height} piksel"
                    )

                with action_col:

                    st.subheader("Konfigurasi Inferensi")

                    st.markdown(
                        """
                        - **Model:** ResNet-50
                        - **Jumlah model:** 5
                        - **Ensemble:** Soft Voting
                        - **Visualisasi:** Ensemble Grad-CAM
                        - **Preprocessing:** 224 × 224
                        """
                    )

                    run_prediction = st.button(
                        "Jalankan Klasifikasi",
                        type="primary",
                        use_container_width=True
                    )

                # ---------------------------------------
                # ENSEMBLE INFERENCE
                # ---------------------------------------

                if run_prediction:

                    try:

                        with st.spinner(
                            """
                            Memuat lima model ResNet-50 dan
                            menjalankan Soft Voting serta
                            Ensemble Grad-CAM...
                            """
                        ):

                            result = predict_ensemble(
                                image,
                                HF_REPO_ID
                            )

                            overlay = make_ensemble_visualization(
                                image,
                                result["cam"]
                            )

                        predicted_class = result[
                            "predicted_class"
                        ]

                        predicted_name = CLASS_NAMES[
                            predicted_class
                        ]

                        confidence = (
                            result["confidence"] * 100
                        )

                        probabilities = result[
                            "probabilities"
                        ]

                        fold_probabilities = result[
                            "fold_probabilities"
                        ]

                        # --------------------------------
                        # RESULT HEADER
                        # --------------------------------

                        st.divider()

                        st.header("Hasil Klasifikasi")

                        result_col, confidence_col = st.columns(2)

                        with result_col:

                            with st.container(border=True):

                                st.caption(
                                    "PREDIKSI TINGKAT DIFERENSIASI"
                                )

                                st.subheader(
                                    predicted_name
                                )

                                st.write(
                                    "Hasil prediksi menggunakan "
                                    "5-Fold Soft Voting Ensemble."
                                )

                        with confidence_col:

                            with st.container(border=True):

                                st.metric(
                                    "Probabilitas Kelas Terpilih",
                                    f"{confidence:.2f}%"
                                )

                                st.progress(
                                    float(result["confidence"])
                                )

                        # --------------------------------
                        # CLASS PROBABILITIES
                        # --------------------------------

                        st.subheader(
                            "Probabilitas Setiap Kelas"
                        )

                        probability_columns = st.columns(3)

                        for index, class_name in enumerate(
                            CLASS_NAMES
                        ):

                            probability = float(
                                probabilities[index]
                            )

                            with probability_columns[index]:

                                with st.container(border=True):

                                    st.markdown(
                                        f"**{class_name}**"
                                    )

                                    st.metric(
                                        "Probabilitas",
                                        f"{probability * 100:.2f}%"
                                    )

                                    st.progress(probability)

                        # --------------------------------
                        # ENSEMBLE GRAD-CAM
                        # --------------------------------

                        st.divider()

                        st.header(
                            "Visualisasi Ensemble Grad-CAM"
                        )

                        st.write(
                            """
                            Visualisasi wilayah citra yang berkontribusi
                            terhadap respons model untuk kelas prediksi.
                            """
                        )

                        cam_col1, cam_col2 = st.columns(2)

                        with cam_col1:

                            st.subheader("Gambar Asli")

                            st.image(
                                image,
                                use_container_width=True
                            )

                        with cam_col2:

                            st.subheader("Grad-CAM Overlay")

                            st.image(
                                overlay,
                                use_container_width=True
                            )

                        st.info(
                            """
                            Grad-CAM merupakan visualisasi respons model.
                            Heatmap bukan segmentasi tumor dan tidak
                            membuktikan diagnosis medis.
                            """
                        )

                        # --------------------------------
                        # FOLD DETAILS
                        # --------------------------------

                        with st.expander(
                            "Lihat Probabilitas Lima Model"
                        ):

                            fold_data = {
                                "Fold": [
                                    f"Fold {i}"
                                    for i in range(1, 6)
                                ]
                            }

                            for class_index, class_name in enumerate(
                                CLASS_NAMES
                            ):

                                fold_data[class_name] = [
                                    float(
                                        fold_probs[class_index]
                                    ) * 100
                                    for fold_probs in fold_probabilities
                                ]

                            fold_df = pd.DataFrame(fold_data)

                            st.dataframe(
                                fold_df.style.format(
                                    {
                                        class_name: "{:.2f}%"
                                        for class_name in CLASS_NAMES
                                    }
                                ),
                                use_container_width=True,
                                hide_index=True
                            )

                        # --------------------------------
                        # INTERPRETATION NOTICE
                        # --------------------------------

                        st.warning(
                            """
                            Aplikasi ini merupakan prototipe penelitian
                            akademik. Hasil klasifikasi dan visualisasi
                            Grad-CAM tidak boleh digunakan sebagai
                            diagnosis medis atau dasar pengambilan
                            keputusan klinis.
                            """
                        )

                    except Exception as exc:

                        st.error(
                            "Terjadi kesalahan saat menjalankan inferensi."
                        )

                        st.exception(exc)


# ============================================================
# 11. TAB ABOUT MODEL
# ============================================================

with tab_about:

    st.header("Informasi Penelitian")

    st.subheader("Tujuan Penelitian")

    st.write(
        """
        Sistem ini dikembangkan untuk mengklasifikasikan
        tingkat diferensiasi kanker kolorektal berdasarkan
        citra histopatologi menggunakan pendekatan
        Deep Learning.
        """
    )

    st.subheader("Arsitektur Model")

    st.markdown(
        """
        - Backbone: ResNet-50
        - Transfer Learning: ImageNet pretrained weights
        - Jumlah kelas: 3
        - Validasi: Stratified 5-Fold
        - Metode Ensemble: Soft Voting
        """
    )

    st.subheader("Kategori Kelas")

    class_data = pd.DataFrame({
        "Kelas": [
            "Grade I",
            "Grade II",
            "Grade III"
        ],
        "Tingkat Diferensiasi": [
            "Well Differentiated",
            "Moderately Differentiated",
            "Poorly Differentiated"
        ]
    })

    st.table(class_data)

    st.subheader("Strategi Training")

    st.write(
        "Oversampling + Data Augmentation."
    )

    st.subheader("Interpretabilitas")

    st.write(
        """
        Ensemble Grad-CAM digunakan untuk memvisualisasikan
        area citra yang berkontribusi terhadap respons model.
        """
    )

    st.subheader("Dataset")

    st.write(
        "CRC-HGD-v1 Histopathology Image Dataset."
    )

    st.divider()

    st.subheader("Batasan Sistem")

    st.warning(
        """
        Aplikasi ini merupakan demonstrasi akademik.
        Model hanya diuji berdasarkan ruang lingkup penelitian
        dan tidak dimaksudkan sebagai sistem diagnosis klinis.
        """
    )

    st.caption(
        f"Hugging Face Model Repository: {HF_REPO_ID}"
    )


# ============================================================
# 12. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    **CRC-HGD Classification System**

    ResNet-50 · 5-Fold Ensemble · Soft Voting · Ensemble Grad-CAM

    Academic Research Prototype — Not for Clinical Diagnosis
    """
)