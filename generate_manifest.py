
import hashlib
import json
from pathlib import Path

# Lokasi folder proyek
BASE_DIR = Path(__file__).parent

# Dataset berada di dalam folder proyek
DATASET_DIR = BASE_DIR / "dataset"

# Hasil hash akan disimpan di folder yang sama dengan app.py
OUTPUT_FILE = BASE_DIR / "dataset_hashes.json"

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg",
    ".tif", ".tiff", ".bmp"
}

hashes = set()
total_images = 0

print("Memulai proses pembuatan manifest...")
print(f"Folder dataset: {DATASET_DIR}")

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Folder dataset tidak ditemukan: {DATASET_DIR}"
    )

for file_path in DATASET_DIR.rglob("*"):
    if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:

        with open(file_path, "rb") as file:
            file_hash = hashlib.sha256(file.read()).hexdigest()

        hashes.add(file_hash)
        total_images += 1

manifest = sorted(hashes)

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(manifest, file, indent=2)

print("\n" + "=" * 45)
print("PEMBUATAN MANIFEST SELESAI")
print("=" * 45)
print(f"Total file gambar ditemukan : {total_images}")
print(f"Total hash unik              : {len(manifest)}")
print(f"File manifest                : {OUTPUT_FILE}")
print("=" * 45)