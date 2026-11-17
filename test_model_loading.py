"""
Pemeriksaan opsional: memverifikasi kelima checkpoint pada repository HF.
Jalankan setelah mengatur HF_REPO_ID.
"""
import os
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from torchvision import models

REPO_ID = os.getenv("HF_REPO_ID", "USERNAME/crc-resnet50-ensemble")
NUM_CLASSES = 3
CLASS_NAMES = ["Grade I", "Grade II", "Grade III"]


def build_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


if REPO_ID.startswith("USERNAME/"):
    raise ValueError("Atur HF_REPO_ID terlebih dahulu.")

for fold in range(1, 6):
    filename = f"resnet50_oversampling_aug_30epoch_fold_{fold}.pth"
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="model",
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    assert checkpoint["num_classes"] == NUM_CLASSES
    assert checkpoint["classes"] == CLASS_NAMES

    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    with torch.no_grad():
        output = model(torch.randn(1, 3, 224, 224))

    assert tuple(output.shape) == (1, 3)
    print(f"Fold {fold}: OK — output {tuple(output.shape)}")

print("Semua checkpoint berhasil diverifikasi.")
