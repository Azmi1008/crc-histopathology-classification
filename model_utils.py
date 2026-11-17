from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import models, transforms


IMAGE_SIZE = (224, 224)
NUM_CLASSES = 3
NUM_FOLDS = 5
CLASS_NAMES = ["Grade I", "Grade II", "Grade III"]
CHECKPOINT_TEMPLATE = "resnet50_oversampling_aug_30epoch_fold_{fold}.pth"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

INFERENCE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model():
    """Buat ResNet-50 dengan tiga output kelas, sama seperti eksperimen."""
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


@lru_cache(maxsize=5)
def load_fold_model(repo_id: str, fold: int):
    """Unduh dan muat satu checkpoint dari Hugging Face."""
    if fold not in range(1, NUM_FOLDS + 1):
        raise ValueError(f"Nomor fold tidak valid: {fold}")

    filename = CHECKPOINT_TEMPLATE.format(fold=fold)
    checkpoint_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="model",
    )

    device = get_device()
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        # Kompatibilitas untuk versi PyTorch lama.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"{filename} tidak memiliki key 'model_state_dict'."
        )

    if checkpoint.get("num_classes") != NUM_CLASSES:
        raise ValueError(
            f"{filename}: jumlah kelas checkpoint tidak sesuai."
        )

    expected_classes = CLASS_NAMES
    if checkpoint.get("classes") != expected_classes:
        raise ValueError(
            f"{filename}: urutan kelas tidak sesuai. "
            f"Ditemukan: {checkpoint.get('classes')}"
        )

    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


class GradCAM:
    """Grad-CAM untuk target layer model.layer4[-1]."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, gradient):
        self.gradients = gradient

    def generate(self, input_tensor, target_class):
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None

        x = input_tensor.detach().clone().requires_grad_(True)

        with torch.enable_grad():
            logits = self.model(x)
            score = logits[:, target_class].sum()
            score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Aktivasi atau gradien Grad-CAM tidak tersedia.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=IMAGE_SIZE,
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0].detach().cpu().numpy()
        return normalize_cam(cam)

    def close(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def normalize_cam(cam):
    cam = np.maximum(cam, 0)
    minimum = float(cam.min())
    maximum = float(cam.max())
    if maximum - minimum < 1e-8:
        return np.zeros_like(cam, dtype=np.float32)
    return ((cam - minimum) / (maximum - minimum)).astype(np.float32)


def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    resized = image.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]))
    tensor = INFERENCE_TRANSFORM(image).unsqueeze(0)
    return resized, tensor


def predict_ensemble(image: Image.Image, repo_id: str):
    """
    Prediksi dengan rata-rata probabilitas lima fold (Soft Voting),
    lalu Ensemble Grad-CAM untuk kelas hasil prediksi ensemble.
    """
    device = get_device()
    resized_image, input_tensor = preprocess_image(image)
    input_tensor = input_tensor.to(device)

    models_list = [
        load_fold_model(repo_id, fold)
        for fold in range(1, NUM_FOLDS + 1)
    ]

    fold_probabilities = []
    with torch.no_grad():
        for model in models_list:
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
            fold_probabilities.append(
                probabilities[0].detach().cpu().numpy()
            )

    fold_probabilities = np.stack(fold_probabilities, axis=0)
    ensemble_probabilities = fold_probabilities.mean(axis=0)
    predicted_class = int(np.argmax(ensemble_probabilities))
    confidence = float(ensemble_probabilities[predicted_class])

    # Grad-CAM untuk kelas hasil Soft Voting pada kelima model.
    cams = []
    gradcam_objects = []
    try:
        for model in models_list:
            gradcam = GradCAM(model, model.layer4[-1])
            gradcam_objects.append(gradcam)
            cam = gradcam.generate(input_tensor, predicted_class)
            cams.append(normalize_cam(cam))
    finally:
        for gradcam in gradcam_objects:
            gradcam.close()

    ensemble_cam = normalize_cam(np.mean(np.stack(cams, axis=0), axis=0))

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": ensemble_probabilities,
        "fold_probabilities": fold_probabilities,
        "cam": ensemble_cam,
        "resized_image": resized_image,
    }


def make_ensemble_visualization(image: Image.Image, cam: np.ndarray):
    """Buat heatmap dan overlay pada citra berukuran 224×224."""
    rgb = np.asarray(image.convert("RGB").resize(IMAGE_SIZE))
    cam = cv2.resize(
        normalize_cam(cam),
        (IMAGE_SIZE[1], IMAGE_SIZE[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    heatmap_bgr = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET,
    )
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(rgb, 0.55, heatmap_rgb, 0.45, 0)
    return overlay
