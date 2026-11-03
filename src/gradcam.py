"""Grad-CAM implementation used for explainability figures and the live demo."""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Minimal Grad-CAM (Selvaraju et al., 2017) for a CNN classifier."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

    def _forward_hook(self, _module, _inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        """Return CAM in [0,1] at input resolution plus the predicted probabilities."""
        self.model.eval()
        x = x.clone().requires_grad_(True)
        handle_f = self.target_layer.register_forward_hook(self._forward_hook)
        handle_b = self.target_layer.register_full_backward_hook(self._backward_hook)
        try:
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)
            if class_idx is None:
                class_idx = int(probs.argmax(dim=1).item())
            self.model.zero_grad(set_to_none=True)
            logits[0, class_idx].backward(retain_graph=False)
            weights = self.gradients[0].mean(dim=(1, 2))  # [C]
            cam = torch.relu((weights[:, None, None] * self.activations[0]).sum(dim=0))
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            cam = F.interpolate(
                cam[None, None], size=x.shape[-2:], mode="bilinear", align_corners=False
            )[0, 0]
            return cam.cpu().numpy(), probs.detach().cpu().numpy()[0]
        finally:
            handle_f.remove()
            handle_b.remove()
            self.activations = None
            self.gradients = None


def cam_overlay(gray: np.ndarray, cam: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Blend a [0,1] CAM over a uint8 grayscale face; returns RGB uint8."""
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), colormap)
    overlay = cv2.addWeighted(base, 0.55, heat, 0.45, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
