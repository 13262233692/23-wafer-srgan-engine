import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class EdgePostProcessor:
    def __init__(
        self,
        canny_threshold1: float = 50.0,
        canny_threshold2: float = 150.0,
        sobel_ksize: int = 3,
        morph_kernel_size: int = 3,
        morph_iterations: int = 2,
    ):
        self.canny_threshold1 = canny_threshold1
        self.canny_threshold2 = canny_threshold2
        self.sobel_ksize = sobel_ksize
        self.morph_kernel_size = morph_kernel_size
        self.morph_iterations = morph_iterations
        self._cv2 = None

    def _get_cv2(self):
        if self._cv2 is None:
            import cv2
            self._cv2 = cv2
        return self._cv2

    def detect_edges_canny(self, image: np.ndarray) -> np.ndarray:
        cv2 = self._get_cv2()
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)

        edges = cv2.Canny(gray, self.canny_threshold1, self.canny_threshold2)
        return edges

    def detect_edges_sobel(self, image: np.ndarray) -> np.ndarray:
        cv2 = self._get_cv2()
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)

        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.sobel_ksize)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.sobel_ksize)
        magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
        return magnitude

    def morphological_close(self, mask: np.ndarray) -> np.ndarray:
        cv2 = self._get_cv2()
        if mask.dtype != np.uint8:
            mask = np.clip(mask, 0, 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.morph_kernel_size, self.morph_kernel_size))
        result = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=self.morph_iterations)
        return result

    def morphological_open(self, mask: np.ndarray) -> np.ndarray:
        cv2 = self._get_cv2()
        if mask.dtype != np.uint8:
            mask = np.clip(mask, 0, 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.morph_kernel_size, self.morph_kernel_size))
        result = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=self.morph_iterations)
        return result

    def detect_defect_edges(self, image: np.ndarray, method: str = "canny") -> np.ndarray:
        if method == "canny":
            edges = self.detect_edges_canny(image)
        elif method == "sobel":
            edges = self.detect_edges_sobel(image)
        else:
            edges = self.detect_edges_canny(image)

        edges = self.morphological_close(edges)
        edges = self.morphological_open(edges)
        return edges

    def overlay_edges(self, image: np.ndarray, edges: np.ndarray, color: tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        cv2 = self._get_cv2()
        overlay = image.copy()
        if overlay.ndim == 2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
        if edges.ndim == 2:
            mask = edges > 0
            overlay[mask] = color
        return overlay
