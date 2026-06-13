import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


class StructureTensorAnalyzer:
    def __init__(
        self,
        sobel_ksize: int = 3,
        gaussian_sigma: float = 2.0,
        window_size: int = 15,
    ):
        self.sobel_ksize = sobel_ksize
        self.gaussian_sigma = gaussian_sigma
        self.window_size = window_size

    def _compute_gradient(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        import cv2
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        if gray.dtype != np.float32:
            gray = gray.astype(np.float32)
        if gray.max() > 1.0 and gray.max() <= 255.0:
            gray = gray / 255.0

        Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.sobel_ksize)
        Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.sobel_ksize)

        return Ix, Iy

    def compute_structure_tensor(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        import cv2
        Ix, Iy = self._compute_gradient(image)

        Ixx = Ix * Ix
        Ixy = Ix * Iy
        Iyy = Iy * Iy

        ksize = int(2 * np.ceil(3 * self.gaussian_sigma) + 1)
        if ksize % 2 == 0:
            ksize += 1
        Sxx = cv2.GaussianBlur(Ixx, (ksize, ksize), self.gaussian_sigma)
        Sxy = cv2.GaussianBlur(Ixy, (ksize, ksize), self.gaussian_sigma)
        Syy = cv2.GaussianBlur(Iyy, (ksize, ksize), self.gaussian_sigma)

        return Sxx, Sxy, Syy

    def compute_eigenvalues(
        self,
        Sxx: np.ndarray,
        Sxy: np.ndarray,
        Syy: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        trace = Sxx + Syy
        det = Sxx * Syy - Sxy * Sxy

        discriminant = np.maximum(trace * trace / 4.0 - det, 0.0)
        sqrt_disc = np.sqrt(discriminant)

        lambda1 = trace / 2.0 + sqrt_disc
        lambda2 = trace / 2.0 - sqrt_disc

        return lambda1, lambda2

    def compute_coherence(self, lambda1: np.ndarray, lambda2: np.ndarray) -> np.ndarray:
        denom = lambda1 + lambda2 + 1e-8
        coherence = ((lambda1 - lambda2) / denom) ** 2
        return np.clip(coherence, 0.0, 1.0)

    def compute_orientation(self, Sxx: np.ndarray, Sxy: np.ndarray, Syy: np.ndarray) -> np.ndarray:
        denominator = Sxx - Syy + 1e-8
        theta = 0.5 * np.arctan2(2 * Sxy, denominator)
        return theta

    def compute_structure_features(
        self,
        image: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> dict:
        Sxx, Sxy, Syy = self.compute_structure_tensor(image)
        lambda1, lambda2 = self.compute_eigenvalues(Sxx, Sxy, Syy)
        coherence = self.compute_coherence(lambda1, lambda2)
        orientation = self.compute_orientation(Sxx, Sxy, Syy)

        result = {
            "Sxx": Sxx,
            "Sxy": Sxy,
            "Syy": Syy,
            "lambda1": lambda1,
            "lambda2": lambda2,
            "coherence": coherence,
            "orientation": orientation,
        }

        if mask is not None and mask.shape == image.shape[:2]:
            mask_float = mask.astype(bool)
            if mask_float.sum() > 0:
                result["mean_coherence"] = float(coherence[mask_float].mean())
                result["mean_anisotropy"] = float(
                    (lambda1[mask_float] - lambda2[mask_float]).mean()
                )
                result["energy"] = float((lambda1[mask_float] + lambda2[mask_float]).mean())
            else:
                result["mean_coherence"] = 0.0
                result["mean_anisotropy"] = 0.0
                result["energy"] = 0.0

        return result

    def compute_region_coherence(
        self,
        image: np.ndarray,
        contour: np.ndarray,
    ) -> dict:
        import cv2
        x, y, w, h = cv2.boundingRect(contour)
        x = max(0, x - 2)
        y = max(0, y - 2)
        w = min(w + 4, image.shape[1] - x)
        h = min(h + 4, image.shape[0] - y)

        roi = image[y : y + h, x : x + w]
        if roi.size == 0:
            return {
                "mean_coherence": 0.0,
                "mean_orientation": 0.0,
                "anisotropy_ratio": 0.0,
            }

        Sxx, Sxy, Syy = self.compute_structure_tensor(roi)
        lambda1, lambda2 = self.compute_eigenvalues(Sxx, Sxy, Syy)
        coherence = self.compute_coherence(lambda1, lambda2)
        orientation = self.compute_orientation(Sxx, Sxy, Syy)

        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        shifted_contour = contour - np.array([x, y]).reshape(1, -1)
        cv2.drawContours(mask, [shifted_contour], -1, 1, -1)
        mask_float = mask.astype(bool)

        if mask_float.sum() > 0:
            mean_coh = float(coherence[mask_float].mean())
            mean_ori = float(np.degrees(orientation[mask_float].mean()))
            ani_ratio = float(
                (lambda1[mask_float].sum()) / (lambda2[mask_float].sum() + 1e-8)
            )
        else:
            mean_coh = 0.0
            mean_ori = 0.0
            ani_ratio = 0.0

        return {
            "mean_coherence": mean_coh,
            "mean_orientation": mean_ori,
            "anisotropy_ratio": ani_ratio,
        }


class TensorMomentAnalyzer:
    def __init__(self):
        pass

    @staticmethod
    def compute_central_moments(contour: np.ndarray) -> dict:
        import cv2
        moments = cv2.moments(contour)
        return moments

    @staticmethod
    def compute_hu_moments(contour: np.ndarray) -> np.ndarray:
        import cv2
        moments = cv2.moments(contour)
        hu = cv2.HuMoments(moments).flatten()
        hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
        return hu_log

    @staticmethod
    def compute_inertia_tensor(contour: np.ndarray) -> dict:
        import cv2
        moments = cv2.moments(contour)

        m00 = moments["m00"] + 1e-8
        cx = moments["m10"] / m00
        cy = moments["m01"] / m00

        mu20 = moments["mu20"]
        mu02 = moments["mu02"]
        mu11 = moments["mu11"]

        trace = mu20 + mu02
        det = mu20 * mu02 - mu11 * mu11

        sqrt_term = np.sqrt(max(trace * trace / 4.0 - det, 0.0))
        lambda1 = trace / 2.0 + sqrt_term
        lambda2 = trace / 2.0 - sqrt_term

        orientation = 0.5 * np.arctan2(2 * mu11, mu20 - mu02 + 1e-8)

        return {
            "centroid_x": float(cx),
            "centroid_y": float(cy),
            "lambda1": float(lambda1),
            "lambda2": float(lambda2),
            "eccentricity": float(np.sqrt(1.0 - lambda2 / (lambda1 + 1e-8))),
            "orientation_rad": float(orientation),
            "orientation_deg": float(np.degrees(orientation)),
            "aspect_ratio": float(np.sqrt(lambda1 / (lambda2 + 1e-8))),
        }

    @staticmethod
    def compute_local_roughness(
        image: np.ndarray,
        contour: np.ndarray,
        kernel_size: int = 5,
    ) -> dict:
        import cv2
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            gray = image.astype(np.float32)

        x, y, w, h = cv2.boundingRect(contour)
        pad = kernel_size // 2
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(w + pad * 2, gray.shape[1] - x)
        h = min(h + pad * 2, gray.shape[0] - y)

        roi = gray[y : y + h, x : x + w]
        if roi.size == 0:
            return {"mean_roughness": 0.0, "std_roughness": 0.0, "max_roughness": 0.0}

        mask = np.zeros((h, w), dtype=np.uint8)
        shifted = contour - np.array([x, y], dtype=np.float32).reshape(1, -1)
        cv2.drawContours(mask, [shifted.astype(np.int32)], -1, 1, -1)

        blurred = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
        roughness_map = np.abs(roi - blurred)

        mask_float = mask.astype(bool)
        if mask_float.sum() > 0:
            mean_rough = float(roughness_map[mask_float].mean())
            std_rough = float(roughness_map[mask_float].std())
            max_rough = float(roughness_map[mask_float].max())
        else:
            mean_rough = 0.0
            std_rough = 0.0
            max_rough = 0.0

        return {
            "mean_roughness": mean_rough,
            "std_roughness": std_rough,
            "max_roughness": max_rough,
        }


__all__ = ["StructureTensorAnalyzer", "TensorMomentAnalyzer"]
