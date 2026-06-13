import logging
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DefectGeometricFeatures:
    area: float = 0.0
    perimeter: float = 0.0
    circularity: float = 0.0
    roundness: float = 0.0
    aspect_ratio: float = 0.0
    major_axis_length: float = 0.0
    minor_axis_length: float = 0.0
    solidity: float = 0.0
    extent: float = 0.0
    convex_area: float = 0.0
    fill_ratio: float = 0.0
    centroid_x: float = 0.0
    centroid_y: float = 0.0
    orientation: float = 0.0
    local_roughness_mean: float = 0.0
    local_roughness_std: float = 0.0
    local_roughness_max: float = 0.0
    coherence: float = 0.0
    anisotropy_ratio: float = 0.0
    line_length: float = 0.0
    avg_line_width: float = 0.0
    hu_moments: list = field(default_factory=lambda: [0.0] * 7)

    def to_vector(self) -> np.ndarray:
        scalar_features = [
            self.area,
            self.perimeter,
            self.circularity,
            self.roundness,
            self.aspect_ratio,
            self.major_axis_length,
            self.minor_axis_length,
            self.solidity,
            self.extent,
            self.convex_area,
            self.fill_ratio,
            self.orientation / 180.0,
            self.local_roughness_mean / 255.0,
            self.local_roughness_std / 255.0,
            self.coherence,
            self.anisotropy_ratio,
            self.line_length,
            self.avg_line_width,
        ]
        all_features = scalar_features + list(self.hu_moments)
        return np.array(all_features, dtype=np.float32)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hu_moments"] = list(self.hu_moments)
        return d


class GeometricFeatureExtractor:
    def __init__(self, min_area: int = 10, max_area: Optional[int] = None):
        self.min_area = min_area
        self.max_area = max_area

    def extract_from_mask(
        self,
        mask: np.ndarray,
        image: Optional[np.ndarray] = None,
    ) -> List[DefectGeometricFeatures]:
        import cv2
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        features_list = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            if self.max_area is not None and area > self.max_area:
                continue

            feat = self._compute_features(contour, image)
            features_list.append(feat)

        return features_list

    def _compute_features(
        self,
        contour: np.ndarray,
        image: Optional[np.ndarray] = None,
    ) -> DefectGeometricFeatures:
        import cv2

        feat = DefectGeometricFeatures()

        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        feat.area = float(area)
        feat.perimeter = float(perimeter)

        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            feat.circularity = float(min(circularity, 1.0))

        rect = cv2.minAreaRect(contour)
        (cx, cy), (w, h), angle = rect
        feat.centroid_x = float(cx)
        feat.centroid_y = float(cy)
        feat.orientation = float(angle)

        major = max(w, h)
        minor = min(w, h)
        feat.major_axis_length = float(major)
        feat.minor_axis_length = float(minor)
        feat.aspect_ratio = float(major / max(minor, 1e-6))

        if major > 0:
            r_min = minor / 2.0
            feat.roundness = float(min(4.0 * area / (np.pi * major * major), 1.0))

        hull = cv2.convexHull(contour)
        convex_area = cv2.contourArea(hull)
        feat.convex_area = float(convex_area)
        feat.solidity = float(area / max(convex_area, 1e-6))

        x, y, w_rect, h_rect = cv2.boundingRect(contour)
        feat.extent = float(area / max(w_rect * h_rect, 1e-6))

        feat.line_length = float(major)
        feat.avg_line_width = float(area / max(major, 1e-6))
        feat.fill_ratio = float(min(area / max(w_rect * h_rect, 1e-6), 1.0))

        from .tensor_ops import TensorMomentAnalyzer

        inertia = TensorMomentAnalyzer.compute_inertia_tensor(contour)
        feat.eccentricity = inertia.get("eccentricity", 0.0)
        feat.orientation = inertia.get("orientation_deg", feat.orientation)

        hu = TensorMomentAnalyzer.compute_hu_moments(contour)
        feat.hu_moments = list(hu)

        if image is not None:
            rough = TensorMomentAnalyzer.compute_local_roughness(image, contour)
            feat.local_roughness_mean = rough["mean_roughness"]
            feat.local_roughness_std = rough["std_roughness"]
            feat.local_roughness_max = rough["max_roughness"]

            from .tensor_ops import StructureTensorAnalyzer

            st = StructureTensorAnalyzer()
            struc = st.compute_region_coherence(image, contour)
            feat.coherence = struc["mean_coherence"]
            feat.anisotropy_ratio = struc["anisotropy_ratio"]

        return feat

    @staticmethod
    def is_line_scratch(
        feat: DefectGeometricFeatures,
        min_aspect_ratio: float = 5.0,
        min_length: float = 50.0,
        min_coherence: float = 0.5,
    ) -> bool:
        cond1 = feat.aspect_ratio >= min_aspect_ratio
        cond2 = feat.major_axis_length >= min_length
        cond3 = feat.coherence >= min_coherence
        return bool(cond1 and (cond2 or cond3))

    @staticmethod
    def is_particle_defect(
        feat: DefectGeometricFeatures,
        min_circularity: float = 0.6,
        max_aspect_ratio: float = 2.5,
        min_solidity: float = 0.7,
    ) -> bool:
        cond1 = feat.circularity >= min_circularity
        cond2 = feat.aspect_ratio <= max_aspect_ratio
        cond3 = feat.solidity >= min_solidity
        return bool(cond1 and cond2 and cond3)


__all__ = ["DefectGeometricFeatures", "GeometricFeatureExtractor"]
