import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class DefectType(Enum):
    UNKNOWN = "unknown"
    LINE_SCRATCH = "line_scratch"
    PARTICLE = "particle"
    BRIDGE = "bridge"
    VOID = "void"
    DISLOCATION = "dislocation"
    RESIDUE = "residue"
    OXIDE_PEEL = "oxide_peel"


@dataclass
class DefectInstance:
    defect_id: str = ""
    defect_type: DefectType = DefectType.UNKNOWN
    confidence: float = 0.0
    centroid_x: float = 0.0
    centroid_y: float = 0.0
    bbox: tuple = (0, 0, 0, 0)
    geometric_features: dict = field(default_factory=dict)
    spatial_attention_score: float = 0.0
    severity_level: int = 0
    layer: str = ""
    wafer_id: str = ""
    die_x: int = 0
    die_y: int = 0
    created_at: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["defect_type"] = self.defect_type.value
        d["bbox"] = list(self.bbox)
        return d

    def to_feature_vector(self) -> np.ndarray:
        geo_feat = self.geometric_features
        feature_keys = [
            "area", "perimeter", "circularity", "roundness",
            "aspect_ratio", "major_axis_length", "minor_axis_length",
            "solidity", "extent", "convex_area", "fill_ratio",
            "orientation", "local_roughness_mean", "local_roughness_std",
            "coherence", "anisotropy_ratio", "line_length", "avg_line_width",
        ]
        feat = []
        for k in feature_keys:
            feat.append(float(geo_feat.get(k, 0.0)))
        feat.extend(list(geo_feat.get("hu_moments", [0.0] * 7)))
        feat.append(self.severity_level / 10.0)
        feat.append(self.spatial_attention_score)
        return np.array(feat, dtype=np.float32)


class DefectClassifier:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self._load_thresholds()

    def _load_thresholds(self):
        self.scratch_config = self.config.get("scratch", {
            "min_aspect_ratio": 5.0,
            "min_length_px": 50.0,
            "min_coherence": 0.4,
            "max_circularity": 0.3,
            "min_roughness": 5.0,
        })

        self.particle_config = self.config.get("particle", {
            "min_circularity": 0.55,
            "max_aspect_ratio": 2.5,
            "min_solidity": 0.65,
            "max_roughness": 15.0,
        })

        self.oxide_peel_config = self.config.get("oxide_peel", {
            "min_circularity": 0.4,
            "max_circularity": 0.8,
            "min_solidity": 0.5,
            "max_solidity": 0.85,
            "max_roughness": 30.0,
            "irregular_shape_bonus": 0.2,
        })

    def classify(
        self,
        geometric_feat: dict,
        image: Optional[np.ndarray] = None,
    ) -> Tuple[DefectType, float]:
        scratch_score = self._score_scratch(geometric_feat)
        particle_score = self._score_particle(geometric_feat)
        oxide_peel_score = self._score_oxide_peel(geometric_feat)

        scores = {
            DefectType.LINE_SCRATCH: scratch_score,
            DefectType.PARTICLE: particle_score,
            DefectType.OXIDE_PEEL: oxide_peel_score,
        }

        best_type = DefectType.UNKNOWN
        best_score = 0.5
        for dtype, score in scores.items():
            if score > best_score:
                best_score = score
                best_type = dtype

        return best_type, float(best_score)

    def _score_scratch(self, feat: dict) -> float:
        cfg = self.scratch_config
        score = 0.0

        aspect_ratio = feat.get("aspect_ratio", 0.0)
        if aspect_ratio >= cfg["min_aspect_ratio"]:
            score += 0.35 * min(aspect_ratio / (cfg["min_aspect_ratio"] * 2), 1.0)

        length = feat.get("major_axis_length", 0.0)
        if length >= cfg["min_length_px"]:
            score += 0.2 * min(length / (cfg["min_length_px"] * 3), 1.0)

        coherence = feat.get("coherence", 0.0)
        if coherence >= cfg["min_coherence"]:
            score += 0.25

        circularity = feat.get("circularity", 0.0)
        if circularity <= cfg["max_circularity"]:
            score += 0.15

        roughness = feat.get("local_roughness_mean", 0.0)
        if roughness >= cfg["min_roughness"]:
            score += 0.05

        return min(score, 0.95)

    def _score_particle(self, feat: dict) -> float:
        cfg = self.particle_config
        score = 0.0

        circularity = feat.get("circularity", 0.0)
        if circularity >= cfg["min_circularity"]:
            score += 0.4 * circularity

        aspect_ratio = feat.get("aspect_ratio", 0.0)
        if aspect_ratio <= cfg["max_aspect_ratio"]:
            score += 0.25 * (1.0 - min(aspect_ratio / cfg["max_aspect_ratio"], 1.0))

        solidity = feat.get("solidity", 0.0)
        if solidity >= cfg["min_solidity"]:
            score += 0.25 * solidity

        roughness = feat.get("local_roughness_mean", 0.0)
        if roughness <= cfg["max_roughness"]:
            score += 0.1

        return min(score, 0.95)

    def _score_oxide_peel(self, feat: dict) -> float:
        cfg = self.oxide_peel_config
        score = 0.0

        circularity = feat.get("circularity", 0.0)
        if cfg["min_circularity"] <= circularity <= cfg["max_circularity"]:
            score += 0.3

        solidity = feat.get("solidity", 0.0)
        if cfg["min_solidity"] <= solidity <= cfg["max_solidity"]:
            score += 0.25

        aspect_ratio = feat.get("aspect_ratio", 0.0)
        if 1.0 <= aspect_ratio <= 3.0:
            score += 0.15

        roughness = feat.get("local_roughness_mean", 0.0)
        if cfg["max_roughness"] * 0.3 <= roughness <= cfg["max_roughness"]:
            score += 0.15

        extent = feat.get("extent", 0.0)
        if 0.2 <= extent <= 0.7:
            score += 0.15

        return min(score, 0.95)

    def classify_batch(
        self,
        features_list: List[dict],
        image: Optional[np.ndarray] = None,
    ) -> List[Tuple[DefectType, float]]:
        results = []
        for feat in features_list:
            dtype, conf = self.classify(feat, image)
            results.append((dtype, conf))
        return results


class SpatialAttentionAnalyzer:
    def __init__(self, image_shape: tuple, attention_radius: float = 100.0):
        self.image_shape = image_shape
        self.attention_radius = attention_radius

    def compute_spatial_attention(
        self,
        defects: List[DefectInstance],
        image: Optional[np.ndarray] = None,
    ) -> List[DefectInstance]:
        if len(defects) <= 1:
            for d in defects:
                d.spatial_attention_score = 0.5
            return defects

        positions = np.array([(d.centroid_x, d.centroid_y) for d in defects])
        n = len(defects)
        density_scores = np.zeros(n)

        for i in range(n):
            dists = np.sqrt(
                np.sum((positions - positions[i]) ** 2, axis=1)
            )
            neighbors = np.sum(dists < self.attention_radius) - 1
            density_scores[i] = neighbors / max(n - 1, 1)

        type_scores = np.array([
            0.9 if d.defect_type in (DefectType.LINE_SCRATCH, DefectType.OXIDE_PEEL) else 0.5
            for d in defects
        ])

        severity_scores = np.array([d.severity_level / 10.0 for d in defects])

        attention_scores = (
            0.4 * density_scores
            + 0.35 * type_scores
            + 0.25 * severity_scores
        )

        for i, d in enumerate(defects):
            d.spatial_attention_score = float(min(attention_scores[i], 1.0))

        return defects

    @staticmethod
    def estimate_severity(feat: dict, defect_type: DefectType) -> int:
        area = feat.get("area", 0.0)
        length = feat.get("major_axis_length", 0.0)
        roughness = feat.get("local_roughness_mean", 0.0)

        base_score = 0.0
        if defect_type == DefectType.LINE_SCRATCH:
            base_score = (
                0.4 * min(length / 200.0, 1.0)
                + 0.3 * min(area / 5000.0, 1.0)
                + 0.3 * min(roughness / 50.0, 1.0)
            )
        elif defect_type == DefectType.OXIDE_PEEL:
            base_score = (
                0.5 * min(area / 3000.0, 1.0)
                + 0.3 * min(roughness / 40.0, 1.0)
                + 0.2 * min(feat.get("aspect_ratio", 1.0) / 5.0, 1.0)
            )
        elif defect_type == DefectType.PARTICLE:
            base_score = 0.7 * min(area / 2000.0, 1.0) + 0.3 * min(roughness / 30.0, 1.0)
        else:
            base_score = 0.3 * min(area / 1000.0, 1.0)

        level = int(base_score * 10) + 1
        return max(1, min(10, level))


__all__ = [
    "DefectType",
    "DefectInstance",
    "DefectClassifier",
    "SpatialAttentionAnalyzer",
]
