import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .classifier import (
    DefectInstance,
    DefectType,
    DefectClassifier,
    SpatialAttentionAnalyzer,
)
from .geometric_features import DefectGeometricFeatures, GeometricFeatureExtractor
from .segmentor import DefectSegmentor
from .tensor_ops import StructureTensorAnalyzer, TensorMomentAnalyzer
from .vector_store import MilvusVectorStore

logger = logging.getLogger(__name__)


@dataclass
class DefectAnalysisResult:
    image_shape: tuple = (0, 0)
    num_defects: int = 0
    num_scratches: int = 0
    num_particles: int = 0
    num_oxide_peels: int = 0
    defects: List[dict] = field(default_factory=list)
    spatial_attention_map: Optional[np.ndarray] = None
    processing_time: float = 0.0
    stored_in_milvus: int = 0

    def to_dict(self) -> dict:
        return {
            "image_shape": list(self.image_shape),
            "num_defects": self.num_defects,
            "num_scratches": self.num_scratches,
            "num_particles": self.num_particles,
            "num_oxide_peels": self.num_oxide_peels,
            "defects": self.defects,
            "processing_time": self.processing_time,
            "stored_in_milvus": self.stored_in_milvus,
        }


class SpatialAttentionPipeline:
    def __init__(
        self,
        config: Optional[dict] = None,
        segmentor: Optional[DefectSegmentor] = None,
        classifier: Optional[DefectClassifier] = None,
        vector_store: Optional[MilvusVectorStore] = None,
    ):
        self.config = config or {}
        self.analysis_config = self.config.get("defect_analysis", {}) if "defect_analysis" in self.config else self.config

        self.enable_segmentation = self.analysis_config.get("enable_segmentation", True)
        self.enable_tensor_analysis = self.analysis_config.get("enable_tensor_analysis", True)
        self.enable_spatial_attention = self.analysis_config.get("enable_spatial_attention", True)
        self.enable_vector_storage = self.analysis_config.get("enable_vector_storage", True)

        self.min_defect_area = self.analysis_config.get("min_defect_area", 16)
        self.attention_radius = self.analysis_config.get("attention_radius", 100.0)
        self.store_only_significant = self.analysis_config.get("store_only_significant", True)
        self.significance_threshold = self.analysis_config.get("significance_threshold", 0.6)

        self.segmentor = segmentor or self._init_segmentor()
        self.classifier = classifier or DefectClassifier(self.analysis_config.get("classifier", {}))
        self.feature_extractor = GeometricFeatureExtractor(min_area=self.min_defect_area)
        self.structure_tensor = StructureTensorAnalyzer()
        self.spatial_analyzer = SpatialAttentionAnalyzer((0, 0), attention_radius=self.attention_radius)

        if self.enable_vector_storage:
            self.vector_store = vector_store or self._init_vector_store()
        else:
            self.vector_store = None

    def _init_segmentor(self) -> DefectSegmentor:
        seg_cfg = self.analysis_config.get("segmentor", {})
        segmentor = DefectSegmentor(
            num_classes=seg_cfg.get("num_classes", 2),
            base_filters=seg_cfg.get("base_filters", 16),
            threshold=seg_cfg.get("threshold", 0.5),
            min_defect_area=seg_cfg.get("min_defect_area", 16),
        )
        checkpoint = seg_cfg.get("checkpoint_path", "")
        if checkpoint:
            try:
                segmentor.load_weights(checkpoint)
            except Exception as e:
                logger.warning(f"Failed to load segmentor weights: {e}, using random init")
        return segmentor

    def _init_vector_store(self) -> MilvusVectorStore:
        vs_cfg = self.analysis_config.get("vector_store", {})
        return MilvusVectorStore(
            collection_name=vs_cfg.get("collection_name", "wafer_defects"),
            host=vs_cfg.get("host", "localhost"),
            port=vs_cfg.get("port", 19530),
            vector_dim=vs_cfg.get("vector_dim", 26),
            auto_create=vs_cfg.get("auto_create", True),
            use_mock_if_unavailable=vs_cfg.get("use_mock_if_unavailable", True),
        )

    def analyze(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        wafer_id: str = "",
        layer: str = "",
        die_x: int = 0,
        die_y: int = 0,
    ) -> DefectAnalysisResult:
        t_start = time.time()

        if mask is None and self.enable_segmentation:
            mask = self.segmentor.segment_tiled(image)
            mask = self.segmentor.cleanup_small_regions(mask, self.min_defect_area)
        elif mask is None:
            raise ValueError("Segmentation is disabled but no mask provided")

        h, w = image.shape[:2]

        geo_features = self.feature_extractor.extract_from_mask(mask, image)

        defect_instances = []
        for i, gf in enumerate(geo_features):
            feat_dict = gf.to_dict()

            defect_type, confidence = self.classifier.classify(feat_dict, image)

            bbox = self._extract_bbox(feat_dict, (h, w))

            severity = SpatialAttentionAnalyzer.estimate_severity(feat_dict, defect_type)

            defect = DefectInstance(
                defect_id=f"{wafer_id}_{die_x}_{die_y}_{i}_{int(time.time()*1000)}",
                defect_type=defect_type,
                confidence=confidence,
                centroid_x=gf.centroid_x,
                centroid_y=gf.centroid_y,
                bbox=bbox,
                geometric_features=feat_dict,
                severity_level=severity,
                layer=layer,
                wafer_id=wafer_id,
                die_x=die_x,
                die_y=die_y,
                created_at=time.time(),
            )
            defect_instances.append(defect)

        if self.enable_spatial_attention and len(defect_instances) > 0:
            self.spatial_analyzer.image_shape = (h, w)
            self.spatial_analyzer.attention_radius = self.attention_radius
            defect_instances = self.spatial_analyzer.compute_spatial_attention(defect_instances, image)

        stored_count = 0
        if self.enable_vector_storage and self.vector_store is not None:
            if self.store_only_significant:
                to_store = [
                    d for d in defect_instances
                    if d.confidence >= self.significance_threshold
                    and d.defect_type in (DefectType.LINE_SCRATCH, DefectType.OXIDE_PEEL, DefectType.PARTICLE)
                ]
            else:
                to_store = defect_instances

            if to_store:
                self.vector_store.insert_batch(to_store, wafer_id, layer, die_x, die_y)
                stored_count = len(to_store)

        num_scratches = sum(1 for d in defect_instances if d.defect_type == DefectType.LINE_SCRATCH)
        num_particles = sum(1 for d in defect_instances if d.defect_type == DefectType.PARTICLE)
        num_oxide_peels = sum(1 for d in defect_instances if d.defect_type == DefectType.OXIDE_PEEL)

        result = DefectAnalysisResult(
            image_shape=(h, w),
            num_defects=len(defect_instances),
            num_scratches=num_scratches,
            num_particles=num_particles,
            num_oxide_peels=num_oxide_peels,
            defects=[d.to_dict() for d in defect_instances],
            processing_time=time.time() - t_start,
            stored_in_milvus=stored_count,
        )

        return result

    def analyze_batch(
        self,
        images: List[np.ndarray],
        wafer_id: str = "",
        layer: str = "",
    ) -> List[DefectAnalysisResult]:
        results = []
        for i, img in enumerate(images):
            result = self.analyze(
                img,
                wafer_id=wafer_id,
                layer=layer,
                die_x=i,
                die_y=0,
            )
            results.append(result)
        return results

    @staticmethod
    def _extract_bbox(feat_dict: dict, img_shape: tuple) -> tuple:
        h, w = img_shape
        cx = feat_dict.get("centroid_x", w / 2)
        cy = feat_dict.get("centroid_y", h / 2)
        major = feat_dict.get("major_axis_length", 10)
        minor = feat_dict.get("minor_axis_length", 10)

        half_w = max(major, minor) / 2 + 5
        half_h = min(major, minor) / 2 + 5

        x1 = max(0, int(cx - half_w))
        y1 = max(0, int(cy - half_h))
        x2 = min(w, int(cx + half_w))
        y2 = min(h, int(cy + half_h))

        return (x1, y1, x2, y2)

    def get_vector_store_stats(self) -> dict:
        if self.vector_store is not None:
            return self.vector_store.get_stats()
        return {"enabled": False}

    def search_similar_defects(
        self,
        defect_instance,
        top_k: int = 10,
    ) -> List[dict]:
        if self.vector_store is None:
            return []
        vector = defect_instance.to_feature_vector()
        return self.vector_store.search(vector, top_k=top_k)

    def close(self):
        if self.vector_store is not None:
            self.vector_store.close()


__all__ = ["SpatialAttentionPipeline", "DefectAnalysisResult"]
