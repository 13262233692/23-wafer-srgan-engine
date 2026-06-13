import logging
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DefectVectorRecord:
    id: str
    vector: np.ndarray
    metadata: dict

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "vector": self.vector.tolist(),
            "metadata": self.metadata,
        }


class MilvusVectorStore:
    VECTOR_DIM = 26

    def __init__(
        self,
        collection_name: str = "wafer_defects",
        host: str = "localhost",
        port: int = 19530,
        vector_dim: int = 26,
        auto_create: bool = True,
        use_mock_if_unavailable: bool = True,
    ):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.vector_dim = vector_dim
        self.auto_create = auto_create
        self.use_mock_if_unavailable = use_mock_if_unavailable

        self._client = None
        self._mock_storage: Dict[str, DefectVectorRecord] = {}
        self._connected = False

        self._init_client()

    def _init_client(self):
        try:
            from pymilvus import (
                connections,
                utility,
                Collection,
                CollectionSchema,
                FieldSchema,
                DataType,
            )
            self._pymilvus_available = True
        except ImportError:
            self._pymilvus_available = False
            if self.use_mock_if_unavailable:
                logger.warning("pymilvus not installed, using in-memory mock storage")
                self._mock_storage = {}
                self._connected = True
                return
            raise

        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
            )
            self._connected = True
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")

            if self.auto_create and not utility.has_collection(self.collection_name):
                self._create_collection()
        except Exception as e:
            logger.warning(f"Failed to connect to Milvus: {e}")
            if self.use_mock_if_unavailable:
                logger.warning("Falling back to in-memory mock storage")
                self._mock_storage = {}
                self._connected = True
            else:
                raise

    def _create_collection(self):
        from pymilvus import (
            Collection,
            CollectionSchema,
            FieldSchema,
            DataType,
            utility,
        )

        if utility.has_collection(self.collection_name):
            return

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim),
            FieldSchema(name="defect_type", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="confidence", dtype=DataType.FLOAT),
            FieldSchema(name="centroid_x", dtype=DataType.FLOAT),
            FieldSchema(name="centroid_y", dtype=DataType.FLOAT),
            FieldSchema(name="wafer_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="layer", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="die_x", dtype=DataType.INT32),
            FieldSchema(name="die_y", dtype=DataType.INT32),
            FieldSchema(name="severity_level", dtype=DataType.INT32),
            FieldSchema(name="created_at", dtype=DataType.FLOAT),
        ]

        schema = CollectionSchema(fields=fields, description="Wafer defect feature vectors")
        collection = Collection(name=self.collection_name, schema=schema)

        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        collection.create_index(field_name="vector", index_params=index_params)
        logger.info(f"Created Milvus collection: {self.collection_name}")

    def insert(
        self,
        defect_instance,
        wafer_id: str = "",
        layer: str = "",
        die_x: int = 0,
        die_y: int = 0,
    ) -> str:
        vector = defect_instance.to_feature_vector()
        if len(vector) < self.vector_dim:
            padded = np.zeros(self.vector_dim, dtype=np.float32)
            padded[:len(vector)] = vector
            vector = padded
        elif len(vector) > self.vector_dim:
            vector = vector[:self.vector_dim]

        record_id = str(uuid.uuid4())

        if self._pymilvus_available and self._connected and not self.use_mock_if_unavailable:
            return self._insert_milvus(record_id, vector, defect_instance, wafer_id, layer, die_x, die_y)
        else:
            return self._insert_mock(record_id, vector, defect_instance, wafer_id, layer, die_x, die_y)

    def _insert_milvus(
        self,
        record_id: str,
        vector: np.ndarray,
        defect_instance,
        wafer_id: str,
        layer: str,
        die_x: int,
        die_y: int,
    ) -> str:
        try:
            from pymilvus import Collection

            collection = Collection(self.collection_name)
            data = [{
                "id": record_id,
                "vector": vector.tolist(),
                "defect_type": defect_instance.defect_type.value,
                "confidence": float(defect_instance.confidence),
                "centroid_x": float(defect_instance.centroid_x),
                "centroid_y": float(defect_instance.centroid_y),
                "wafer_id": wafer_id,
                "layer": layer,
                "die_x": int(die_x),
                "die_y": int(die_y),
                "severity_level": int(defect_instance.severity_level),
                "created_at": float(time.time()),
            }]
            collection.insert(data)
            collection.flush()
            return record_id
        except Exception as e:
            logger.error(f"Milvus insert failed: {e}, using mock fallback")
            return self._insert_mock(record_id, vector, defect_instance, wafer_id, layer, die_x, die_y)

    def _insert_mock(
        self,
        record_id: str,
        vector: np.ndarray,
        defect_instance,
        wafer_id: str,
        layer: str,
        die_x: int,
        die_y: int,
    ) -> str:
        metadata = {
            "defect_type": defect_instance.defect_type.value,
            "confidence": float(defect_instance.confidence),
            "centroid_x": float(defect_instance.centroid_x),
            "centroid_y": float(defect_instance.centroid_y),
            "wafer_id": wafer_id,
            "layer": layer,
            "die_x": int(die_x),
            "die_y": int(die_y),
            "severity_level": int(defect_instance.severity_level),
            "geometric_features": defect_instance.geometric_features,
            "created_at": time.time(),
        }
        record = DefectVectorRecord(id=record_id, vector=vector.copy(), metadata=metadata)
        self._mock_storage[record_id] = record
        return record_id

    def insert_batch(
        self,
        defect_instances: List,
        wafer_id: str = "",
        layer: str = "",
        die_x: int = 0,
        die_y: int = 0,
    ) -> List[str]:
        ids = []
        for d in defect_instances:
            rid = self.insert(d, wafer_id, layer, die_x, die_y)
            ids.append(rid)
        return ids

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter_expr: Optional[str] = None,
    ) -> List[dict]:
        if len(query_vector) < self.vector_dim:
            padded = np.zeros(self.vector_dim, dtype=np.float32)
            padded[:len(query_vector)] = query_vector
            query_vector = padded
        elif len(query_vector) > self.vector_dim:
            query_vector = query_vector[:self.vector_dim]

        if self._pymilvus_available and self._connected and not self.use_mock_if_unavailable:
            return self._search_milvus(query_vector, top_k, filter_expr)
        else:
            return self._search_mock(query_vector, top_k)

    def _search_milvus(
        self,
        query_vector: np.ndarray,
        top_k: int,
        filter_expr: Optional[str],
    ) -> List[dict]:
        try:
            from pymilvus import Collection

            collection = Collection(self.collection_name)
            collection.load()

            search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
            results = collection.search(
                data=[query_vector.tolist()],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=["defect_type", "confidence", "centroid_x", "centroid_y", "wafer_id"],
            )

            formatted = []
            for hits in results:
                for hit in hits:
                    formatted.append({
                        "id": hit.id,
                        "distance": float(hit.distance),
                        "defect_type": hit.entity.get("defect_type", ""),
                        "confidence": float(hit.entity.get("confidence", 0.0)),
                        "centroid_x": float(hit.entity.get("centroid_x", 0.0)),
                        "centroid_y": float(hit.entity.get("centroid_y", 0.0)),
                        "wafer_id": hit.entity.get("wafer_id", ""),
                    })
            return formatted
        except Exception as e:
            logger.error(f"Milvus search failed: {e}")
            return []

    def _search_mock(
        self,
        query_vector: np.ndarray,
        top_k: int,
    ) -> List[dict]:
        if not self._mock_storage:
            return []

        results = []
        q_norm = np.linalg.norm(query_vector) + 1e-8

        for rid, record in self._mock_storage.items():
            r_norm = np.linalg.norm(record.vector) + 1e-8
            similarity = float(np.dot(query_vector, record.vector) / (q_norm * r_norm))
            distance = 1.0 - similarity
            results.append({
                "id": rid,
                "distance": distance,
                "defect_type": record.metadata.get("defect_type", ""),
                "confidence": record.metadata.get("confidence", 0.0),
                "centroid_x": record.metadata.get("centroid_x", 0.0),
                "centroid_y": record.metadata.get("centroid_y", 0.0),
                "wafer_id": record.metadata.get("wafer_id", ""),
            })

        results.sort(key=lambda x: x["distance"])
        return results[:top_k]

    def get_stats(self) -> dict:
        if self._pymilvus_available and self._connected and not self.use_mock_if_unavailable:
            try:
                from pymilvus import Collection

                collection = Collection(self.collection_name)
                stats = {
                    "backend": "milvus",
                    "collection": self.collection_name,
                    "connected": self._connected,
                    "vector_dim": self.vector_dim,
                }
                try:
                    stats["num_entities"] = int(collection.num_entities)
                except Exception:
                    stats["num_entities"] = 0
                return stats
            except Exception as e:
                return {"backend": "milvus", "error": str(e), "connected": False}
        else:
            return {
                "backend": "mock_memory",
                "connected": self._connected,
                "vector_dim": self.vector_dim,
                "num_entities": len(self._mock_storage),
            }

    def clear(self):
        if self._pymilvus_available and self._connected:
            try:
                from pymilvus import utility

                if utility.has_collection(self.collection_name):
                    utility.drop_collection(self.collection_name)
                    logger.info(f"Dropped Milvus collection: {self.collection_name}")
            except Exception as e:
                logger.error(f"Failed to drop Milvus collection: {e}")
        else:
            self._mock_storage.clear()

    def close(self):
        if self._pymilvus_available and self._connected:
            try:
                from pymilvus import connections

                connections.disconnect("default")
                self._connected = False
                logger.info("Milvus connection closed")
            except Exception as e:
                logger.debug(f"Milvus close error: {e}")


__all__ = ["MilvusVectorStore", "DefectVectorRecord"]
