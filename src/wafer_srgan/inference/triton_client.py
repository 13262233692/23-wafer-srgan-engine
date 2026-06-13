import gc
import logging
import threading
import time
from typing import Optional, List

import numpy as np

logger = logging.getLogger(__name__)


class TritonSRClient:
    def __init__(
        self,
        model_name: str = "srgan_generator",
        model_version: str = "1",
        grpc_url: str = "localhost:8001",
        http_url: str = "localhost:8000",
        protocol: str = "grpc",
        timeout: float = 30.0,
        max_concurrent_requests: int = 16,
        enable_memory_pool: bool = True,
        memory_pool_max_buffers: int = 64,
        memory_pool_reclaim_threshold: float = 0.85,
        auto_cleanup_interval: int = 100,
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.grpc_url = grpc_url
        self.http_url = http_url
        self.protocol = protocol
        self.timeout = timeout
        self.max_concurrent_requests = max_concurrent_requests
        self.enable_memory_pool = enable_memory_pool
        self.auto_cleanup_interval = auto_cleanup_interval

        self._client = None
        self._local_generator = None
        self._device = None
        self._memory_pool = None

        self._infer_counter = 0
        self._counter_lock = threading.Lock()

        self._memory_pool_max_buffers = memory_pool_max_buffers
        self._memory_pool_reclaim_threshold = memory_pool_reclaim_threshold

    def _init_memory_pool(self):
        if not self.enable_memory_pool or self._memory_pool is not None:
            return

        try:
            import torch
            from wafer_srgan.inference.memory_pool import TensorPoolManager

            device = self._get_device()
            self._memory_pool = TensorPoolManager.get_instance().get_pool(
                device=device,
                max_buffers=self._memory_pool_max_buffers,
                buffer_dtype=torch.float32,
                auto_reclaim_threshold=self._memory_pool_reclaim_threshold,
                preallocate=True,
            )
            logger.info(f"Memory pool initialized on {device}")
        except Exception as e:
            logger.warning(f"Failed to initialize memory pool: {e}")
            self._memory_pool = None

    def _get_device(self):
        if self._device is None:
            import torch
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            import tritonclient.grpc as grpc_client
            import tritonclient.http as http_client

            if self.protocol == "grpc":
                self._client = grpc_client.InferenceServerClient(
                    url=self.grpc_url,
                    verbose=False,
                )
            else:
                self._client = http_client.InferenceServerClient(
                    url=self.http_url,
                    verbose=False,
                )
        except ImportError:
            logger.warning("tritonclient not installed, using local fallback mode")
            self._client = None

        return self._client

    def _trigger_periodic_cleanup(self) -> bool:
        with self._counter_lock:
            self._infer_counter += 1
            should_clean = self._infer_counter % self.auto_cleanup_interval == 0

        if should_clean:
            self._cleanup_memory()

        return should_clean

    def _cleanup_memory(self):
        try:
            import torch

            if torch.cuda.is_available():
                current_device = self._get_device()
                if current_device.type == "cuda":
                    allocated = torch.cuda.memory_allocated(current_device)
                    reserved = torch.cuda.memory_reserved(current_device)
                    total = torch.cuda.get_device_properties(current_device).total_memory
                    usage = reserved / total

                    if usage > 0.7 or self._infer_counter % (self.auto_cleanup_interval * 5) == 0:
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize(current_device)

                        if self._memory_pool is not None:
                            self._memory_pool._reclaim_unused_buffers()

                        new_allocated = torch.cuda.memory_allocated(current_device)
                        new_reserved = torch.cuda.memory_reserved(current_device)
                        logger.info(
                            f"Memory cleanup: allocated {allocated/1e6:.1f}MB -> {new_allocated/1e6:.1f}MB, "
                            f"reserved {reserved/1e6:.1f}MB -> {new_reserved/1e6:.1f}MB, "
                            f"usage {usage:.2%}"
                        )
        except Exception as e:
            logger.debug(f"Memory cleanup failed: {e}")

    def is_server_live(self) -> bool:
        client = self._get_client()
        if client is None:
            return False
        try:
            return client.is_server_live()
        except Exception as e:
            logger.error(f"Triton server not live: {e}")
            return False

    def infer_single(self, lr_image: np.ndarray) -> np.ndarray:
        client = self._get_client()

        if client is None:
            return self._local_infer(lr_image)

        if lr_image.dtype != np.float32:
            lr_image = lr_image.astype(np.float32)

        if lr_image.ndim == 3:
            lr_image = lr_image[np.newaxis, :]

        if lr_image.max() > 1.0:
            lr_image = lr_image / 255.0

        try:
            import tritonclient.grpc as grpc_client
            import tritonclient.http as http_client

            inputs = []
            if self.protocol == "grpc":
                inp = grpc_client.InferInput("input", lr_image.shape, "FP32")
            else:
                inp = http_client.InferInput("input", lr_image.shape, "FP32")
            inp.set_data_from_numpy(lr_image)
            inputs.append(inp)

            outputs = []
            if self.protocol == "grpc":
                out = grpc_client.InferRequestedOutput("output")
            else:
                out = http_client.InferRequestedOutput("output")
            outputs.append(out)

            result = client.infer(
                model_name=self.model_name,
                model_version=self.model_version,
                inputs=inputs,
                outputs=outputs,
                client_timeout=self.timeout if self.timeout > 0 else None,
            )

            sr_image = result.as_numpy("output")

            del inputs, outputs, result
            self._trigger_periodic_cleanup()

            return sr_image

        except Exception as e:
            logger.error(f"Triton inference failed: {e}, falling back to local")
            return self._local_infer(lr_image)

    def infer_batch(self, lr_images: list[np.ndarray]) -> list[np.ndarray]:
        results = []
        for img in lr_images:
            sr = self.infer_single(img)
            results.append(sr)
        return results

    def _local_infer(self, lr_image: np.ndarray) -> np.ndarray:
        import torch
        from wafer_srgan.models.builder import build_generator
        from wafer_srgan.inference.memory_pool import release_pooled_tensor, pooled_tensor

        if not hasattr(self, "_local_generator") or self._local_generator is None:
            cfg = self._make_default_cfg()
            self._local_generator = build_generator(cfg)
            self._local_generator.eval()
            self._device = self._get_device()
            self._local_generator.to(self._device)
            logger.info("Initialized local generator for fallback inference")
            self._init_memory_pool()

        input_tensor = None
        sr_tensor = None

        try:
            with torch.no_grad():
                if lr_image.ndim == 3:
                    np_tensor = lr_image.transpose(2, 0, 1).astype(np.float32)
                    np_tensor = np_tensor[np.newaxis, :]
                elif lr_image.ndim == 4:
                    np_tensor = lr_image.astype(np.float32)
                else:
                    raise ValueError(f"Unexpected shape: {lr_image.shape}")

                if np_tensor.max() > 1.0:
                    np_tensor = np_tensor / 255.0

                if self.enable_memory_pool and self._memory_pool is not None:
                    input_tensor = pooled_tensor(
                        shape=np_tensor.shape,
                        dtype=torch.float32,
                        device=self._device,
                    )
                    input_tensor.copy_(torch.from_numpy(np_tensor))
                else:
                    input_tensor = torch.from_numpy(np_tensor).to(self._device)

                try:
                    sr_tensor = self._local_generator(input_tensor)
                except torch.cuda.OutOfMemoryError as oom:
                    logger.error(f"CUDA OOM during generator forward: {oom}")
                    self._emergency_cleanup()
                    sr_tensor = self._local_generator(input_tensor)

                sr_tensor = sr_tensor.clamp(0.0, 1.0)

                if self.enable_memory_pool and self._memory_pool is not None:
                    output_shape = sr_tensor.shape
                    output_tensor = pooled_tensor(output_shape, torch.float32, device=self._device)
                    output_tensor.copy_(sr_tensor)
                    sr_numpy = output_tensor.cpu().numpy()
                    release_pooled_tensor(output_tensor)
                else:
                    sr_numpy = sr_tensor.cpu().numpy()

                self._trigger_periodic_cleanup()

                return sr_numpy

        finally:
            if input_tensor is not None and self.enable_memory_pool and self._memory_pool is not None:
                release_pooled_tensor(input_tensor)

            if sr_tensor is not None:
                if not (self.enable_memory_pool and self._memory_pool is not None):
                    sr_tensor.data = torch.empty(0, device=self._device)
                del sr_tensor

            if 'output_tensor' in locals() and output_tensor is not None:
                release_pooled_tensor(output_tensor)

            torch.cuda.synchronize(self._device) if self._device.type == "cuda" else None

    def _emergency_cleanup(self):
        logger.warning("=== EMERGENCY CUDA MEMORY CLEANUP ===")
        import torch

        gc.collect()

        try:
            if self._memory_pool is not None:
                self._memory_pool.emergency_reclaim()
        except Exception:
            pass

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()

        logger.warning("Emergency cleanup complete")

    @staticmethod
    def _make_default_cfg():
        from omegaconf import OmegaConf
        return OmegaConf.create({
            "model": {
                "generator": {
                    "in_channels": 3, "out_channels": 3, "num_features": 64,
                    "num_residual_blocks": 16, "upscale_factor": 4, "residual_scaling": 0.2,
                }
            }
        })

    def load_local_weights(self, checkpoint_path: str):
        import torch
        from wafer_srgan.models.builder import build_generator

        cfg = self._make_default_cfg()
        self._local_generator = build_generator(cfg)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "generator" in ckpt:
            self._local_generator.load_state_dict(ckpt["generator"])
        else:
            self._local_generator.load_state_dict(ckpt)
        self._device = self._get_device()
        self._local_generator.to(self._device)
        self._local_generator.eval()
        self._init_memory_pool()
        logger.info(f"Loaded local generator weights from {checkpoint_path}")

    def get_memory_stats(self) -> dict:
        import torch

        stats = {
            "infer_count": self._infer_counter,
        }

        if self._memory_pool is not None:
            stats["memory_pool"] = self._memory_pool.get_stats()

        if torch.cuda.is_available() and self._device is not None and self._device.type == "cuda":
            stats["cuda"] = {
                "allocated_mb": torch.cuda.memory_allocated(self._device) / 1e6,
                "reserved_mb": torch.cuda.memory_reserved(self._device) / 1e6,
                "peak_allocated_mb": torch.cuda.max_memory_allocated(self._device) / 1e6,
                "peak_reserved_mb": torch.cuda.max_memory_reserved(self._device) / 1e6,
            }

        return stats

    def manual_cleanup(self):
        self._cleanup_memory()
        logger.info("Manual memory cleanup triggered")

    def close(self):
        if self._memory_pool is not None:
            try:
                self._memory_pool.close()
            except Exception:
                pass
            self._memory_pool = None

        if self._local_generator is not None:
            import torch
            self._local_generator.cpu()
            del self._local_generator
            self._local_generator = None

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        logger.info("TritonSRClient closed")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
