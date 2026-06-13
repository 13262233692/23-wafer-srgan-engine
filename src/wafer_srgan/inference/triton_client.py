import logging
import time
from typing import Optional

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
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.grpc_url = grpc_url
        self.http_url = http_url
        self.protocol = protocol
        self.timeout = timeout
        self.max_concurrent_requests = max_concurrent_requests
        self._client = None

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
            )

            sr_image = result.as_numpy("output")
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

        if not hasattr(self, "_local_generator"):
            cfg = self._make_default_cfg()
            self._local_generator = build_generator(cfg)
            self._local_generator.eval()
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._local_generator.to(self._device)
            logger.info("Initialized local generator for fallback inference")

        with torch.no_grad():
            if lr_image.ndim == 3:
                t = torch.from_numpy(lr_image).permute(2, 0, 1).unsqueeze(0).float()
            elif lr_image.ndim == 4:
                t = torch.from_numpy(lr_image).float()
            else:
                raise ValueError(f"Unexpected shape: {lr_image.shape}")

            t = t.to(self._device)
            sr = self._local_generator(t)
            sr = sr.clamp(0.0, 1.0)
            return sr.cpu().numpy()

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
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._local_generator.to(self._device)
        self._local_generator.eval()
        logger.info(f"Loaded local generator weights from {checkpoint_path}")
