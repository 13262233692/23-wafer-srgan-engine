import pytest
import numpy as np
from wafer_srgan.inference.edge_postprocess import EdgePostProcessor


class TestEdgePostProcessor:
    def test_canny_detection(self):
        proc = EdgePostProcessor(canny_threshold1=50, canny_threshold2=150)
        image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        edges = proc.detect_edges_canny(image)
        assert edges.shape == (128, 128)
        assert edges.dtype == np.uint8

    def test_sobel_detection(self):
        proc = EdgePostProcessor(sobel_ksize=3)
        image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        edges = proc.detect_edges_sobel(image)
        assert edges.shape == (128, 128)

    def test_defect_edges(self):
        proc = EdgePostProcessor(morph_kernel_size=3, morph_iterations=1)
        image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        edges = proc.detect_defect_edges(image, method="canny")
        assert edges.shape == (128, 128)

    def test_overlay(self):
        proc = EdgePostProcessor()
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        edges = np.zeros((64, 64), dtype=np.uint8)
        edges[10:20, 10:20] = 255
        overlay = proc.overlay_edges(image, edges)
        assert overlay.shape == (64, 64, 3)
