"""Tests voor HeatmapRenderer en camera."""

import numpy as np
import pytest

from interface.heatmap import HeatmapRenderer, _PLASMA_LUT
from interface.camera import Camera


class TestPlasmaLUT:

    def test_lut_shape(self):
        assert _PLASMA_LUT.shape == (256, 3)

    def test_lut_dtype_uint8(self):
        assert _PLASMA_LUT.dtype == np.uint8

    def test_eerste_waarde_donkerblauw(self):
        # Eerste kleur (index 0) moet donkerblauw zijn: lage R, lage G, hoge B
        r, g, b = _PLASMA_LUT[0]
        assert b > r
        assert b > g

    def test_laatste_waarde_geelwit(self):
        # Laatste kleur (index 255) moet geel/wit zijn: hoge R, hoge G
        r, g, b = _PLASMA_LUT[255]
        assert r > 200
        assert g > 200


class TestHeatmapRenderer:

    def test_get_lut_geeft_kopie(self):
        lut1 = HeatmapRenderer.get_lut()
        lut2 = HeatmapRenderer.get_lut()
        assert lut1 is not lut2  # kopie, niet dezelfde referentie
        assert np.array_equal(lut1, lut2)

    def test_aanmaken_zonder_crash(self):
        renderer = HeatmapRenderer(896, 720)
        assert renderer.panel_w == 896
        assert renderer.panel_h == 720


class TestCamera:

    @pytest.fixture
    def camera(self):
        return Camera(896, 720)

    def test_world_to_screen_identity_bij_zoom1(self, camera):
        """Bij zoom=1 en offset=0 is world_to_screen identiek aan de invoer."""
        pos = np.array([100.0, 200.0], dtype=np.float32)
        screen = camera.world_to_screen(pos)
        assert abs(screen[0] - 100.0) < 0.1
        assert abs(screen[1] - 200.0) < 0.1

    def test_screen_to_world_inverse_van_world_to_screen(self, camera):
        world_pos = np.array([350.0, 450.0], dtype=np.float32)
        screen = camera.world_to_screen(world_pos)
        terug = camera.screen_to_world(screen)
        assert np.allclose(world_pos, terug, atol=0.5)

    def test_zoom_schaalt_schermcoordinaten(self, camera):
        camera.zoom_by(1.0)  # zoom = 2.0
        pos = np.array([100.0, 100.0], dtype=np.float32)
        screen = camera.world_to_screen(pos)
        assert abs(screen[0] - 200.0) < 0.1

    def test_zoom_begrenzing(self, camera):
        camera.zoom_by(100.0)
        assert camera.zoom <= 4.0
        camera.zoom_by(-100.0)
        assert camera.zoom >= 0.25

    def test_pan_verschuift_weergave(self, camera):
        camera.pan(100, 0)
        pos = np.array([0.0, 0.0], dtype=np.float32)
        screen = camera.world_to_screen(pos)
        assert screen[0] < 0  # verschoven naar links

    def test_reset_herstelt_standaard(self, camera):
        camera.zoom_by(1.5)
        camera.pan(200, 300)
        camera.reset()
        assert camera.zoom == 1.0
        assert np.allclose(camera.offset, [0, 0])
