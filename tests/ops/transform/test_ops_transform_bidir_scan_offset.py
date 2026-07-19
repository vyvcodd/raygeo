import numpy as np
import pytest

from raygeo.ops import Ops
from raygeo.ops.types import CommandType


class TestBidirScanOffset:
    def test_empty_ops(self):
        ops = Ops()
        ops.apply_bidir_scan_offset(1.0)
        assert ops.is_empty()

    def test_zero_offset_noop(self):
        ops = Ops()
        ops.move_to(0, 0, 0)
        ops.scan_to(10, 0, 0, power_values=[128] * 10)
        n = ops.len()
        ops.apply_bidir_scan_offset(0.0)
        assert ops.len() == n

    def test_ltr_unchanged(self):
        ops = Ops()
        ops.move_to(0, 0, 0)
        ops.scan_to(10, 0, 0, power_values=[128] * 10)
        ops.apply_bidir_scan_offset(5.0)
        assert ops.len() == 2
        assert ops.endpoint(0) == (0.0, 0.0, 0.0)
        assert ops.endpoint(1) == (10.0, 0.0, 0.0)

    def test_rtl_shifted(self):
        ops = Ops()
        ops.move_to(10, 0, 0)
        ops.scan_to(0, 0, 0, power_values=[128] * 10)
        ops.apply_bidir_scan_offset(3.0)
        assert ops.len() == 2
        assert ops.endpoint(0) == (13.0, 0.0, 0.0)
        assert ops.endpoint(1) == (3.0, 0.0, 0.0)

    def test_state_between_transferred(self):
        ops = Ops()
        ops.move_to(10, 0, 0)
        ops.set_power(0.5)
        ops.set_feed_rate(500.0)
        ops.scan_to(0, 0, 0, power_values=[64] * 5)
        ops.apply_bidir_scan_offset(2.0)
        assert ops.len() == 4
        assert ops.command_type(0) == CommandType.MOVE_TO
        assert ops.command_type(1) == CommandType.SET_POWER
        assert ops.command_type(2) == CommandType.SET_FEED_RATE
        assert ops.command_type(3) == CommandType.SCAN_LINE

    def test_multiple_passes(self):
        ops = Ops()
        ops.move_to(0, 0, 0)
        ops.scan_to(10, 0, 0, power_values=[100] * 10)
        ops.move_to(10, 1, 0)
        ops.scan_to(0, 1, 0, power_values=[200] * 10)
        ops.apply_bidir_scan_offset(4.0)
        assert ops.len() == 4
        assert ops.endpoint(0) == (0.0, 0.0, 0.0)
        assert ops.endpoint(1) == (10.0, 0.0, 0.0)
        assert ops.endpoint(2) == (14.0, 1.0, 0.0)
        assert ops.endpoint(3) == (4.0, 1.0, 0.0)

    def test_yz_preserved(self):
        ops = Ops()
        ops.move_to(10, 5, 2)
        ops.scan_to(0, 5, 2, power_values=[128] * 10)
        ops.apply_bidir_scan_offset(3.0)
        assert ops.endpoint(0) == (13.0, 5.0, 2.0)
        assert ops.endpoint(1) == (3.0, 5.0, 2.0)

    def test_power_values_preserved(self):
        ops = Ops()
        ops.move_to(10, 0, 0)
        ops.scan_to(0, 0, 0, power_values=[10, 20, 30])
        ops.apply_bidir_scan_offset(1.0)
        assert list(ops.scanline_data(1)) == [10, 20, 30]


class TestBidirScanOffsetAtAngle:
    """scan_angle_deg must classify/shift relative to the actual scan
    direction, not always along X."""

    def test_default_angle_matches_x_only_behavior(self):
        ops_default = Ops()
        ops_default.move_to(10, 0, 0)
        ops_default.scan_to(0, 0, 0, power_values=[128] * 10)
        ops_default.apply_bidir_scan_offset(3.0)

        ops_explicit = Ops()
        ops_explicit.move_to(10, 0, 0)
        ops_explicit.scan_to(0, 0, 0, power_values=[128] * 10)
        ops_explicit.apply_bidir_scan_offset(3.0, 0.0)

        assert ops_default.endpoint(0) == ops_explicit.endpoint(0)
        assert ops_default.endpoint(1) == ops_explicit.endpoint(1)

    def test_vertical_forward_pass_unchanged(self):
        # raster generation emits output-space geometry with Y flipped, so
        # at scan_angle_deg=90 the reference direction is (0, -1): a pass
        # running -Y is "forward" (unchanged), not +Y.
        ops = Ops()
        ops.move_to(0, 10, 0)
        ops.scan_to(0, 0, 0, power_values=[128] * 10)
        ops.apply_bidir_scan_offset(2.0, 90.0)
        assert ops.endpoint(0) == pytest.approx((0.0, 10.0, 0.0))
        assert ops.endpoint(1) == pytest.approx((0.0, 0.0, 0.0))

    def test_vertical_reversed_pass_shifted_along_y_not_x(self):
        ops = Ops()
        ops.move_to(0, 0, 0)
        ops.scan_to(0, 10, 0, power_values=[128] * 10)
        ops.apply_bidir_scan_offset(2.0, 90.0)
        assert ops.endpoint(0) == pytest.approx((0.0, -2.0, 0.0))
        assert ops.endpoint(1) == pytest.approx((0.0, 8.0, 0.0))

    def test_vertical_classification_robust_to_x_noise(self):
        # Tiny x drift (like real rotation rounding) must not flip
        # classification, which is dominated by the real y component.
        ops = Ops()
        ops.move_to(0.0, 0.0, 0)
        ops.scan_to(1e-12, 10.0, 0, power_values=[128] * 10)
        ops.apply_bidir_scan_offset(2.0, 90.0)
        ex, ey, ez = ops.endpoint(0)
        assert ex == pytest.approx(0.0, abs=1e-9)
        assert ey == pytest.approx(-2.0)

    def test_45_degree_shift_splits_between_x_and_y(self):
        import math

        ops = Ops()
        ops.move_to(10, 0, 0)
        ops.scan_to(0, 10, 0, power_values=[128] * 10)
        ops.apply_bidir_scan_offset(2.0, 45.0)
        shift_x = 2.0 * math.cos(math.radians(45.0))
        shift_y = -2.0 * math.sin(math.radians(45.0))
        assert ops.endpoint(0) == pytest.approx(
            (10 + shift_x, 0 + shift_y, 0.0)
        )
        assert ops.endpoint(1) == pytest.approx(
            (0 + shift_x, 10 + shift_y, 0.0)
        )

    def test_no_more_collision_on_vertical_raster(self):
        # Regression test: column density must stay uniform after the
        # offset, not collide into half density.
        mask = np.ones((40, 100), dtype=np.uint8)
        baseline = Ops.from_mask_scan(
            mask, (10.0, 10.0), 0.0, 0.0, 0.2, 1.0, angle=90.0
        )
        offset = baseline.copy()
        offset.apply_bidir_scan_offset(0.5, 90.0)

        def column_count(ops):
            xs = set()
            i, n = 0, ops.len()
            while i < n:
                if ops.command_type(i) == CommandType.MOVE_TO:
                    j = i + 1
                    if j < n and ops.command_type(j) == CommandType.SCAN_LINE:
                        xs.add(round(ops.endpoint(i)[0], 6))
                        i = j + 1
                        continue
                i += 1
            return len(xs)

        assert column_count(offset) == column_count(baseline)

    def test_no_more_collision_at_45_degrees(self):
        # Regression test for the reference-direction sign bug: at 45
        # degrees the unflipped (cos, sin) reference made the
        # classification dot product exactly zero (not just small),
        # degrading into the same floating-point coin flip as the
        # original 90-degree bug and colliding lines together.
        import math

        mask = np.ones((100, 100), dtype=np.uint8)
        angle = 45.0
        baseline = Ops.from_mask_scan(
            mask, (10.0, 10.0), 0.0, 0.0, 0.2, 1.0, angle=angle
        )
        offset = baseline.copy()
        offset.apply_bidir_scan_offset(0.05, angle)

        def line_count(ops):
            c = math.cos(math.radians(angle))
            s = -math.sin(math.radians(angle))
            perps = set()
            i, n = 0, ops.len()
            while i < n:
                if ops.command_type(i) == CommandType.MOVE_TO:
                    j = i + 1
                    if j < n and ops.command_type(j) == CommandType.SCAN_LINE:
                        mx, my, _ = ops.endpoint(i)
                        perps.add(round(-mx * s + my * c, 4))
                        i = j + 1
                        continue
                i += 1
            return len(perps)

        assert line_count(offset) == line_count(baseline)
