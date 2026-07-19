use crate::ops::container::Ops;
use crate::ops::enums::{CommandCategory, CommandType};
use crate::ops::transform::{Phase, TransformCtx, Transformer};

/// Parameters for the [`apply_bidir_scan_offset`] transformer.
#[derive(Clone, Debug, PartialEq)]
pub struct BidirScanOffsetSpec {
    /// Offset in millimeters applied along the scan reference direction to
    /// passes running opposite it.
    pub offset_mm: f64,
    /// Scan angle in degrees (same convention as `raster()`'s `angle`).
    pub scan_angle_deg: f64,
}

impl Transformer for BidirScanOffsetSpec {
    fn phase(&self) -> Phase {
        Phase::PostProcessing
    }

    fn apply(&self, ctx: &mut TransformCtx<'_>) {
        apply_bidir_scan_offset(ctx.ops, self.offset_mm, self.scan_angle_deg);
    }

    fn name(&self) -> &str {
        "bidir_scan_offset"
    }

    fn cache_key(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.name().hash(&mut h);
        self.offset_mm.to_bits().hash(&mut h);
        h.finish()
    }
}

/// Shifts raster passes running opposite the scan reference direction
/// (derived from `scan_angle_deg`) by `offset_mm` along that direction.
///
/// Classifying by projected (dx, dy) rather than raw `scan_end.x <
/// move_end.x` matters once `scan_angle_deg != 0`: at a vertical scan,
/// dx is ~0, so the old raw-x test degraded into a coin flip on
/// floating-point noise. At `scan_angle_deg = 0.0` this reduces exactly
/// to the original behavior.
///
/// The reference direction is `(cos θ, -sin θ)`, not `(cos θ, sin θ)`:
/// raster generation emits geometry in output space with Y flipped
/// (`convert_y_to_output`), so a scan line generated at angle θ has
/// real output-space direction `(cos θ, -sin θ)`. Using the unflipped
/// sign matches by coincidence at θ = 0 (sin = 0) and still separates
/// passes correctly at θ = 90 (a clean negation, harmless to a sign
/// test), which is why both angles tested fine — but at θ = 45 the
/// unflipped reference is an exact mirror of the true direction, making
/// the dot product mathematically zero rather than merely small, and
/// classification degenerates into the same floating-point coin flip
/// this fix was meant to eliminate.
pub fn apply_bidir_scan_offset(
    ops: &mut Ops,
    offset_mm: f64,
    scan_angle_deg: f64,
) {
    if ops.is_empty() || offset_mm == 0.0 {
        return;
    }

    let angle_rad = scan_angle_deg.to_radians();
    let (ref_x, ref_y) = (angle_rad.cos(), -angle_rad.sin());

    let source = ops.copy();
    ops.clear();
    let n = source.len();
    let mut idx = 0;

    while idx < n {
        if source.command_type(idx) == CommandType::MoveTo {
            let move_end = source.endpoint(idx);

            // Skip STATE commands between MoveTo and potential ScanLine.
            let mut j = idx + 1;
            while j < n && source.category(j) == CommandCategory::State {
                j += 1;
            }

            if j < n && source.is_scanline(j) {
                let scan_end = source.endpoint(j);
                let dx = scan_end.x - move_end.x;
                let dy = scan_end.y - move_end.y;
                let along_ref = dx * ref_x + dy * ref_y;

                if along_ref < 0.0 {
                    // Opposite the reference direction: shift by offset_mm
                    // along that direction, not the pass's own.
                    let (shift_x, shift_y) =
                        (ref_x * offset_mm, ref_y * offset_mm);
                    let extra = source.extra_axes(idx).map(|ea| {
                        ea.iter().map(|(a, v)| (*a, *v)).collect::<Vec<_>>()
                    });
                    ops.move_to(
                        move_end.x + shift_x,
                        move_end.y + shift_y,
                        move_end.z,
                        extra,
                    );
                    // Transfer state commands between MoveTo and ScanLine.
                    for k in (idx + 1)..j {
                        ops.transfer_command_from(&source, k);
                    }
                    let scan_extra = source.extra_axes(j).map(|ea| {
                        ea.iter().map(|(a, v)| (*a, *v)).collect::<Vec<_>>()
                    });
                    ops.scan_to(
                        scan_end.x + shift_x,
                        scan_end.y + shift_y,
                        scan_end.z,
                        source.scanline_data(j),
                        scan_extra,
                    );
                } else {
                    // Along the reference direction, or zero-length.
                    for k in idx..=j {
                        ops.transfer_command_from(&source, k);
                    }
                }
                idx = j + 1;
                continue;
            }
        }
        ops.transfer_command_from(&source, idx);
        idx += 1;
    }
}
