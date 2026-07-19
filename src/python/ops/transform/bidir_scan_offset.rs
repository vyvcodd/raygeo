//! PyO3 binding for [`BidirScanOffsetSpec`](crate::ops::transform::bidir_scan_offset::BidirScanOffsetSpec).

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::ops::transform::bidir_scan_offset::BidirScanOffsetSpec as CoreBidirScanOffsetSpec;

/// Register the `BidirScanOffsetSpec` class on the `bidir_scan_offset`
/// submodule.
pub(crate) fn register(transform_mod: &Bound<'_, PyModule>) -> PyResult<()> {
    let bidir_mod = PyModule::new(transform_mod.py(), "bidir_scan_offset")?;
    bidir_mod.add_class::<BidirScanOffsetSpec>()?;
    transform_mod.add_submodule(&bidir_mod)?;

    let sys_modules = transform_mod.py().import("sys")?.getattr("modules")?;
    sys_modules
        .set_item("raygeo.ops.transform.bidir_scan_offset", &bidir_mod)?;

    Ok(())
}

/// Parameters for the ``BidirScanOffset`` transformer.
///
/// ``scan_angle_deg`` must match the ``angle`` the raster was generated
/// with (``raster()``'s ``angle``); defaults to 0.0 (horizontal).
#[gen_stub_pyclass]
#[pyclass(
    module = "raygeo.ops.transform.bidir_scan_offset",
    name = "BidirScanOffsetSpec",
    frozen,
    eq,
    from_py_object
)]
#[derive(Clone, PartialEq)]
pub struct BidirScanOffsetSpec {
    /// Offset in mm applied along the scan direction to opposing passes.
    #[pyo3(get)]
    pub offset_mm: f64,
    /// Scan angle in degrees the offset is applied relative to.
    #[pyo3(get)]
    pub scan_angle_deg: f64,
}

impl BidirScanOffsetSpec {
    /// Convert into the core-layer spec.
    pub fn into_core(self) -> CoreBidirScanOffsetSpec {
        CoreBidirScanOffsetSpec {
            offset_mm: self.offset_mm,
            scan_angle_deg: self.scan_angle_deg,
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl BidirScanOffsetSpec {
    #[new]
    #[pyo3(signature = (offset_mm, scan_angle_deg = 0.0))]
    fn new(offset_mm: f64, scan_angle_deg: f64) -> Self {
        Self {
            offset_mm,
            scan_angle_deg,
        }
    }
}
