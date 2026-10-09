//! GPU mining backends.
//!
//! Every kernel is embedded in the binary — for CUDA, as a prebuilt cubin — and driven in-process
//! through the driver API. Nothing here spawns a process.
//!
//! Job feeding (turning a real `mining.notify` job into kernel arguments) lands with the mining
//! kernel itself. Today a backend's contract is that it loads its embedded image and computes the
//! right answer on the device it found, which is what [`GpuBackend::self_test`] enforces.

pub mod bufs;
pub mod cubins;
pub mod cuda;
pub mod fatbin;
pub mod pipeline;
pub mod triton;

use serde::Serialize;

/// What a probe found, for the Miners page and the logs.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DeviceInfo {
    /// The backend that owns the device, e.g. `cuda`.
    pub backend: String,
    pub name: String,
    /// As the driver reports it, e.g. `12.0`.
    pub compute_capability: String,
    /// The `sm_` target whose image this device runs, e.g. `sm_120a`.
    pub arch: String,
    pub multiprocessors: i32,
}

impl DeviceInfo {
    pub fn describe(&self) -> String {
        format!(
            "{} [{}/{}] {} SMs",
            self.name, self.arch, self.compute_capability, self.multiprocessors
        )
    }
}

/// A GPU mining backend: an embedded kernel plus the plumbing to run it.
pub trait GpuBackend: Send + Sync {
    /// Stable identifier, e.g. `cuda/sm_120a`.
    fn name(&self) -> String;

    fn device(&self) -> &DeviceInfo;

    /// Loads the embedded image and proves it computes the right answer on this device.
    ///
    /// This is the gate a backend passes before it is allowed to mine. It catches a missing image,
    /// a driver mismatch and a mis-compiled kernel without touching the network.
    fn self_test(&mut self) -> Result<(), String>;

    fn shutdown(&mut self);
}

/// Probes for a usable backend. `None` means this machine has no GPU we ship an image for, which is
/// a normal state — a machine with no NVIDIA card, or one built without a CUDA toolkit.
pub fn probe() -> Option<Box<dyn GpuBackend>> {
    match cudarc::driver::CudaContext::device_count() {
        Ok(count) if count > 0 => {}
        Ok(_) => {
            log::info!("gpu: no CUDA devices present");
            return None;
        }
        Err(error) => {
            log::info!("gpu: the CUDA driver is unavailable ({error})");
            return None;
        }
    }

    match cuda::CudaBackend::open(0) {
        Ok(backend) => Some(Box::new(backend)),
        Err(error) => {
            log::warn!("gpu: no CUDA backend for device 0: {error}");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end on real hardware: the embedded cubin loads through the driver and computes the
    /// right answer.
    ///
    /// Passes trivially where there is no NVIDIA device or the build had no CUDA toolkit, because
    /// there is nothing to assert on those machines. On a machine with a supported GPU this is the
    /// test that catches a bad image, a bad arch mapping or a bad launch.
    #[test]
    fn the_embedded_image_loads_and_computes_on_the_device() {
        let Some(mut backend) = probe() else {
            return;
        };

        let name = backend.name();
        eprintln!("self-testing {} on {}", name, backend.device().describe());
        backend
            .self_test()
            .unwrap_or_else(|error| panic!("{name} failed its self-test: {error}"));
        backend.shutdown();
    }
}
