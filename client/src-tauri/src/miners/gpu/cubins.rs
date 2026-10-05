//! The CUDA cubins `build.rs` produced, embedded in the binary.
//!
//! They are compiled at build time and `include_bytes!`'d here, so the shipped binary carries every
//! image it needs and never writes to disk or fetches anything at run time.
//!
//! A build machine without `nvcc` still produces the files — empty — so the crate keeps building.
//! Presence therefore has to be checked, never assumed: see [`kernels_cubin`].

/// A cubin compiled for one architecture.
pub struct Cubin {
    pub arch: &'static str,
    pub bytes: &'static [u8],
}

macro_rules! cubins {
    ($($arch:literal),* $(,)?) => {
        &[$(Cubin {
            arch: $arch,
            bytes: include_bytes!(concat!(env!("OUT_DIR"), "/kernels.", $arch, ".cubin")),
        }),*]
    };
}

/// Every architecture we embed an image for, in build order.
///
/// One cubin per architecture, carrying both the self-test probes and the mining kernels: the driver
/// loads it as a single module, so they have to be in the same image.
pub static KERNEL_CUBINS: &[Cubin] = cubins!("sm_75", "sm_86", "sm_89", "sm_90a", "sm_120a");

/// The embedded image for `arch`, or `None` when that arch was not built.
pub fn kernels_cubin(arch: &str) -> Option<&'static [u8]> {
    KERNEL_CUBINS
        .iter()
        .find(|cubin| cubin.arch == arch)
        .map(|cubin| cubin.bytes)
        .filter(|bytes| !bytes.is_empty())
}

/// The architectures that actually have an image, for error messages and the UI.
pub fn embedded_arches() -> Vec<&'static str> {
    KERNEL_CUBINS
        .iter()
        .filter(|cubin| !cubin.bytes.is_empty())
        .map(|cubin| cubin.arch)
        .collect()
}

/// True when `nvcc` was available at build time.
pub fn any_embedded() -> bool {
    KERNEL_CUBINS.iter().any(|cubin| !cubin.bytes.is_empty())
}
