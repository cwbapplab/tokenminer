fn main() {
    tauri_build::build();
    build_cuda_kernels();
    build_fatbin();
}

/// The compute capabilities we ship a cubin for.
///
/// A cubin only loads on the architecture it was built for, so this list is also the set of GPUs
/// the client can mine on. Adding a card means adding a row here.
const ARCHES: &[&str] = &["sm_75", "sm_86", "sm_89", "sm_90a", "sm_120a"];

/// Compiles every kernel to a cubin per architecture, into `OUT_DIR` where `cubins.rs` embeds it.
///
/// Any missing piece of the toolchain writes the cubins empty instead of failing the build, so the
/// crate keeps compiling on a machine without CUDA and the runtime reports "no embedded cubin"
/// rather than the build breaking.
///
/// A toolchain that is *present* and still cannot compile the kernel is a different thing, and it
/// fails the build: an `mma` instruction on an architecture that predates it, or a typo in a
/// signature the launcher does not check, would otherwise embed five empty cubins and leave a build
/// that looks fine and a miner that silently never starts. See [`build_arches`].
fn build_cuda_kernels() {
    let out_dir = std::path::PathBuf::from(
        std::env::var("OUT_DIR").expect("cargo always sets OUT_DIR for build scripts"),
    );
    let manifest = std::path::PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("cargo always sets CARGO_MANIFEST_DIR"),
    );
    // One translation unit, several sources: `kernels.cu` includes the rest, because a cubin is a
    // single module and every entry point has to live in the one image.
    let kernels = manifest.join("src/miners/gpu/kernels");
    let kernel = kernels.join("kernels.cu");

    // Listed one by one rather than watching the directory, so a missing or renamed source is
    // visible here instead of silently producing a stale cubin.
    for source in ["kernels.cu", "blake3.cuh", "probe.cu", "pearl.cu"] {
        println!("cargo:rerun-if-changed={}", kernels.join(source).display());
    }
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    let Some(nvcc) = find_nvcc() else {
        println!(
            "cargo:warning=nvcc not found: no CUDA cubins will be embedded, so GPU mining will \
             report 'no embedded cubin'. Install the CUDA toolkit and rebuild to enable it."
        );
        write_empty_cubins(&out_dir);
        return;
    };

    // Windows only: nvcc shells out to `cl.exe` even for `-cubin`, and it is not on PATH outside a
    // developer prompt, so we have to point nvcc at it.
    let host_compiler = if cfg!(windows) { find_msvc_cl() } else { None };
    if cfg!(windows) && host_compiler.is_none() {
        println!(
            "cargo:warning=no MSVC host compiler (cl.exe) found: no CUDA cubins will be embedded. \
             Install the Visual Studio C++ build tools and rebuild."
        );
        write_empty_cubins(&out_dir);
        return;
    }

    build_arches(&nvcc, host_compiler.as_deref(), &kernel, &out_dir);
}

/// Compiles one cubin per architecture, and refuses to succeed if none of them compiled.
///
/// Losing *some* architectures is a real configuration -- a toolkit older than the newest card in
/// `ARCHES` simply cannot target it, and that card is not the one the developer is holding -- so a
/// partial failure is a warning and an empty cubin for the architectures that did not build. Losing
/// *all* of them has no such innocent reading: the toolchain ran and rejected the source, and the
/// resulting image is indistinguishable at runtime from a client with no GPU in it.
///
/// So that case panics. The cost of being wrong in the other direction is a broken build on a
/// machine that never had a GPU in it, and that machine is already covered above: no nvcc, or no
/// `cl.exe`, returns before this point and writes empty cubins without complaining.
fn build_arches(
    nvcc: &std::path::Path,
    host_compiler: Option<&std::path::Path>,
    kernel: &std::path::Path,
    out_dir: &std::path::Path,
) {
    let mut built = Vec::new();
    let mut failed = Vec::new();

    for arch in ARCHES {
        let out = out_dir.join(format!("kernels.{arch}.cubin"));

        let mut command = std::process::Command::new(nvcc);
        command.arg("-cubin").arg(format!("-arch={arch}"));
        if let Some(host_compiler) = host_compiler {
            command.arg("-ccbin").arg(host_compiler);
        }

        match command.arg("-o").arg(&out).arg(kernel).status() {
            Ok(status) if status.success() => built.push(*arch),
            Ok(status) => {
                println!(
                    "cargo:warning=nvcc failed for {arch} (exit {status}); no cubin embedded for it"
                );
                let _ = std::fs::write(&out, []);
                failed.push(*arch);
            }
            Err(error) => {
                println!("cargo:warning=could not run nvcc for {arch}: {error}");
                let _ = std::fs::write(&out, []);
                failed.push(*arch);
            }
        }
    }

    if built.is_empty() {
        panic!(
            "nvcc compiled the kernel for none of {ARCHES:?} (all failed: {failed:?}), so the \
             client would build with no GPU support at all. The compiler's diagnostics are above. \
             If this machine genuinely has no CUDA toolkit, remove it from PATH and unset \
             CUDA_PATH instead -- a toolchain that is absent is handled without failing the build."
        );
    }

    if !failed.is_empty() {
        println!(
            "cargo:warning=embedded cubins for {built:?} only; {failed:?} did not compile, so the \
             client will not mine on those GPUs."
        );
    }
}

fn write_empty_cubins(out_dir: &std::path::Path) {
    for arch in ARCHES {
        let _ = std::fs::write(out_dir.join(format!("kernels.{arch}.cubin")), []);
    }
}

/// The architectures the fatbin carries SASS for.
///
/// Unlike the cubins this is **one image for every card**: `cuModuleLoadData` picks the SASS entry
/// that matches the device, and falls back to the `compute_86` PTX when none matches. So this list is
/// not the set of GPUs the client can mine on the way the cubin list is — a card above `sm_86` runs
/// through the driver JIT, and a card below it has nothing to load at all. `sm_75` is in the cubin
/// table but cannot be in this one, because PTX is forward-compatible only.
const FATBIN_ARCHES: &[&str] = &["sm_86", "sm_89", "sm_120"];

/// The PTX the driver JITs when no SASS entry matches the device.
const FATBIN_PTX: &str = "compute_86";

/// Compiles the split search path's kernels into one multi-arch fatbin, into `OUT_DIR` where
/// `fatbin.rs` embeds it.
///
/// This is the reference's `csrc/build_fatbin.sh`, moved into the build script so the image is
/// embedded in the binary instead of loaded from a file at runtime.
///
/// Toolchain policy matches the cubins: an absent toolchain warns and writes an empty image, a
/// present one that cannot compile fails the build. The difference is that this image is now the only
/// search path, so an empty fatbin is a miner that never starts rather than a client with no GPU in
/// it — which is exactly why a present-but-failing toolchain has to be loud.
///
/// One consequence of the single-image shape: the whole build needs a toolkit that knows `sm_120`, so
/// a toolkit older than CUDA 12.4 fails here even on a machine holding a 4090. The crate already
/// assumes 12.4 (`cudarc`'s `cuda-12040` feature), so that is not a new requirement — it is the same
/// one, now enforced at compile time rather than at the first launch.
fn build_fatbin() {
    let out_dir = std::path::PathBuf::from(
        std::env::var("OUT_DIR").expect("cargo always sets OUT_DIR for build scripts"),
    );
    let manifest = std::path::PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("cargo always sets CARGO_MANIFEST_DIR"),
    );
    let kernels = manifest.join("src/miners/gpu/kernels");
    let unit = kernels.join("kernels_only.cu");
    let out = out_dir.join("pearl_gemm.fatbin");

    // Listed one by one, as the cubins are: a header edited after the last build leaves a stale
    // image, and a stale image reads at launch exactly like a fresh one.
    for source in [
        "kernels_only.cu",
        "extern_c_shims.inc",
        "pearl_gemm/blake3_sm80.cuh",
        "pearl_gemm/noise_generation_sm80.cuh",
        "pearl_gemm/pearl_blake3_compare_sm80.cuh",
        "pearl_gemm/pow_scan_emit_sm80.cuh",
    ] {
        println!("cargo:rerun-if-changed={}", kernels.join(source).display());
    }
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    let Some(nvcc) = find_nvcc() else {
        println!(
            "cargo:warning=nvcc not found: no fatbin will be embedded, so the split search path \
             has no kernels to run. Install the CUDA toolkit and rebuild."
        );
        let _ = std::fs::write(&out, []);
        return;
    };

    // Windows only: nvcc shells out to `cl.exe` even for a device-only image. Whether `-fatbin`
    // actually needs it is not settled — it emits no host code — so passing it when it is found is
    // the safe reading, and the policy for a machine that has no `cl.exe` is the same as the cubins'.
    let host_compiler = if cfg!(windows) { find_msvc_cl() } else { None };
    if cfg!(windows) && host_compiler.is_none() {
        println!(
            "cargo:warning=no MSVC host compiler (cl.exe) found: no fatbin will be embedded. \
             Install the Visual Studio C++ build tools and rebuild."
        );
        let _ = std::fs::write(&out, []);
        return;
    }

    let mut command = std::process::Command::new(nvcc);
    command.args([
        "-O3",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
    ]);
    command.arg("-I").arg(kernels.join("pearl_gemm"));
    for arch in FATBIN_ARCHES {
        let ptx = format!("compute_{}", arch.trim_start_matches("sm_"));
        command.arg("-gencode").arg(format!("arch={ptx},code={arch}"));
    }
    command.arg("-gencode")
        .arg(format!("arch={FATBIN_PTX},code={FATBIN_PTX}"));
    if let Some(host_compiler) = host_compiler.as_deref() {
        command.arg("-ccbin").arg(host_compiler);
    }

    match command.arg("-fatbin").arg("-o").arg(&out).arg(&unit).status() {
        Ok(status) if status.success() => {}
        Ok(status) => panic!(
            "nvcc could not build the fatbin (exit {status}), so the client would build with no \
             search kernels at all. Its diagnostics are above. The build needs a toolkit that knows \
             sm_120 (CUDA 12.4+); if this machine genuinely has no CUDA toolkit, remove it from PATH \
             and unset CUDA_PATH instead — a toolchain that is absent is handled without failing the \
             build."
        ),
        Err(error) => panic!("could not run nvcc for the fatbin: {error}"),
    }
}

/// `nvcc` from `CUDA_PATH`, then `PATH`, then the newest toolkit under the default install root.
fn find_nvcc() -> Option<std::path::PathBuf> {
    if let Ok(root) = std::env::var("CUDA_PATH") {
        let candidate = std::path::Path::new(&root).join("bin").join("nvcc.exe");
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    // `output()` fails when the program cannot be spawned, which is the check we want.
    if std::process::Command::new("nvcc")
        .arg("--version")
        .output()
        .is_ok()
    {
        return Some(std::path::PathBuf::from("nvcc"));
    }

    let root = std::path::Path::new("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA");
    let mut toolkits: Vec<_> = std::fs::read_dir(root)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .collect();
    toolkits.sort();

    // Newest first: CUDA 13 dropped offline compilation below sm_75, so the newest is the one that
    // understands the most architectures.
    toolkits
        .into_iter()
        .rev()
        .map(|dir| dir.join("bin").join("nvcc.exe"))
        .find(|candidate| candidate.is_file())
}

/// The directory holding MSVC's `cl.exe`, for nvcc's `-ccbin`.
///
/// Located through `vswhere` rather than by walking Program Files, because the toolset version in
/// the path changes with every Visual Studio update.
fn find_msvc_cl() -> Option<std::path::PathBuf> {
    let vswhere = std::env::var("ProgramFiles(x86)")
        .ok()
        .map(|root| {
            std::path::Path::new(&root)
                .join("Microsoft Visual Studio")
                .join("Installer")
                .join("vswhere.exe")
        })
        .filter(|path| path.is_file())?;

    let output = std::process::Command::new(vswhere)
        .args([
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-find",
            r"VC\Tools\MSVC\**\bin\Hostx64\x64\cl.exe",
        ])
        .output()
        .ok()?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut candidates: Vec<_> = stdout
        .lines()
        .map(|line| std::path::PathBuf::from(line.trim()))
        .filter(|path| path.is_file())
        .collect();
    candidates.sort();

    // The toolset version is a path component, so the highest sorts last.
    candidates
        .pop()
        .and_then(|cl| cl.parent().map(std::path::Path::to_path_buf))
}
