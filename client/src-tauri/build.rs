fn main() {
    tauri_build::build();
    build_cuda_kernels();
    build_pearl_gemm_extension();
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
    // The geometry switches (`POW_MBARRIER_RING`, `POW_XOR_SWIZZLE`, `POW_SERPENTINE`, `POW_BK`) are
    // compile-time, so a benchmark run has to reach `nvcc`. Space-separated, exactly as `nvcc` takes
    // them:
    //
    //     PEARL_NVCC_DEFINES="-DPOW_SERPENTINE=1" cargo build
    //
    // Every arch gets the same defines, so the cubin the client loads is the one being measured. The
    // env change is declared here because a switch flipped without a rebuild does nothing, and a stale
    // cubin reads as a measurement.
    println!("cargo:rerun-if-env-changed=PEARL_NVCC_DEFINES");
    let defines: Vec<String> = std::env::var("PEARL_NVCC_DEFINES")
        .unwrap_or_default()
        .split_whitespace()
        .map(String::from)
        .collect();

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

    build_arches(&nvcc, host_compiler.as_deref(), &kernel, &out_dir, &defines);
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
    defines: &[String],
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
        for define in defines {
            command.arg(define);
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

/// Rebuild the CUTLASS extension (`pearl_gemm_cuda`) that the Python side imports.
///
/// It is a CUDA build like the cubins, but it lives outside the crate, so nothing in the Tauri
/// pipeline would touch it: a header edited after the last build leaves a stale `.pyd`, and a
/// stale image reads at import time exactly like a fresh one — a measurement taken against it
/// measures the old kernel. So every source that feeds it is watched, and the build is invoked on
/// every build-script run; ninja recompiles what changed and no-ops otherwise.
///
/// The generated instantiations are watched through the config that generates them, not through
/// `csrc/gemm/instantiations`: `setup.py` rewrites that directory on every run, and watching it
/// would make the hook re-trigger itself on every build.
///
/// Toolchain policy matches the cubins: an absent toolchain warns and skips, a present one that
/// cannot compile fails the build. A failed rebuild that only warns would leave the stale image in
/// place, which is the exact failure this hook exists to prevent.
fn build_pearl_gemm_extension() {
    let manifest = std::path::PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("cargo always sets CARGO_MANIFEST_DIR"),
    );
    let pearl = manifest.join("../miners/pearl");
    let gemm = pearl.join("miner/pearl-gemm");
    if !gemm.join("setup.py").is_file() {
        println!("cargo:warning=pearl-gemm is not present: its extension will not be rebuilt");
        return;
    }

    println!("cargo:rerun-if-changed={}", gemm.join("setup.py").display());
    for source in [
        "csrc/gemm/collective_mainloop.hpp",
        "csrc/gemm/collective_epilogue.hpp",
        "csrc/gemm/pearl_noisingA_kernel.h",
        "csrc/gemm/pearl_noisingB_kernel.h",
        "csrc/gemm/pow_utils.hpp",
        "csrc/gemm/pearl_gemm_kernel.h",
        "csrc/gemm/kernel_traits.hpp",
        "csrc/gemm/heuristics.hpp",
        "csrc/gemm/named_barrier.hpp",
        "csrc/gemm/utils.h",
        "csrc/gemm/convert_util.h",
        "csrc/gemm/error_check.hpp",
        "csrc/gemm/host_signal_header.hpp",
        "csrc/gemm/print_matrix.hpp",
        "csrc/gemm/static_switch.h",
        "csrc/gemm/static_switch_matmul.h",
        "csrc/gemm/static_switch_noisingA.h",
        "csrc/gemm/static_switch_noisingB.h",
        "csrc/gemm/tile_scheduler.hpp",
        "csrc/gemm/pearl_api_params.h",
        "csrc/gemm/pearl_gemm_api.cpp",
        "csrc/gemm/pearl_gemm_constants.hpp",
        "csrc/gemm/pearl_gemm_decl.h",
        "csrc/gemm/pearl_gemm_host.h",
        "csrc/gemm/pearl_gemm_launch_template.h",
        "csrc/gemm/pearl_noisingA_host.h",
        "csrc/gemm/pearl_noisingB_host.h",
        "csrc/gemm/denoise_converter.cu",
        "csrc/gemm/denoise_converter_host.h",
        "csrc/gemm/denoise_converter_kernel.h",
        "csrc/gemm/inner_hash_kernel.cu",
        "csrc/gemm/inner_hash_kernel.h",
        "csrc/gemm/noise_generation.cu",
        "csrc/gemm/noise_generation_host.h",
        "csrc/gemm/noise_generation_kernel.h",
    ] {
        println!("cargo:rerun-if-changed={}", gemm.join(source).display());
    }
    // Directories `setup.py` never writes into, so watching them cannot self-trigger.
    for dir in ["csrc/blake3", "csrc/moe", "csrc/tensor_hash"] {
        println!("cargo:rerun-if-changed={}", gemm.join(dir).display());
    }
    println!(
        "cargo:rerun-if-changed={}",
        pearl
            .join("miner/pearl-gemm-build-utils/src/pearl_gemm_build_utils")
            .display()
    );
    println!("cargo:rerun-if-env-changed=PEARL_GEMM_ARCH");

    if find_nvcc().is_none() {
        println!(
            "cargo:warning=nvcc not found: the pearl-gemm extension will not be rebuilt, so a \
             stale .pyd may still be imported. Install the CUDA toolkit and rebuild."
        );
        return;
    }

    // The project pins Python 3.12, so the workspace venv is the interpreter to build with.
    // `uv run` is deliberately not used: it syncs the lock first, and the lock carries a
    // Linux-only wheel (`nvidia-cutlass-dsl-libs-base`) that cannot install on Windows, so the
    // sync fails before the build even starts. The venv is used directly; `PYTHONPATH` covers the
    // build-utils for a machine that has no venv.
    let build_utils = pearl.join("miner/pearl-gemm-build-utils/src");
    let mut interpreter = if cfg!(windows) {
        pearl.join(".venv/Scripts/python.exe")
    } else {
        pearl.join(".venv/bin/python")
    };
    if !interpreter.is_file() && !command_exists("python") {
        println!(
            "cargo:warning=no Python (the pearl .venv or `python` on PATH): the pearl-gemm \
             extension will not be rebuilt, so a stale .pyd may still be imported."
        );
        return;
    }
    if !interpreter.is_file() {
        // No venv on this machine; the plain interpreter is the fallback, and it may not be the
        // pinned 3.12, so say which one is being used rather than silently building for another.
        println!("cargo:warning=pearl .venv not found: building the extension with `python` on PATH");
        interpreter = std::path::PathBuf::from("python");
    }

    // The build runs through a batch file, not an inline `cmd /c` string: `Command` escapes the
    // quotes around `vcvarsall.bat` as `\"`, which `cmd` does not unescape, so an inline command
    // fails with "not recognized as a command" and — because `cmd` still exits 0 — the failure is
    // invisible. A batch file has no quotes to mangle.
    let out_dir = std::path::PathBuf::from(
        std::env::var("OUT_DIR").expect("cargo always sets OUT_DIR for build scripts"),
    );

    // `setup.py` defaults to sm_90a, which cannot load on a consumer Blackwell card. The build is
    // for the device this machine has, so the hook pins sm_120a unless the environment already
    // names an arch.
    let arches = std::env::var("PEARL_GEMM_ARCH").unwrap_or_else(|_| "sm_120a".to_string());

    let status = if cfg!(windows) {
        // nvcc shells out to `cl.exe` for the extension build too, and `cl.exe` needs the MSVC
        // environment (INCLUDE/LIB), not just the binary. `DISTUTILS_USE_SDK` tells setuptools to
        // trust that environment instead of probing for a developer prompt it cannot find from
        // inside a build script.
        let Some(vcvarsall) = find_vcvarsall() else {
            println!(
                "cargo:warning=no vcvarsall.bat found: the pearl-gemm extension will not be \
                 rebuilt, so a stale .pyd may still be imported. Install the Visual Studio C++ \
                 build tools and rebuild."
            );
            return;
        };
        let script = out_dir.join("pearl_gemm_build.bat");
        let contents = format!(
            "@echo off\r\ncall \"{}\" x64\r\nset DISTUTILS_USE_SDK=1\r\nset PYTHONPATH={}\r\nset PEARL_GEMM_ARCH={}\r\n{} setup.py build_ext --inplace\r\n",
            vcvarsall.display(),
            build_utils.display(),
            arches,
            interpreter.display()
        );
        if let Err(error) = std::fs::write(&script, &contents) {
            panic!("could not write the pearl-gemm build script: {error}");
        }
        std::process::Command::new("cmd")
            .current_dir(&gemm)
            .arg("/c")
            .arg(&script)
            .status()
    } else {
        std::process::Command::new("sh")
            .current_dir(&gemm)
            .env("PYTHONPATH", &build_utils)
            .env("PEARL_GEMM_ARCH", &arches)
            .args(["-c", &format!("{} setup.py build_ext --inplace", interpreter.display())])
            .status()
    };

    match status {
        Ok(status) if status.success() => {}
        Ok(status) => panic!(
            "the pearl-gemm extension build failed (exit {status}), so the stale .pyd from the \
             last build would still be imported as if it were current. Its diagnostics are above."
        ),
        Err(error) => panic!("could not run the pearl-gemm extension build: {error}"),
    }
}

/// True when `program` can be spawned, which is the check we want.
fn command_exists(program: &str) -> bool {
    std::process::Command::new(program)
        .arg("--version")
        .output()
        .is_ok()
}

/// `vcvarsall.bat` from the newest Visual Studio that has the C++ toolset.
fn find_vcvarsall() -> Option<std::path::PathBuf> {
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
            r"VC\Auxiliary\Build\vcvarsall.bat",
        ])
        .output()
        .ok()?;

    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(|line| std::path::PathBuf::from(line.trim()))
        .find(|path| path.is_file())
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
