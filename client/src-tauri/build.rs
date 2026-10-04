fn main() {
    tauri_build::build();
    build_cuda_kernels();
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
fn build_cuda_kernels() {
    let out_dir = std::path::PathBuf::from(
        std::env::var("OUT_DIR").expect("cargo always sets OUT_DIR for build scripts"),
    );
    let manifest = std::path::PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("cargo always sets CARGO_MANIFEST_DIR"),
    );
    let kernel = manifest.join("src/miners/gpu/kernels/probe.cu");

    println!("cargo:rerun-if-changed={}", kernel.display());
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

    for arch in ARCHES {
        let out = out_dir.join(format!("probe.{arch}.cubin"));

        let mut command = std::process::Command::new(&nvcc);
        command.arg("-cubin").arg(format!("-arch={arch}"));
        if let Some(host_compiler) = &host_compiler {
            command.arg("-ccbin").arg(host_compiler);
        }

        match command.arg("-o").arg(&out).arg(&kernel).status() {
            Ok(status) if status.success() => {}
            Ok(status) => {
                println!(
                    "cargo:warning=nvcc failed for {arch} (exit {status}); no cubin embedded for it"
                );
                let _ = std::fs::write(&out, []);
            }
            Err(error) => {
                println!("cargo:warning=could not run nvcc for {arch}: {error}");
                let _ = std::fs::write(&out, []);
            }
        }
    }
}

fn write_empty_cubins(out_dir: &std::path::Path) {
    for arch in ARCHES {
        let _ = std::fs::write(out_dir.join(format!("probe.{arch}.cubin")), []);
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
    if std::process::Command::new("nvcc").arg("--version").output().is_ok() {
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
