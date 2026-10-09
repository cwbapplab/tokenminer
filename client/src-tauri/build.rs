//! Build script: compile llmjob's vendored Pearl CUDA core into a static
//! library and link it into the client.
//!
//! The search kernel is not ours. It lives in `vendor/llmjob/earn/native/src/`
//! and is compiled here, from source, by the same recipe llmjob's own
//! `build-local.ps1` and `native-core.yml` use — `nvcc -c` per translation
//! unit, archived into one static library, with a thin flat-ABI shim of our own
//! (`native/pearl/pearl_shim.cpp`) on top of its `extern "C"` API.
//!
//! Toolchain policy, unchanged from the cubin era:
//!   * `nvcc` or `cl.exe` absent -> warn and build WITHOUT the native core. The
//!     crate still compiles and the engine reports "no CUDA core" at runtime,
//!     exactly as it used to report "no embedded cubin".
//!   * toolchain present but the source will not compile -> **panic**. A build
//!     that silently ships an image with no search in it is the one failure this
//!     policy exists to prevent.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    watch_bundle_icons();
    tauri_build::build();
    build_pearl_native();
}

/// Re-run when a bundled icon changes.
///
/// `tauri_build::build()` watches the config and the capabilities, but not the
/// icon files: only its `codegen`-feature path emits those. The Windows resource
/// library embeds `icons/icon.ico` during the build script, so without this an
/// incremental rebuild relinks the executable against the icon it had last time
/// it happened to re-run — a re-skinned app that still shows the old one until a
/// full `cargo clean`.
fn watch_bundle_icons() {
    let manifest =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"));

    // The icon list is read from the config rather than hardcoded, so a set that
    // is trimmed here (the .icns is macOS-only, say) is still the set watched.
    let config_path = manifest.join("tauri.conf.json");
    let config = std::fs::read_to_string(&config_path)
        .unwrap_or_else(|e| panic!("could not read {}: {e}", config_path.display()));
    let config: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("could not parse {}: {e}", config_path.display()));

    let icons = config
        .get("bundle")
        .and_then(|bundle| bundle.get("icon"))
        .and_then(serde_json::Value::as_array)
        .expect("tauri.conf.json has no `bundle.icon` array");

    for icon in icons {
        let icon = icon
            .as_str()
            .expect("every `bundle.icon` entry must be a string");
        println!("cargo:rerun-if-changed={}", manifest.join(icon).display());
    }
}

/// The compute capabilities to compile for, given a toolkit that knows them.
///
/// This is llmjob's own shipping list (`native-core.yml`): Turing through
/// Blackwell, with **`sm_90a` and never `sm_90`** because the Hopper fold is
/// wgmma, which only the `a` target has.
///
/// `PEARL_CUDA_ARCHES` overrides the list (comma-separated `compute_XX`), which
/// is what kernel work wants: six architectures of `-O3` is minutes of ptxas,
/// and one is seconds.
const PEARL_ARCHES: &[&str] = &[
    "compute_75",
    "compute_80",
    "compute_86",
    "compute_89",
    "compute_90a",
    "compute_120",
];

/// The vendored sources the static library is built from, in compile order.
///
/// `pearl_kernel.cu` is listed first only because it is the slow one; order
/// does not matter to the archive.
const VENDORED_SOURCES: &[&str] = &["pearl_kernel.cu", "pearl_host.cu"];

/// Our shim, compiled with the same flags and archived alongside them.
const SHIM_SOURCE: &str = "pearl_shim.cpp";

/// Compile the vendored core and the shim into `OUT_DIR/pearl_cuda.lib` and tell
/// cargo to link it.
fn build_pearl_native() {
    let out_dir = PathBuf::from(
        std::env::var("OUT_DIR").expect("cargo always sets OUT_DIR for build scripts"),
    );
    let manifest = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("cargo always sets CARGO_MANIFEST_DIR"),
    );
    let native = manifest.join("native/pearl");
    // The submodule root, resolved relative to this crate. Kept as one place so
    // a moved vendor directory fails here rather than inside nvcc.
    let vendored = manifest.join("../../vendor/llmjob/earn/native");

    // Rebuild when any input moves. Listed by name rather than by watching the
    // directories, so a renamed source is visible here instead of leaving a
    // stale library behind: an archive built from a file that no longer exists
    // is indistinguishable at link time from a current one.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=PEARL_CUDA_ARCHES");
    for name in [SHIM_SOURCE, "pearl_ffi.h", "pearl_abi_check.cpp"] {
        println!("cargo:rerun-if-changed={}", native.join(name).display());
    }
    for name in VENDORED_SOURCES {
        println!("cargo:rerun-if-changed={}", vendored.join("src").join(name).display());
    }
    for name in ["pearl_config.h", "pearl_tensor_map.h", "pearl_fold_bd.cuh"] {
        println!("cargo:rerun-if-changed={}", vendored.join("src").join(name).display());
    }
    // Missing submodule is not a toolchain problem and must not be silent: the
    // client would build with no search in it and no message saying so.
    if !vendored.join("src/pearl_host.cu").is_file() {
        panic!(
            "the vendored Pearl core is missing at {}. Run `git submodule update --init \
             vendor/llmjob` to fetch it.",
            vendored.display()
        );
    }

    check_sentinels(&vendored);

    let Some(nvcc) = find_nvcc() else {
        println!(
            "cargo:warning=nvcc not found: the vendored Pearl CUDA core will NOT be built, so \
             Pearl mining will report 'no CUDA core'. Install the CUDA toolkit (12.8 or newer for \
             Blackwell) and rebuild to enable it."
        );
        return;
    };

    // Windows: nvcc shells out to `cl.exe` for the host half of every
    // translation unit, and it is not on PATH outside a developer prompt.
    let host_compiler = if cfg!(windows) { find_msvc_cl() } else { None };
    if cfg!(windows) && host_compiler.is_none() {
        println!(
            "cargo:warning=no MSVC host compiler (cl.exe) found: the vendored Pearl CUDA core will \
             NOT be built. Install the Visual Studio C++ build tools and rebuild."
        );
        return;
    }

    let arches = selected_arches();

    // One object per translation unit, in a directory keyed by nothing in
    // particular: cargo already rebuilds this script only when an input moved,
    // so the objects are as fresh as the sources.
    let obj_dir = out_dir.join("pearl-objects");
    if let Err(error) = std::fs::create_dir_all(&obj_dir) {
        panic!("could not create {}: {error}", obj_dir.display());
    }

    let mut objects = Vec::new();
    // Source and its include directory, kept together: the vendored pair live in
    // the submodule and include their own headers by bare name, the shim lives
    // here and includes `pearl_ffi.h` beside it.
    let units: [(PathBuf, PathBuf); 3] = [
        (vendored.join("src/pearl_kernel.cu"), vendored.join("src")),
        (vendored.join("src/pearl_host.cu"), vendored.join("src")),
        (native.join(SHIM_SOURCE), native.clone()),
    ];

    for (index, (source, include)) in units.iter().enumerate() {
        let stem = source
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unit");
        let object = obj_dir.join(format!("{index}-{stem}.o"));

        compile_unit(
            &nvcc,
            host_compiler.as_deref(),
            source,
            include,
            &object,
            &arches,
        );

        objects.push(object);
    }

    let library = out_dir.join(archive_name());

    // `nvcc -lib` is what llmjob archives with on Windows and it is a thin
    // wrapper over the host linker; on Linux the toolchain's answer is a plain
    // `ar` (`native-core.yml` does exactly this split).
    let archived = if cfg!(windows) {
        let mut command = Command::new(&nvcc);
        command.arg("-lib").arg("-cudart").arg("static");
        if let Some(host_compiler) = host_compiler.as_deref() {
            command.arg("-ccbin").arg(host_compiler);
        }
        command.args(&objects).arg("-o").arg(&library);
        run(command, "archiving the Pearl core")
    } else {
        let mut command = Command::new("ar");
        command.arg("rcs").arg(&library).args(&objects);
        run(command, "archiving the Pearl core")
    };
    if !archived {
        panic!(
            "could not archive the Pearl CUDA core into {}. A toolchain that is present must not \
             produce a build with no search in it; the diagnostics are above.",
            library.display()
        );
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=pearl_cuda");

    // The CUDA runtime, statically, from the toolkit that built the core. On
    // Windows this is `cudart_static.lib` under `lib/x64`; elsewhere `libcudart.a`
    // has to come from the toolkit too, since the vendored core is built with
    // `-cudart static`.
    link_cuda_runtime(&nvcc);

    // The ABI check: compiled against the vendored header, run, and made to fail
    // the build when our flat profile no longer matches the core's.
    run_abi_check(&nvcc, &native, &vendored, &out_dir);
}

/// The architectures this toolkit will be asked for.
///
/// The default list is llmjob's; `PEARL_CUDA_ARCHES` narrows it, which is what
/// makes kernel iteration bearable. An unknown name is passed through rather
/// than filtered, so a toolkit that supports something not in the list can still
/// be targeted explicitly and a typo fails loudly in nvcc.
fn selected_arches() -> Vec<String> {
    match std::env::var("PEARL_CUDA_ARCHES") {
        Ok(list) if !list.trim().is_empty() => list
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect(),
        _ => PEARL_ARCHES.iter().map(|s| s.to_string()).collect(),
    }
}

/// `pearl_cuda.lib` on Windows, `libpearl_cuda.a` elsewhere — cargo takes the
/// bare name from `rustc-link-lib`, so the file has to carry the platform's own
/// decoration.
fn archive_name() -> &'static str {
    if cfg!(windows) {
        "pearl_cuda.lib"
    } else {
        "libpearl_cuda.a"
    }
}

/// Compile one translation unit.
///
/// Flags are llmjob's own (`native-core.yml`), minus the flags that exist for
/// the addon rather than the code: `-Xptxas -v` is a diagnostic their CI parses
/// and we do not, and `-Xcompiler -fPIC` is only needed when the result is
/// linked into a shared object, which a Rust *static* library is not.
fn compile_unit(
    nvcc: &Path,
    host_compiler: Option<&Path>,
    source: &Path,
    include: &Path,
    output: &Path,
    arches: &[String],
) {
    let mut command = Command::new(nvcc);
    command.args(["-O3", "-std=c++17", "-cudart", "static"]);
    for arch in arches {
        // `compute_120` -> `-gencode arch=compute_120,code=sm_120`. The `a` in
        // `compute_90a` is part of both halves and must survive the rewrite.
        let sm = arch.replacen("compute_", "sm_", 1);
        command.arg("-gencode").arg(format!("arch={arch},code={sm}"));
    }
    command.arg("-I").arg(include);
    if let Some(host_compiler) = host_compiler {
        command.arg("-ccbin").arg(host_compiler);
    }
    command.arg("-c").arg(source).arg("-o").arg(output);

    if !run(command, &format!("compiling {}", source.display())) {
        panic!(
            "nvcc could not compile {} for {}. The vendored core is the search kernel; a build \
             without it mines nothing while appearing to work, so this is a hard failure. The \
             compiler's diagnostics are above. If this machine genuinely has no CUDA toolkit, \
             remove it from PATH and unset CUDA_PATH instead — an absent toolchain is handled \
             without failing the build.",
            source.display(),
            arches.join(", ")
        );
    }
}

/// Point the link at the CUDA runtime that built the core.
fn link_cuda_runtime(nvcc: &Path) {
    let root = cuda_root(nvcc);
    match root {
        Some(root) => {
            let lib = if cfg!(windows) {
                root.join("lib").join("x64")
            } else {
                root.join("lib64")
            };
            println!("cargo:rustc-link-search=native={}", lib.display());
            println!("cargo:rustc-link-lib=static={}", cudart_name());
        }
        None => {
            // The toolkit's binaries were found without a layout we recognise,
            // so name the library and let the linker's own search path try.
            println!("cargo:rustc-link-lib=static={}", cudart_name());
        }
    }
}

fn cudart_name() -> &'static str {
    if cfg!(windows) {
        "cudart_static"
    } else {
        "cudart"
    }
}

/// Compile and run `pearl_abi_check.cpp`.
///
/// Compiled by the host compiler rather than nvcc: it includes the vendored
/// header, which under `__CUDACC__` would pull in the device-side macros, and
/// the check is about the plain-C++ layout both sides agree on.
fn run_abi_check(nvcc: &Path, native: &Path, vendored: &Path, out_dir: &Path) {
    let source = native.join("pearl_abi_check.cpp");
    let exe = out_dir.join(if cfg!(windows) {
        "pearl_abi_check.exe"
    } else {
        "pearl_abi_check"
    });

    // Compiled and linked in one nvcc call. nvcc is used rather than the host
    // compiler directly because it is the only C++ compiler this script can
    // locate on every platform: on Windows a bare `cl.exe` needs a full vcvars
    // environment, not just a path, and nvcc brings it up itself.
    let mut command = Command::new(nvcc);
    command
        .args(["-std=c++17", "-O0", "-cudart", "static"])
        .arg("-I")
        .arg(vendored.join("src"))
        .arg("-I")
        .arg(native);
    if let Some(host_compiler) = find_msvc_cl() {
        command.arg("-ccbin").arg(host_compiler);
    }
    command.arg("-o").arg(&exe).arg(&source);

    if !run(command, "building the Pearl ABI check") {
        panic!(
            "the Pearl ABI check does not compile against the vendored header. That header moved \
             (or the submodule was bumped) in a way our flat ABI no longer matches; see \
             native/pearl/pearl_abi_check.cpp."
        );
    }

    let output = Command::new(&exe)
        .output()
        .unwrap_or_else(|e| panic!("could not run {}: {e}", exe.display()));
    if !output.status.success() {
        panic!(
            "the Pearl ABI check failed. Our flat PearlProfileFlat no longer matches the vendored \
             PearlProfile, which means every profile we create would be read as a different \
             geometry by the core. Its output: {}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    println!(
        "cargo:warning={}",
        String::from_utf8_lossy(&output.stdout).trim()
    );
}

/// Assert the vendored geometry constants still match what the Rust side
/// hardcodes.
///
/// `pearl_abi_check.cpp` does the authoritative check at compile time, against
/// the real header, and this is the same check one layer earlier — for the case
/// where the check itself cannot compile because a constant it names is gone.
/// A submodule bump that moves `PEARL_FOLD_K` changes every hash the miner
/// produces, and nothing else in the build would notice.
fn check_sentinels(vendored: &Path) {
    let header = vendored.join("src/pearl_config.h");
    let text = std::fs::read_to_string(&header)
        .unwrap_or_else(|e| panic!("could not read {}: {e}", header.display()));

    for (name, expected) in [
        ("PEARL_CONFIG_BYTES", "52"),
        ("PEARL_HEADER_BYTES", "76"),
        ("PEARL_HASH_BYTES", "32"),
        ("PEARL_FOLD_RANK", "128u"),
        ("PEARL_FOLD_K", "2048u"),
        ("PEARL_ROWS_COUNT", "16"),
        ("PEARL_COLS_COUNT", "16"),
    ] {
        let needle = format!("#define {name} {expected}");
        if !text.contains(&needle) {
            panic!(
                "the vendored Pearl core no longer defines `{needle}`. config.rs hardcodes the \
                 Rust side of the same geometry, so the submodule was changed in a way that would \
                 silently alter every hash this miner produces. Update config.rs and this list \
                 together."
            );
        }
    }
}

/// Run a command, echo it, and report whether it succeeded.
fn run(mut command: Command, what: &str) -> bool {
    println!("cargo:warning={what}: {}", describe(&command));
    match command.status() {
        Ok(status) if status.success() => true,
        Ok(status) => {
            println!("cargo:warning={what} failed with {status}");
            false
        }
        Err(error) => {
            println!("cargo:warning={what} could not run: {error}");
            false
        }
    }
}

/// A readable form of a command, for the build log.
fn describe(command: &Command) -> String {
    let mut out = command.get_program().to_string_lossy().into_owned();
    for arg in command.get_args() {
        out.push(' ');
        out.push_str(&arg.to_string_lossy());
    }
    out
}

/// `nvcc` from `CUDA_PATH`, then `PATH`, then the newest toolkit under the
/// default install root.
fn find_nvcc() -> Option<PathBuf> {
    if let Ok(root) = std::env::var("CUDA_PATH") {
        let candidate = Path::new(&root).join("bin").join("nvcc.exe");
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    // `output()` fails when the program cannot be spawned, which is the check we want.
    if Command::new("nvcc").arg("--version").output().is_ok() {
        return Some(PathBuf::from("nvcc"));
    }

    let root = Path::new("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA");
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

/// The toolkit root that owns `nvcc`, if the path has the usual `bin/nvcc` shape.
fn cuda_root(nvcc: &Path) -> Option<PathBuf> {
    if let Ok(root) = std::env::var("CUDA_PATH") {
        let root = PathBuf::from(root);
        if root.is_dir() {
            return Some(root);
        }
    }
    // `<root>/bin/nvcc` -> `<root>`.
    nvcc.parent().and_then(|bin| bin.parent()).map(Path::to_path_buf)
}

/// The directory holding MSVC's `cl.exe`, for nvcc's `-ccbin`.
///
/// Located through `vswhere` rather than by walking Program Files, because the
/// toolset version in the path changes with every Visual Studio update.
fn find_msvc_cl() -> Option<PathBuf> {
    let vswhere = std::env::var("ProgramFiles(x86)")
        .ok()
        .map(|root| {
            Path::new(&root)
                .join("Microsoft Visual Studio")
                .join("Installer")
                .join("vswhere.exe")
        })
        .filter(|path| path.is_file())?;

    let output = Command::new(vswhere)
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
        .map(|line| PathBuf::from(line.trim()))
        .filter(|path| path.is_file())
        .collect();
    candidates.sort();

    // The toolset version is a path component, so the highest sorts last.
    candidates
        .pop()
        .and_then(|cl| cl.parent().map(Path::to_path_buf))
}
