use std::env;
use std::path::PathBuf;

fn main() {
    // Locate Homebrew (Apple Silicon) or a LIBRAW_DIR override.
    let prefix = env::var("LIBRAW_DIR")
        .ok()
        .or_else(|| brew_prefix("libraw"))
        .unwrap_or_else(|| "/usr/local".into());
    let inc = PathBuf::from(&prefix).join("include");
    let lib = PathBuf::from(&prefix).join("lib");

    cc::Build::new()
        .file("csrc/shim.c")
        .include(&inc)
        .warnings(false)
        .compile("aml_raw_shim");

    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=dylib=raw");
    println!("cargo:rerun-if-changed=csrc/shim.c");
    println!("cargo:rerun-if-env-changed=LIBRAW_DIR");
}

fn brew_prefix(pkg: &str) -> Option<String> {
    let out = std::process::Command::new("brew")
        .args(["--prefix", pkg])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}
