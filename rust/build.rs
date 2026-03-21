fn main() {
    let version = std::fs::read_to_string("../../VERSION")
        .unwrap_or_else(|_| "0.0.0".to_string())
        .trim()
        .to_string();
    println!("cargo:rustc-env=NPCSH_VERSION={}", version);
    println!("cargo:rerun-if-changed=../../VERSION");
}
