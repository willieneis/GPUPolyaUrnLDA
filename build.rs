fn main() {
    std::process::Command::new("make")
        .arg("RELEASE=1")
        .arg("target/cuda/libGPUPolyaUrnLDA.so")
        .spawn()
        .expect("Failed to make");
    println!("cargo:rustc-link-search=target/cuda/");
}
