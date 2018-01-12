fn main() {
    std::process::Command::new("make")
        .arg("target/cuda/libGPUPolyaUrnLDA.a")
        .spawn()
        .expect("Failed to make");
    println!("cargo:rustc-link-search=target/cuda/");
}
