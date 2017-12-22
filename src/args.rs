#[allow(non_snake_case)] // annoyingly, needed only for K, but has to be enabled for entire struct
pub struct Args {
  pub alpha: f32,
  pub beta: f32,
  pub K: u32,
  pub n_mc: u32,
  pub seed: u32,
  pub buffer_size: usize,
  pub buffer_max_docs: usize,
  pub input: String,
  pub output: String,
  pub z_temp_file: String,
  pub w_temp_file: String,
  pub d_temp_file: String,
  pub c_temp_file: String,
}

lazy_static! { pub static ref ARGS: Args = parse(); }

pub fn parse() -> Args {
  Args {
    alpha: 0.1,
    beta: 0.1,
    K: 10,
    n_mc: 100,
    seed: 0,
    buffer_size: 1024,
    buffer_max_docs: 32,
    input: "data/small.txt".to_string(),
    output: "output/small.txt".to_string(),
    z_temp_file: "temp/z.bin".to_string(),
    w_temp_file: "temp/w.bin".to_string(),
    d_temp_file: "temp/d.bin".to_string(),
    c_temp_file: "temp/c.txt".to_string(),
  }
}
