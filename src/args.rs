use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(name = r" __   __            __
/ _  |__) |  | |   |  \  /\
\__) |    |__| |__ |__/ /--\
P    ó    r    a   i    l
U    l    n    t   r    l
     y         e   i    o
     a         n   c    c
               t   h    a
                   l    t
                   e    i
                   t    o
                        n
Version")]
pub struct Args {
    #[structopt(long="alpha", display_order_raw="1", help = "α parameter", default_value = "0.1")]
    pub alpha: f32,

    #[structopt(long="beta", display_order_raw="2", help = "β parameter", default_value = "0.1")]
    pub beta: f32,

    #[structopt(long="K", display_order_raw="3", help = "Number of topics", default_value = "100")]
    pub k: u32,

    #[structopt(long="buffer_size", display_order_raw="4", help = "Buffer size", default_value = "16384")]
    pub buffer_size: u32,

    #[structopt(long="max_D", display_order_raw="5", help = "Maximum number of documents per buffer", default_value = "10")]
    pub max_d: u32,

    #[structopt(long="max_N_d", display_order_raw="6", help = "Maximum document size", default_value = "1024")]
    pub max_n_d: u32,

    #[structopt(long="n_mc", display_order_raw="7", help = "Number of Monte Carlo iterations", default_value = "1")]
    pub n_mc: u32,

    #[structopt(long="seed", display_order_raw="8", help = "Random number seed", default_value = "0")]
    pub seed: u32,

    #[structopt(long="input", display_order_raw="9", help = "Input file", default_value = "data/small.txt")]
    pub input: String,

    #[structopt(long="output", display_order_raw="10", help = "Output file", default_value = "output/small.txt")]
    pub output: String,

    #[structopt(long="z_temp_file", display_order_raw="11", help = "Temporary file for z in binary format", default_value = "temp/z.bin")]
    pub z_temp_file: String,

    #[structopt(long="w_temp_file", display_order_raw="12", help = "Temporary file for w in binary format", default_value = "temp/w.bin")]
    pub w_temp_file: String,

    #[structopt(long="d_temp_file", display_order_raw="13", help = "Temporary file for d in binary format", default_value = "temp/d.bin")]
    pub d_temp_file: String,

    #[structopt(long="K_d_temp_file", display_order_raw="14", help = "Temporary file for K_d in binary format", default_value = "temp/k_d.bin")]
    pub k_d_temp_file: String,

    #[structopt(long="c_temp_file", display_order_raw="15", help = "Temporary file for C in text format", default_value = "temp/c.txt")]
    pub c_temp_file: String,
}

lazy_static! { pub static ref ARGS: Args = Args::from_args(); }
