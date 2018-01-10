#[macro_use]
extern crate lazy_static;
extern crate libc;
extern crate rand;
extern crate structopt;
#[macro_use]
extern crate structopt_derive;

mod args;
mod preprocess;
mod train;
mod output;
mod buffer;

fn main() {
    lazy_static::initialize(&args::ARGS);
    preprocess::preprocess();
    train::train();
    output::output();
}
