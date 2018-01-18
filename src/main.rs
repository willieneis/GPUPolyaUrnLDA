extern crate byteorder;
#[macro_use]
extern crate lazy_static;
extern crate libc;
extern crate memmap;
extern crate rand;
extern crate structopt;
#[macro_use]
extern crate structopt_derive;

mod args;
mod buffer;
mod ffi;
mod output;
mod preprocess;
mod train;

fn main() {
    lazy_static::initialize(&args::ARGS);
    preprocess::preprocess();
    train::train();
    output::output();
}
