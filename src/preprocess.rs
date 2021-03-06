use args::ARGS;
use std::fs::{File, remove_file};
use std::io::prelude::{BufRead, Write};
use std::io::{BufReader, BufWriter};
use rand::{thread_rng};
use rand::distributions::{Range, IndependentSample};
use std::slice;
use std::mem;
use std::collections::HashMap;


fn get_tokens(line: String) -> Vec<String> {
    line.split("\t").skip(2).next().unwrap().split_whitespace().map(|s| s.to_string()).collect()
}

fn count_tokens() -> HashMap<String, u32> {
    let mut count: HashMap<String, u32> = HashMap::new();
    let input = BufReader::new(File::open(&ARGS.input).unwrap());
    for line in input.lines() {
        get_tokens(line.unwrap()).iter().fold((),|_,token| *count.entry(token.clone()).or_insert(0) += 1);
    }
    let mut count_vec: Vec<(String, u32)> = count.iter().map(|t| (t.0.clone(),t.1.clone())).collect();
    count_vec.sort_by(|t1,t2| t2.1.cmp(&t1.1));
    let mut c = BufWriter::new(File::create(&ARGS.c_temp_file).unwrap());
    let mut token_ids: HashMap<String, u32> = HashMap::new();
    for (idx,&(ref token,count)) in count_vec.iter().enumerate() {
        write!(&mut c, "{}\t{}\t{}\n", token, idx, count).unwrap();
        token_ids.insert(token.clone(), idx as u32);
    }
    c.flush().unwrap();
    token_ids
}

fn as_bytes<'a>(v: &'a Vec<u32>) -> &'a [u8] {
    let s = v.as_slice();
    unsafe {
        slice::from_raw_parts(s.as_ptr() as *const u8,
                              s.len() * mem::size_of::<u32>() / mem::size_of::<u8>())
    }
}

pub fn preprocess() {
    println!("preprocessing");
    remove_file(&ARGS.z_temp_file).unwrap_or_else(|_| ());
    remove_file(&ARGS.w_temp_file).unwrap_or_else(|_| ());
    remove_file(&ARGS.d_temp_file).unwrap_or_else(|_| ());
    remove_file(&ARGS.k_d_temp_file).unwrap_or_else(|_| ());

    let token_ids = count_tokens();

    let input = BufReader::new(File::open(&ARGS.input).unwrap());
    let mut z = BufWriter::new(File::create(&ARGS.z_temp_file).unwrap());
    let mut w = BufWriter::new(File::create(&ARGS.w_temp_file).unwrap());
    let mut d = BufWriter::new(File::create(&ARGS.d_temp_file).unwrap());
    let mut k_d = BufWriter::new(File::create(&ARGS.k_d_temp_file).unwrap());

    let mut rand = thread_rng();
    let unif = Range::new(0, ARGS.k);

    for line in input.lines() {
        let tokens = get_tokens(line.unwrap());
        let mut indicators = (0..tokens.len()).map(|_| unif.ind_sample(&mut rand)).collect();
        z.write_all(as_bytes(&indicators)).unwrap();
        w.write_all(as_bytes(&tokens.iter().map(|t| *token_ids.get(t).unwrap()).collect())).unwrap();
        d.write_all(as_bytes(&vec![tokens.len() as u32])).unwrap();
        indicators.sort_unstable();
        indicators.dedup();
        k_d.write_all(as_bytes(&vec![indicators.len() as u32])).unwrap();
    }

    z.flush().unwrap(); w.flush().unwrap(); d.flush().unwrap(); k_d.flush().unwrap();
}



#[cfg(test)]
mod preprocess_test {
    use super::*;

    #[test]
    fn test_get_tokens() {
        let line = String::from("docno:1	X	hi alex and Kunal\n");
        let tokens = get_tokens(line);
        assert_eq!(4, tokens.len());
        assert_eq!("hi", tokens[0]);
        assert_eq!("Kunal", tokens[3]);
        println!("{}", tokens[0]);
    }


}
