use std::io::prelude::*;
use std::fs::{File, OpenOptions};
use std::path::Path;

pub struct Logger {
    pub log_file: File,
}

impl Logger {

    /// Log file will be opened for writing in append mode, if it does not
    /// exist, it will be created.
    pub fn new(log_file_path:&str) -> Logger {
        let file_path = Path::new(log_file_path);
        let log_file = OpenOptions::new()
            .read(true)
            .write(true)
            .append(true)
            .open(file_path)
            .unwrap_or(File::create(file_path).unwrap());

        Logger{ log_file }
    }

    pub fn write(&mut self, msg: &str) -> () {

        self.log_file.write(msg.as_bytes()).unwrap();
    }


}