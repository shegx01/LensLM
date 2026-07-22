use std::path::PathBuf;
fn main() {
    let data_dir = PathBuf::from("/data");
    let opt: Option<PathBuf> = None;
    let to = opt.map(|p| p).unwrap_or(data_dir.clone());
    println!("{}", to.display());
}
