fn main() {
    match raw_decode::decode("/nonexistent/file.dng") {
        Err(e) => println!("libraw runtime OK, propagated error: {e}"),
        Ok(_) => println!("unexpected success"),
    }
}
