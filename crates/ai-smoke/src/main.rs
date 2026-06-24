//! ONNX Runtime smoke test — proves local inference works with the CoreML
//! execution provider on Apple Silicon (and CPU fallback everywhere else).
//!
//!   export ORT_DYLIB_PATH="$(brew --prefix onnxruntime)/lib/libonnxruntime.dylib"
//!   ai-smoke                                  # init + provider report
//!   ai-smoke path/to/sam2_image_encoder.onnx  # + load session, run zeros
//!
//! With no model, it still exercises the real ORT load + EP registration path,
//! which is the Phase-0 de-risking goal. With a SAM2 image-encoder ONNX, it
//! also runs a 1024x1024x3 zero tensor through it and prints output shapes.

use ndarray::Array4;
use ort::execution_providers::CoreMLExecutionProvider;
use ort::session::Session;
use ort::value::Tensor;

fn main() -> ort::Result<()> {
    ort::init().with_name("aml-smoke").commit()?;
    println!("ONNX Runtime initialized (load-dynamic).");

    let model = std::env::args().nth(1);
    let Some(path) = model else {
        println!("no model given — pass a SAM2 image-encoder ONNX to run inference.");
        return Ok(());
    };

    let mut session = Session::builder()?
        .with_execution_providers([CoreMLExecutionProvider::default().build()])?
        .commit_from_file(&path)?;

    println!("loaded: {path}");
    for i in &session.inputs {
        println!("  input  {:<24} {:?}", i.name, i.input_type);
    }
    for o in &session.outputs {
        println!("  output {:<24} {:?}", o.name, o.output_type);
    }

    // SAM2 image encoder expects [1, 3, 1024, 1024] float NCHW.
    let input_name = session.inputs[0].name.clone();
    let dummy: Array4<f32> = Array4::zeros((1, 3, 1024, 1024));
    let tensor = Tensor::from_array(dummy)?;

    let t0 = std::time::Instant::now();
    let outputs = session.run(ort::inputs![input_name.as_str() => tensor])?;
    let dt = t0.elapsed();

    println!("inference ok in {dt:.2?}");
    for (name, val) in outputs.iter() {
        if let Ok((shape, _)) = val.try_extract_tensor::<f32>() {
            println!("  {name:<24} shape {shape:?}");
        }
    }
    Ok(())
}
