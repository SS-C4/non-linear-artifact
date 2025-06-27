use quantized::quantized::Quantized;

#[test]
fn main(a: Quantized, b: Quantized) -> pub Quantized {
    let c = a / b;
    c + b
}