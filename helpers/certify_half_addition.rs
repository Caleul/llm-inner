// Diagnostic certificate, never included in the generated model. Exhausts
// unordered finite F16 magnitudes and both sum/difference; sign symmetry
// covers all signed ordered pairs. Two independent quantizers compare direct
// F64 rounding with the native IEEE F32 boundary followed by integer F16 rounding.
fn half64(x: f64) -> u16 {
    let bits=x.to_bits();let sign=((bits>>48)&0x8000) as u16;let v=x.abs();
    if v>=65520.0 {return sign|0x7c00;}
    if v<2.0f64.powi(-14) {return sign|((v*2.0f64.powi(24)).round_ties_even() as u16);}
    let fraction=bits&0xfffffffffffff;
    let rounded=(fraction+((1u64<<41)-1)+((fraction>>42)&1))>>42;
    sign|((((bits>>52)&0x7ff)-1008)*1024+rounded) as u16
}
fn half32(x: f32) -> u16 {
    let bits=x.to_bits();let sign=((bits>>16)&0x8000) as u16;
    let exponent=(bits>>23)&255;let fraction=bits&0x7fffff;
    if exponent<102 {return sign;}
    if exponent>=143 {return sign|0x7c00;}
    let (mantissa,shift,base)=if exponent<113 {(fraction|0x800000,126-exponent,0)} else {(fraction,13,(exponent-112)*1024)};
    let rounded=(mantissa+((1u32<<(shift-1))-1)+((mantissa>>shift)&1))>>shift;
    sign|(base+rounded) as u16
}
fn main() {
    let values:Vec<f64>=(0u32..0x7c00).map(|b|if b<1024 {b as f64*2.0f64.powi(-24)}
        else {((b&1023)+1024) as f64*2.0f64.powi((b>>10) as i32-25)}).collect();
    let mut pairs=0u64;
    for a in 0..values.len() {for b in a..values.len() {
        for raw in [values[a]+values[b],values[a]-values[b]] {
            let direct=half64(raw);let reference=half32(raw as f32);
            assert_eq!(direct,reference,"a={a}, b={b}, raw={raw}");pairs+=1;
        }
    }}
    for a in [0.0f64,-0.0] {for b in [0.0f64,-0.0] {for raw in [a+b,a-b] {assert_eq!(half64(raw),half32(raw as f32));}}}
    println!("{{\"magnitudeValues\":{},\"sumAndDifferenceComparisons\":{},\"signedZeroComparisons\":8,\"mismatches\":0}}",values.len(),pairs);
}
