include!("/content/llm-inner-sympy-scalar-locals-2g/next-token.rs");
fn main() {
 let data=std::fs::read("/content/llm-inner-sympy-scalar-locals-2g/arm64-silu-reference.bin").unwrap();let mut cases=0;let mut mismatches=0;
 for bytes in data.chunks_exact(16) {
  let x=f64::from_le_bytes(bytes[0..8].try_into().unwrap()); let expected=f64::from_le_bytes(bytes[8..16].try_into().unwrap());
  let actual=activation_half(x);
  if actual.to_bits()!=expected.to_bits() {if mismatches<5 {println!("mismatch {} {:x} {:x}",x,actual.to_bits(),expected.to_bits());}mismatches+=1;}
  cases+=1;
 }
 println!("ARM64 activation cases={} mismatches={}",cases,mismatches);
 if mismatches!=0 {std::process::exit(1);}
}