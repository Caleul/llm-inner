include!("/content/llm-inner-sympy-scalar-locals-2g/next-token.rs");
fn main() {
 let data=std::fs::read("/content/llm-inner-sympy-scalar-locals-2g/arm64-exp-reference.bin").unwrap();let mut cases=0;let mut mismatches=0;
 for (i,bytes) in data.chunks_exact(8).enumerate() {
  let x=-(i as f64)/16777216.0; let expected=f64::from_le_bytes(bytes.try_into().unwrap());
  let actual=exponential_small_negative(x);
  if actual.to_bits()!=expected.to_bits() {if mismatches<5 {println!("mismatch {} {:x} {:x}",x,actual.to_bits(),expected.to_bits());}mismatches+=1;}
  cases+=1;
 }
 println!("ARM64 exponential cases={} mismatches={}",cases,mismatches);
 if mismatches!=0 {std::process::exit(1);}
}