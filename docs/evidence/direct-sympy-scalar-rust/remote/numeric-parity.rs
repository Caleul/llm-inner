include!("/content/llm-inner-sympy-scalar-locals-2g/next-token.rs");

fn main() {
 let mut failures=0u64;
 let mut checked=0u64;
 for word in 0x3f800000u32..0x40800000u32 {
  let x=f32::from_bits(word);
  let actual=root_binary32(x as f64); let wanted=x.sqrt() as f64;
  if actual.to_bits()!=wanted.to_bits() { if failures<3 { println!("root mismatch {:x} {:x} {:x}",word,actual.to_bits(),wanted.to_bits()); } failures+=1; }
  checked+=1;
 }
 println!("root cases={} mismatches={}",checked,failures);
 let mut state=0x123456789abcdef0u64; let before=failures;
 for _ in 0..1000000 {
  state=state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
  let bits=state & !(1u64<<52); let x=f64::from_bits(bits);
  if !x.is_finite() {continue;}
  let actual=round_binary32(x); let wanted=(x as f32) as f64;
  if actual.to_bits()!=wanted.to_bits() { failures+=1; }
 }
 println!("round32 random-cases=1000000 mismatches={}",failures-before);
 for (name,kind) in [("exp-gold.bin",0),("silu-gold.bin",1)] {
  let bytes=std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(name)).unwrap();
  let mut count=0;let before=failures;
  for row in bytes.chunks_exact(16) {
   let x=f64::from_le_bytes(row[0..8].try_into().unwrap());let expected=f64::from_le_bytes(row[8..16].try_into().unwrap());
   let value=if kind==0 {exponential_small_negative(x)} else {activation_half(x)};
   if value.to_bits()!=expected.to_bits() {if failures-before<3 {println!("{} mismatch {} {:x} {:x}",name,x,value.to_bits(),expected.to_bits());}failures+=1;}
   count+=1;
  }
  println!("{} cases={} mismatches={}",name,count,failures-before);
 }
 if failures!=0 {std::process::exit(1);}
}
