/** Streaming lexical admission check, not a calculation representation.
 * It holds one token and comment/string state; never an expression, graph or
 * activation. Passing is necessary, not sufficient, for a reduction proof.
 */
export class IncompleteDirectReduction extends Error {
  constructor(readonly token:string){super(`Direct reduction is incomplete: ${token} remains in the expression. Rust compilation is not admitted.`);}
}
export class DirectReductionGate {
  private token="";
  private quoted=false;
  private escaped=false;
  private lineComment=false;
  private blockDepth=0;
  private previous="";
  private conditionDepth=0;
  private pendingCondition=false;
  private parentheses=0;
  private conditionParentheses=0;
  private braces:boolean[]=[];
  private identifier():void{
    if(!this.token)return;
    const token=this.token;this.token="";
    if(token==="if"){
      if(this.conditionDepth>0)throw new IncompleteDirectReduction("nested condition requiring propagation/flattening");
      this.conditionDepth++;this.pendingCondition=true;this.conditionParentheses=this.parentheses;
    }
    if(["let","mut","for","while","loop","as"].includes(token)||
      /^(?:round_|half_encode_|half_decode_|sqrt_|exp_|silu_)/.test(token)||
      ["f16","f32","f16Bits","f32Bits","ropeBits","Math","exp","sqrt","floor","powi","round","round_ties_even","from_bits"].includes(token))throw new IncompleteDirectReduction(token);
  }
  accept(source:string):void{
    for(const c of source){
      if(this.lineComment){if(c==="\n")this.lineComment=false;this.previous=c;continue;}
      if(this.blockDepth){
        if(this.previous==="/"&&c==="*"){this.blockDepth++;this.previous="";continue;}
        if(this.previous==="*"&&c==="/"){this.blockDepth--;this.previous="";continue;}
        this.previous=c;continue;
      }
      if(this.quoted){if(this.escaped)this.escaped=false;else if(c==="\\")this.escaped=true;else if(c==='"')this.quoted=false;this.previous=c;continue;}
      if(this.previous==="/"&&c==="/"){this.identifier();this.lineComment=true;this.previous=c;continue;}
      if(this.previous==="/"&&c==="*"){this.identifier();this.blockDepth=1;this.previous="";continue;}
      if(c==='"'){this.identifier();this.quoted=true;this.previous=c;continue;}
      if(/[A-Za-z0-9_]/.test(c)){this.token+=c;if(this.token.length>1024)throw new Error("Invalid source token");}
      else {
        this.identifier();
        if(c==="(")this.parentheses++;
        else if(c===")")this.parentheses--;
        else if(c==="{"){
          const body=this.pendingCondition&&this.parentheses===this.conditionParentheses;
          this.braces.push(body);if(body)this.pendingCondition=false;
        }
        else if(c==="}"){if(this.braces.pop())this.conditionDepth--;}
      }
      this.previous=c;
    }
  }
  finish():void{this.identifier();if(this.quoted||this.blockDepth)throw new Error("Incomplete source lexical context");}
}
