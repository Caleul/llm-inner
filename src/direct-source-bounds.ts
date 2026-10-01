import { createHash } from "node:crypto";

export interface SourceBounds { minimum:number;maximum:number;key:string;opaque?:boolean }
/** Streaming arithmetic reduction over the emitted scalar source. Only an
 * operator stack and operand ranges are held; no AST, nodes, model program,
 * activation values or source fragments are constructed or retained.
 * Operations are evaluated in Rust's original binary64 order.
 */
export class DirectSourceBounds {
  private token="";
  private field="";
  private fieldClosed=false;
  private expectOperand=true;
  private readonly operands:SourceBounds[]=[];
  private readonly operators:string[]=[];
  constructor(private readonly resolve:(name:string)=>SourceBounds){}
  accept(source:string):void{
    for(const char of source){
      if(this.field){
        if(this.fieldClosed&&char!=="["){
          this.push(this.resolve(this.field));this.field="";this.fieldClosed=false;
        }else{this.field+=char;this.fieldClosed=char==="]";continue;}
      }
      if(/[A-Za-z0-9_.:]/.test(char)||((char==="+"||char==="-")&&/[eE]$/.test(this.token))){
        this.token+=char;if(this.token.length>128)throw new Error("Invalid scalar token");continue;
      }
      if(char==="["&&/^[a-z][a-z0-9_]*$/.test(this.token)){this.field=this.token+char;this.token="";continue;}
      this.flushToken();
      if(/\s/.test(char))continue;
      if(char==="("){this.operators.push(char);this.expectOperand=true;continue;}
      if(char===")"){
        while(this.operators.length&&this.operators.at(-1)!=="(")this.reduce();
        if(this.operators.pop()!=="(")throw new Error("Unbalanced scalar arithmetic");
        this.expectOperand=false;continue;
      }
      if(!"+-*/".includes(char))throw new Error(`Unsupported scalar source token ${char}`);
      const op=this.expectOperand?(char==="-"?"neg":char==="+"?"pos":"invalid"):char;
      if(op==="invalid")throw new Error("Invalid scalar operator");
      while(this.operators.length&&this.operators.at(-1)!=="("&&
        (precedence(this.operators.at(-1)!)>precedence(op)||
          (precedence(this.operators.at(-1)!)===precedence(op)&&op!=="neg"&&op!=="pos")))this.reduce();
      this.operators.push(op);this.expectOperand=true;
    }
  }
  finish():SourceBounds{
    this.flushToken();if(this.field){
      if(!this.fieldClosed)throw new Error("Incomplete scalar input coordinate");
      this.push(this.resolve(this.field));this.field="";
    }
    while(this.operators.length)this.reduce();
    if(this.operands.length!==1)throw new Error("Incomplete scalar arithmetic");
    return this.operands[0]!;
  }
  private push(value:SourceBounds):void{this.operands.push(value);this.expectOperand=false;}
  private flushToken():void{
    if(!this.token)return;const token=this.token;this.token="";
    const number=Number(token.replace(/_f64$/, ""));
    if(Number.isFinite(number)){this.push({minimum:number,maximum:number,key:key("number",String(number))});return;}
    if(/^[a-z][a-z0-9_]*$/.test(token)){this.push(this.resolve(token));return;}
    throw new Error(`Unsupported scalar token ${token}`);
  }
  private reduce():void{
    const op=this.operators.pop()!,b=this.operands.pop();
    if(!b||op==="(")throw new Error("Invalid scalar reduction");
    if(op==="neg"||op==="pos"){
      this.push(op==="pos"?b:{minimum:-b.maximum,maximum:-b.minimum,key:key(op,b.key),...(b.opaque?{opaque:true}:{})});return;
    }
    const a=this.operands.pop();if(!a)throw new Error("Missing scalar operand");
    const identity=key(op,a.key,b.key);
    if(a.opaque||b.opaque){this.push({minimum:-Infinity,maximum:Infinity,key:identity,opaque:true});return;}
    if(op==="-"&&a.key===b.key){this.push({minimum:0,maximum:0,key:identity});return;}
    if(op==="*"&&a.key===b.key){
      const minimum=a.minimum<=0&&a.maximum>=0?0:Math.min(a.minimum*a.minimum,a.maximum*a.maximum);
      const maximum=Math.max(a.minimum*a.minimum,a.maximum*a.maximum);
      this.push({minimum,maximum,key:identity,...(!Number.isFinite(maximum)?{opaque:true}:{})});return;
    }
    if(op==="/"&&b.minimum<=0&&b.maximum>=0){this.push({minimum:-Infinity,maximum:Infinity,key:identity,opaque:true});return;}
    const apply=(x:number,y:number)=>op==="+"?x+y:op==="-"?x-y:op==="*"?x*y:x/y;
    const values=[apply(a.minimum,b.minimum),apply(a.minimum,b.maximum),apply(a.maximum,b.minimum),apply(a.maximum,b.maximum)];
    this.push({minimum:Math.min(...values),maximum:Math.max(...values),key:identity,
      ...(values.some(x=>!Number.isFinite(x))?{opaque:true}:{})});
  }
}
function precedence(op:string):number{return op==="neg"||op==="pos"?3:op==="*"||op==="/"?2:1;}
function key(...values:string[]):string{return createHash("sha256").update(values.join("|")).digest("hex");}
