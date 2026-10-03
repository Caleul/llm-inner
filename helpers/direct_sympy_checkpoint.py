"""First-coordinate checkpoint migration to mathematical strings.

This writes compiler working expressions only. Numerical primitives must still
be lowered to elementary syntax before a final artifact can be admitted.
Safetensors scalar reads are incremental; no tensor/checkpoint is materialized.
"""
import argparse
import ast
import json
import math
import signal
from pathlib import Path
import struct
from contextlib import ExitStack

from safetensors import safe_open
from direct_sympy_strings import Domain,StringCompiler,syntax
from fractions import Fraction as F
from direct_sympy_conversions import ConversionSession,FiniteSource


def rms_half_bound(width,epsilon):
    """Conservative bound for the actual rounded normalization, not real RMS.

    With u=2^-23 (covering both F64 calculation and F32 quantization), positive
    accumulation gives sum >= (1-width*u)*real_sum by Bernoulli. sqrt/div/mul
    losses give factor <1.25 for width<=1e6. Half rounding adds at most relative
    2^-11 or absolute 2^-25. Thus both the raw product and its rounded Half
    are below 2*sqrt(width). Epsilon excludes zero denominators/underflow;
    the upper certificate also excludes variance overflow. This is a compile
    proof; it is not a runtime RMS helper or a checkpoint-size assumption.
    """
    if not 1<=width<=1000000 or not 2**-126<=epsilon<=1.7014117331926443e38:return None
    u=2**-23
    factor=(1+u)**3/((1-u)*math.sqrt(1-width*u))
    if factor>=1.25:return None
    return 2*math.sqrt(width)


def f32(x):
    return struct.unpack("f",struct.pack("f",x))[0]


class CheckpointStrings:
    def __init__(self,directory,compiler,*,lower_conversions=True):
        self.directory=Path(directory)
        self.config=json.loads((self.directory/"config.json").read_text())
        if self.config.get("model_type")!="llama":
            raise ValueError("This adapter admits Llama; other families require their own string adapter")
        self.width=self.config["hidden_size"]
        self.layers=self.config["num_hidden_layers"]
        self.compiler=compiler
        self.domains={f"X{i+1}":Domain(F(-65504),F(65504),-24,False) for i in range(self.width)}
        self.conversions=ConversionSession(compiler,self.domains,input_dtype="f16") if lower_conversions else None
        self.memo={}
        self.read_weights=0
        self.sources=[]
        self.stack=[]
        self.events=[]
        self.on_completed=None

    def __enter__(self):
        for path in sorted(self.directory.glob("*.safetensors")):
            handle=safe_open(path,framework="pt",device="cpu")
            handle.__enter__()
            self.sources.append(handle)
        if not self.sources:
            raise ValueError("No Safetensors files")
        return self

    def __exit__(self,*args):
        for handle in reversed(self.sources):handle.__exit__(*args)

    def weight(self,name,row,column=None):
        matches=[h for h in self.sources if name in h.keys()]
        if len(matches)!=1:raise ValueError("Missing or duplicated checkpoint weight: "+name)
        tensor=matches[0].get_slice(name)
        if tensor.get_dtype()!="F16":raise ValueError("Current numerical admission requires F16 weights")
        scalar=tensor[row:row+1] if column is None else tensor[row:row+1,column:column+1]
        self.read_weights+=1
        return repr(float(scalar.item()))

    def shape(self,name):
        for handle in self.sources:
            if name in handle.keys():return handle.get_slice(name).get_shape()
        raise ValueError("Missing weight: "+name)

    def op(self,operation,a,b):
        # Actual F32 evaluation order is retained in the mathematical syntax.
        if operation=="*" and a==b and self.conversions is not None:
            operand=syntax(a)
            if self.conversions.value_kind(operand)=="half" and self.conversions.bounds(operand) is not None:
                # Finite Half squares are exact F32 products. Do not build
                # the redundant cast and reparse the entire closed operand
                # to prove its removal a second time. SymPy still processes
                # the substitution before the next dependency is requested.
                return self.compiler.substitute("(X999999998 ** 2)","X999999998",a,self.domains)
        template="R32(X999999998 "+operation+" X999999999)"
        left=self.compiler.substitute(template,"X999999998",a,self.domains)
        return self.compiler.substitute(left,"X999999999",b,self.domains)

    def producer(self,key,build):
        if key in self.memo:return self.memo[key]
        before=len(self.compiler.events)
        expression=build()
        result=self.compiler.stabilize("("+expression+")",self.domains)
        if self.conversions is not None:result=self.conversions.close(result)
        self.compiler.register_completed_region(result,self.domains,word_closed=self.conversions is not None)
        self.memo[key]=result
        self.events.append((key,len(expression),len(result),len(self.compiler.events)-before))
        if self.on_completed is not None:self.on_completed(self)
        return result

    def reduction(self,terms):
        lanes=[]
        for lane in range(4):
            value="0.0"
            for term in terms[lane::4]:value=self.op("+",value,term)
            lanes.append(value)
        return "R16("+self.op("+",self.op("+",lanes[0],lanes[1]),self.op("+",lanes[2],lanes[3]))+")"

    def linear(self,name,row,input_value):
        shape=self.shape(name)
        if len(shape)!=2 or not 0<=row<shape[0]:raise ValueError("Invalid projection coordinate")
        terms=[]
        for column in range(shape[1]):
            coefficient=self.weight(name,row,column)
            # +0 lane accumulation makes a signed zero product irrelevant.
            # Read the weight before asking for an irrelevant producer.
            if float(coefficient)==0:terms.append("0.0")
            else:
                operand=input_value(column)
                terms.append(operand if float(coefficient)==1 else self.op("*",operand,coefficient))
        return self.reduction(terms)

    def rms_sum(self,input_value):
        width=self.width
        vector=width>=4
        rows=width//4 if vector else width
        groups=rows//4
        step=2**max(4,math.floor(math.ceil(math.log2(max(groups,1)))/4))
        def load(row,lane):
            x=input_value(row*4+lane if vector else row)
            return self.op("*",x,x)
        def component(lane):
            def partial(part):
                def block(level,start):
                    if level==0:return load(part+4*start,lane)
                    span=step**(level-1)
                    value=block(level-1,start)
                    for i in range(1,step):value=self.op("+",value,block(level-1,start+i*span))
                    return value
                value=None
                for level in range(4):
                    span=step**level;end=groups//span*span
                    start=0 if level==3 else groups//(span*step)*span*step
                    accumulator=None
                    for i in range(start,end,span):
                        x=block(level,i);accumulator=x if accumulator is None else self.op("+",accumulator,x)
                    if accumulator is not None:value=accumulator if value is None else self.op("+",value,accumulator)
                if part==0:
                    for i in range(4*groups,rows):
                        x=load(i,lane);value=x if value is None else self.op("+",value,x)
                return "0.0" if value is None else value
            value=partial(0)
            if groups:
                for part in range(1,4):value=self.op("+",value,partial(part))
            return value
        if not vector:return component(0)
        value=None
        for i in range(rows*4,width):
            x=input_value(i);square=self.op("*",x,x)
            value=square if value is None else self.op("+",value,square)
        for lane in range(4):
            x=component(lane);value=x if value is None else self.op("+",value,x)
        return value

    def norm(self,key,name,coordinate,input_value):
        def build():
            epsilon=repr(f32(self.config["rms_norm_eps"]))
            inverse=self.producer(key+":inverse",lambda:self.op("/","1.0",
                "R32(sqrt("+self.op("+",self.op("/",self.rms_sum(input_value),str(self.width)),epsilon)+"))"))
            product=self.op("*",input_value(coordinate),inverse)
            normalized="R16("+product+")"
            if self.conversions is not None:
                bound=rms_half_bound(self.width,f32(self.config["rms_norm_eps"]))
                # The correlation proof assumes finite Half operands. A
                # later residual may overflow even when the original inputs
                # are finite; never carry the bound across that frontier.
                finite_half=bound is not None and all(
                    self.conversions.value_kind(syntax(input_value(i)))=="half"
                    and self.conversions.bounds(syntax(input_value(i))) is not None
                    for i in range(self.width))
                if finite_half:
                    raw=syntax(product)
                    for node in (raw,raw.args[0],syntax(normalized)):
                        existing=self.conversions.bounds(node)
                        q=existing.quantum if existing is not None else (-149 if node is raw else -1074)
                        if node is not raw and isinstance(node,ast.Call) and node.func.id=="R16":q=-24
                        self.conversions.completed[self.conversions.key(node)]=FiniteSource(-bound,bound,q)
            return "R16("+self.op("*",normalized,self.weight(name,coordinate))+")"
        return self.producer(key+":"+str(coordinate),build)

    def gated(self,prefix,neuron,input_value):
        # Each scalar projection owns its rounding frontier. Close it before
        # substituting it into the activation/product; composing both raw
        # reductions first postpones numerical simplification incorrectly.
        gate=self.producer(prefix+"gate:"+str(neuron),lambda:
            self.linear(prefix+"mlp.gate_proj.weight",neuron,input_value))
        up=self.producer(prefix+"up:"+str(neuron),lambda:
            self.linear(prefix+"mlp.up_proj.weight",neuron,input_value))
        return "R16("+self.op("*","Silu16("+gate+")",up)+")"

    def hidden(self,layer,coordinate):
        if layer<0:return f"X{coordinate+1}"
        prefix=f"model.layers.{layer}."
        def pre(column):return self.norm(prefix+"pre",prefix+"input_layernorm.weight",column,lambda i:self.hidden(layer-1,i))
        def residual(column):
            def value(i):return self.producer(prefix+"v:"+str(i),lambda:self.linear(prefix+"self_attn.v_proj.weight",i,pre))
            heads=self.config["num_attention_heads"];kv_heads=self.config["num_key_value_heads"]
            head_dim=self.config.get("head_dim",self.width//heads)
            if heads%kv_heads:raise ValueError("Invalid grouped-query geometry")
            # Position zero has one causal key: its probability is exactly 1.
            # This is a coordinate proof, never prompt/length specialization.
            context=lambda i:self.producer(prefix+"context:"+str(i),lambda:self.reduction([
                self.op("*","1.0",value((i//head_dim//(heads//kv_heads))*head_dim+i%head_dim))]))
            attention=self.linear(prefix+"self_attn.o_proj.weight",column,context)
            return "R16("+self.op("+",self.hidden(layer-1,column),attention)+")"
        def post(column):return self.norm(prefix+"post",prefix+"post_attention_layernorm.weight",column,
            lambda i:self.producer(prefix+"residual:"+str(i),lambda:residual(i)))
        return self.producer(prefix+"hidden:"+str(coordinate),lambda:
            "R16("+self.op("+",self.producer(prefix+"residual:"+str(coordinate),lambda:residual(coordinate)),
                self.linear(prefix+"mlp.down_proj.weight",coordinate,
                    lambda i:self.producer(prefix+"gated:"+str(i),lambda:self.gated(prefix,i,post))))+")")

    def coordinate(self,dimension):
        if self.config.get("attention_bias") or self.config.get("mlp_bias"):
            raise ValueError("Biased Llama requires a string adapter extension")
        name="model.embed_tokens.weight" if self.config.get("tie_word_embeddings") else "lm_head.weight"
        return self.producer("output:0:"+str(dimension),lambda:self.linear(name,dimension,
            lambda i:self.norm("final","model.norm.weight",i,lambda j:self.hidden(self.layers-1,j))))


def main():
    parser=argparse.ArgumentParser(description="Compile one Llama working coordinate as mathematical strings with SymPy")
    parser.add_argument("checkpoint");parser.add_argument("output")
    parser.add_argument("--dimension",type=int,default=2)
    parser.add_argument("--max-characters",type=int,default=1048576)
    parser.add_argument("--max-seconds",type=int,default=60)
    parser.add_argument("--reference-boundaries",action="store_true",help="Keep numerical primitives for reference-string validation only")
    parser.add_argument("--savepoint-directory")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    if args.max_characters<1 or args.max_seconds<1:parser.error("Resource budgets must be positive")
    if args.resume and not args.savepoint_directory:parser.error("--resume requires --savepoint-directory")
    def timeout(*_):raise TimeoutError("Compilation wall-clock budget exceeded")
    signal.signal(signal.SIGALRM,timeout);signal.alarm(args.max_seconds)
    compiler=StringCompiler(max_characters=args.max_characters)
    with ExitStack() as resources:
        model=resources.enter_context(CheckpointStrings(args.checkpoint,compiler,lower_conversions=not args.reference_boundaries))
        if args.savepoint_directory:
            from direct_sympy_savepoints import ProducerSavepoints
            store=resources.enter_context(ProducerSavepoints(args.savepoint_directory,model,args.dimension))
            if args.resume:print(f"Restored completed dependencies: {store.restore(model)}")
            elif (store.directory/'frontier.json').exists():raise ValueError("Existing savepoint requires --resume; refusing to overwrite")
            model.on_completed=store.save
        try:expression=model.coordinate(args.dimension)
        except (ValueError,TimeoutError) as error:
            signal.alarm(0)
            path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
            Path(str(path)+".growth.tsv").write_text("dependency\tbeforeCharacters\tafterCharacters\tCASPasses\n"+
                "\n".join("\t".join(map(str,e)) for e in model.events)+"\n")
            Path(str(path)+".substitutions.tsv").write_text("variable\ttemplateCharacters\treplacementCharacters\toccurrences\testimatedCharacters\tfinalCharacters\tstatus\n"+
                "\n".join("\t".join(map(str,e)) for e in compiler.substitution_events)+"\n")
            Path(str(path)+".budget-simplifications.tsv").write_text("virtualCharacters\tstabilizedVirtualCharacters\texpandedCharacters\tcompletedRegions\tCASPasses\tadmitted\n"+
                "\n".join("\t".join(map(str,e)) for e in compiler.budget_events)+"\n")
            if compiler.failed_substitution is not None:
                template,name,replacement=compiler.failed_substitution
                Path(str(path)+".failed-template.work.expr").write_text(template+"\n")
                Path(str(path)+".failed-replacement.work.expr").write_text(replacement+"\n")
            last=next(reversed(model.memo),None)
            if last is not None:Path(str(path)+".prefix.work.expr").write_text(model.memo[last]+"\n")
            print(f"Compilation stopped: {error}; completedDependencies={len(model.events)} lastDependency={last} closedConversions={model.conversions.closed if model.conversions else 0} redundantConversions={model.conversions.redundant if model.conversions else 0} reusedConvertedRegions={model.conversions.reused_regions if model.conversions else 0} visitedConversionNodes={model.conversions.visited_nodes if model.conversions else 0}; no coordinate artifact admitted")
            return 1
        signal.alarm(0)
        path=Path(args.output)
        path.parent.mkdir(parents=True,exist_ok=True)
        # Working-expression suffix prevents mistaking residual numerical
        # primitives for a fully lowered final artifact.
        working=Path(str(path)+".work.expr")
        working.write_text(expression+"\n")
        growth=Path(str(path)+".growth.tsv")
        growth.write_text("dependency\tbeforeCharacters\tafterCharacters\tCASPasses\n"+
            "\n".join("\t".join(map(str,e)) for e in model.events)+"\n")
        print(f"SymPy working coordinate: position=0 dimension={args.dimension} characters={len(expression)} weightReads={model.read_weights} CASPasses={len(compiler.events)}")
        print(f"Working string: {working}; no final artifact admitted: R32/R16/sqrt/Silu16 lowering and emitted-artifact parity remain pending")
        return 2


if __name__=="__main__":
    raise SystemExit(main())
