"""First-coordinate checkpoint migration to mathematical strings.

This writes compiler working expressions only. Numerical primitives must still
be lowered to elementary syntax before a final artifact can be admitted.
Safetensors scalar reads are incremental; no tensor/checkpoint is materialized.
"""
import argparse
import ast
import json
import math
import re
import signal
from pathlib import Path
import struct
from contextlib import ExitStack

from safetensors import safe_open
from direct_sympy_strings import Domain,StringCompiler,syntax
from fractions import Fraction as F
from direct_sympy_conversions import ConversionSession,FiniteSource


def dominant_half_rms_source(sources,epsilon):
    """Certify the original F32 mean equals one exact Half square / width.

    Widths one/two use the original scalar reduction. Every Half square
    and its division by 1/2 is exact normal F32. Strict smaller-neighbor
    cells prove both the other square and epsilon invisible, including
    binade boundaries. Unknown, zero-crossing and wider reductions fail.
    """
    width=len(sources)
    if width not in (1,2) or not math.isfinite(epsilon) or epsilon<=0:return None
    if any(source is None or max(abs(source.minimum),abs(source.maximum))>65504 for source in sources):return None
    def radius(value):
        exponent=value.numerator.bit_length()-value.denominator.bit_length()
        if value<F(2)**exponent:exponent-=1
        return F(2)**max(-150,exponent-25)
    for index,source in enumerate(sources):
        minimum=max(source.minimum_magnitude,source.minimum if source.minimum>0 else -source.maximum if source.maximum<0 else 0)
        if not minimum:continue
        square=F(minimum)**2
        other=sum((F(max(abs(own.minimum),abs(own.maximum)))**2 for i,own in enumerate(sources) if i!=index),F(0))
        if other<radius(square) and F(epsilon)<radius(square/width):return index
    return None


def rms_half_bound(width,epsilon):
    """Conservative bound for the actual rounded normalization, not real RMS.

    With u=2^-23 (covering both F64 calculation and F32 quantization), positive
    accumulation gives sum >= (1-width*u)*real_sum by Bernoulli. sqrt/div/mul
    losses are enclosed by three (1+u) factors and three reciprocal (1-u)
    factors, conservatively covering variance division/addition, sqrt,
    reciprocal and multiplication. Half rounding adds at most relative
    2^-11 or absolute 2^-25. Epsilon excludes zero denominators/underflow;
    the upper certificate also excludes variance overflow. This is a compile
    proof; it is not a runtime RMS helper or a checkpoint-size assumption.
    """
    if not 1<=width<=1000000 or not 2**-126<=epsilon<=1.7014117331926443e38:return None
    u=2**-23
    factor=(1+u)**3/((1-u)**3*math.sqrt(1-width*u))
    if factor>=1.25:return None
    return math.nextafter(math.sqrt(width)*factor*(1+2**-11)+2**-25,math.inf)


def rms_component_enclosures(sources,coordinate,epsilon):
    """Correlated enclosures of the ordered F32 product and stored Half.

    Caller certifies finite Half operands. For absolute component t and
    other squared magnitudes B, real RMS gives n*t²/(t²+B+n*epsilon).
    This ratio increases with t and decreases with B. A conservative
    F32 loss factor encloses the original reduction/division/sqrt/inverse/
    product; the ratio is a proof only, never a replacement computation.
    Fraction comparisons direct every endpoint outward, including sqrt.
    """
    width=len(sources)
    if rms_half_bound(width,epsilon) is None or not 0<=coordinate<width:return None
    if any(max(abs(d.minimum),abs(d.maximum))>65504 for d in sources):return None
    def directed(q,up):
        value=float(q)
        if (F(value)<q if up else F(value)>q):value=math.nextafter(value,math.inf if up else -math.inf)
        return value
    def sqrt_bounds(q):
        value=math.sqrt(float(q));low=high=value
        while F(low)*F(low)>q:low=math.nextafter(low,-math.inf)
        while F(high)*F(high)<q:high=math.nextafter(high,math.inf)
        return low,high
    def magnitudes(d):
        small=max(d.minimum_magnitude,d.minimum if d.minimum>0 else -d.maximum if d.maximum<0 else 0)
        return F(small),F(max(abs(d.minimum),abs(d.maximum)))
    magnitudes_all=[magnitudes(d) for d in sources]
    minimum,maximum=magnitudes_all[coordinate]
    lower_sum=sum(a*a for i,(a,b) in enumerate(magnitudes_all) if i!=coordinate)
    upper_sum=sum(b*b for i,(a,b) in enumerate(magnitudes_all) if i!=coordinate)
    eps=width*F(epsilon)
    real_low=sqrt_bounds(width*minimum*minimum/(minimum*minimum+upper_sum+eps))[0]
    real_high=sqrt_bounds(width*maximum*maximum/(maximum*maximum+lower_sum+eps))[1]
    u=F(1,2**23);accumulation=1-width*u
    root_low=sqrt_bounds(accumulation)[0]
    loss=directed((1+u)**3/((1-u)**3*F(root_low)),True)
    low=directed(F(real_low)/F(loss),False)
    high=directed(F(real_high)*F(loss),True)
    d=sources[coordinate]
    # A numerical zero endpoint admits both IEEE zero signs. Keep those
    # signs in the enclosure without inventing nonzero values of the
    # opposite sign; endpoint Half payloads then prevent unsafe folding.
    signed=(low if low else -0.0,high) if d.minimum>=0 else (-high,-low if low else 0.0) if d.maximum<=0 else (-high,high)
    cast=lambda value:struct.unpack('e',struct.pack('e',value))[0]
    stored=tuple(map(cast,signed))
    return (signed,low),(stored,abs(cast(low)))


def f32(x):
    return struct.unpack("f",struct.pack("f",x))[0]


def write_expression(path,expression):
    """Preserve literal bytes and trailing newline without a full-size copy."""
    with Path(path).open('w',encoding='utf-8') as stream:
        for offset in range(0,len(expression),1024*1024):
            stream.write(expression[offset:offset+1024*1024])
        stream.write('\n')


class CheckpointStrings:
    def __init__(self,directory,compiler,*,lower_conversions=True,parallel_budget=None,input_domains=None):
        self.directory=Path(directory)
        self.config=json.loads((self.directory/"config.json").read_text())
        if self.config.get("model_type")!="llama":
            raise ValueError("This adapter admits Llama; other families require their own string adapter")
        self.width=self.config["hidden_size"]
        self.layers=self.config["num_hidden_layers"]
        self.compiler=compiler
        self.domains={f"X{i+1}":Domain(F(-65504),F(65504),-24,False) for i in range(self.width)}
        if input_domains is not None:
            from direct_sympy_input_partitions import validate_domains
            self.domains=validate_domains(input_domains,self.domains)
        self.conversions=ConversionSession(compiler,self.domains,input_dtype="f16") if lower_conversions else None
        self.memo={}
        self.read_weights=0
        self.sources=[]
        self.stack=[]
        self.events=[]
        self.on_completed=None
        self.layer_bound_cache={}
        self.norm_bound_cache={}
        self.norm_vectors={}
        self.constant_projections=0
        self.elided_updates=[]
        self.early_half_products=0
        self.parallel_budget=parallel_budget
        self.parallel_events=[]
        self.rms_constant_components=0
        self.rms_dominant_squares=0

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
        exact_half_product=False
        if operation=="*" and self.conversions is not None:
            operands=[self.conversions.analyze_expression(value)[0] for value in (a,b)]
            exact_half_product=all(self.conversions.value_kind(operand)=="half" and self.conversions.bounds(operand) is not None for operand in operands)
            if exact_half_product and a==b:
                # Finite Half squares are exact F32 products. Do not build
                # the redundant cast and reparse the entire closed operand
                # to prove its removal a second time. SymPy still processes
                # the substitution before the next dependency is requested.
                return self.compiler.substitute("(X999999998 ** 2)","X999999998",a,self.domains)
        # Two finite Half significands use at most 22 product bits and the
        # nonzero exponent range is [-48,32]. Their widened product is exact
        # in F32, including signed zeros. Drop this proven boundary before
        # literal substitution, instead of constructing it for close to remove.
        template="(X999999998 * X999999999)" if exact_half_product else "R32(X999999998 "+operation+" X999999999)"
        left=self.compiler.substitute(template,"X999999998",a,self.domains)
        result=self.compiler.substitute(left,"X999999999",b,self.domains)
        if exact_half_product:self.early_half_products+=1
        return result

    def producer(self,key,build):
        if key in self.memo:return self.memo[key]
        before=len(self.compiler.events)
        expression=build()
        result=self.compiler.stabilize("("+expression+")",self.domains)
        if self.conversions is not None:
            result=self.conversions.close(result)
            if re.search(r"\b(?:R16|R32|sqrt|Silu16)\s*\(",result):
                raise ValueError(f"Numeric closure incomplete for producer {key}; residual numeric primitive; producer not published")
            query,_=self.conversions.analyze_expression(result)
            if self.conversions.key(query) in self.conversions.converted_regions:
                # Numeric closure already reached a certified CAS fixed
                # point. Register it before selector search so its mandatory
                # stabilization can use the same-context literal envelope.
                self.compiler.register_completed_region(result,self.domains,word_closed=True)
                synchronized=self.compiler.synchronize(result,self.domains)
                if synchronized!=result:self.conversions.propagate_closed_identity(result,synchronized)
                result=synchronized
        self.compiler.register_completed_region(result,self.domains,word_closed=self.conversions is not None)
        self.memo[key]=result
        self.events.append((key,len(expression),len(result),len(self.compiler.events)-before))
        if self.on_completed is not None:self.on_completed(self)
        return result

    def reduction(self,terms):
        if self.parallel_budget is not None:
            from direct_sympy_parallel import parallel_lanes
            lanes,stats=parallel_lanes(terms,self.compiler,self.domains,self.parallel_budget)
            self.parallel_events.append(stats)
            return "R16("+self.op("+",self.op("+",lanes[0],lanes[1]),self.op("+",lanes[2],lanes[3]))+")"
        lanes=[]
        for lane in range(4):
            value="0.0"
            for term in terms[lane::4]:value=self.op("+",value,term)
            lanes.append(value)
        return "R16("+self.op("+",self.op("+",lanes[0],lanes[1]),self.op("+",lanes[2],lanes[3]))+")"

    def linear(self,name,row,input_value,*,input_bounds=None):
        shape=self.shape(name)
        if len(shape)!=2 or not 0<=row<shape[0]:raise ValueError("Invalid projection coordinate")
        if input_bounds is not None:
            intervals=input_bounds()
            if intervals is not None:
                from direct_sympy_layer_bounds import dot_interval
                enclosed=dot_interval(self,name,row,intervals)
                if enclosed is not None and struct.pack('e',enclosed[0])==struct.pack('e',enclosed[1]):
                    self.constant_projections+=1
                    return repr(enclosed[0])
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

    def norm_intervals(self,name,input_value):
        """Certify weighted RMS Half intervals without building mean/inverse."""
        if self.conversions is None:return None
        inputs=tuple(input_value(i) for i in range(self.width))
        key=(self.compiler.context(self.domains),name,inputs)
        if key in self.norm_bound_cache:return self.norm_bound_cache[key]
        sources=[]
        for value in inputs:
            node,_=self.conversions.analyze_expression(value)
            bound=self.conversions.bounds(node)
            if self.conversions.value_kind(node)!='half' or bound is None:
                self.norm_bound_cache[key]=None;return None
            sources.append(bound)
        epsilon=f32(self.config['rms_norm_eps']);result=[]
        for i in range(self.width):
            enclosure=rms_component_enclosures(sources,i,epsilon)
            if enclosure is None:self.norm_bound_cache[key]=None;return None
            weight=float(self.weight(name,i))
            if not math.isfinite(weight):self.norm_bound_cache[key]=None;return None
            values=sorted(value*weight for value in enclosure[1][0])
            try:result.append(tuple(struct.unpack('e',struct.pack('e',value))[0] for value in values))
            except OverflowError:self.norm_bound_cache[key]=None;return None
        self.norm_bound_cache[key]=result
        return result

    def norm(self,key,name,coordinate,input_value):
        certificate=[]
        def build():
            epsilon_value=f32(self.config['rms_norm_eps'])
            epsilon=repr(epsilon_value)
            bound=rms_half_bound(self.width,epsilon_value)
            source_bounds=[];input_expressions=[]
            if self.conversions is not None and bound is not None:
                for i in range(self.width):
                    operand=input_value(i);input_expressions.append(operand)
                    query,_=self.conversions.analyze_expression(operand)
                    own=self.conversions.bounds(query)
                    if self.conversions.value_kind(query)!='half' or own is None:
                        source_bounds=[];break
                    source_bounds.append(own)
            if source_bounds and self.width<=8 and all(float(self.weight(name,i)) in (-1,1) for i in range(self.width)):
                from direct_sympy_projection_constraints import norm_squared_floor,norm_error_bound
                floor=norm_squared_floor(source_bounds,epsilon_value)
                if floor:
                    minimum=lambda s:F(max(s.minimum_magnitude,s.minimum if s.minimum>0 else -s.maximum if s.maximum<0 else 0))
                    metadata={'sources':tuple(input_expressions),'epsilon':F(epsilon_value),
                        'context':self.compiler.context(self.domains),
                        'sourceNormFloor':sum((minimum(s)**2 for s in source_bounds),F(0)),
                        'inputMagnitudes':{i:F(max(abs(s.minimum),abs(s.maximum))) for i,s in enumerate(source_bounds)},
                        'gamma':tuple(F(float(self.weight(name,i))) for i in range(self.width)),
                        'roundingError':norm_error_bound(self.width,epsilon_value)}
                    certificate.append(((self.compiler.context(self.domains),name,tuple(input_expressions)),floor,metadata))
            components=rms_component_enclosures(source_bounds,coordinate,epsilon_value) if source_bounds else None
            if components is not None:
                low,high=components[1][0]
                if struct.pack('e',low)==struct.pack('e',high):
                    # Prove irrelevance before expanding mean/inverse roots.
                    # The weighted Half operation still follows its original
                    # numerical boundary, including the sign of zero.
                    self.rms_constant_components+=1
                    return 'R16('+self.op('*',repr(low),self.weight(name,coordinate))+')'
            dominant=dominant_half_rms_source(source_bounds,epsilon_value) if source_bounds else None
            if dominant is None:
                mean=self.producer(key+":mean",lambda:self.op("+",self.op("/",self.rms_sum(input_value),str(self.width)),epsilon))
                root="R32(sqrt("+mean+"))"
            else:
                # Exhaustive native proof over every finite nonzero Half
                # certifies sqrt(x*x/width) rounded to F32 for widths 1/2.
                # Keep the division by this rounded root in original order.
                self.rms_dominant_squares+=1
                operand=input_value(dominant)
                mean=self.producer(key+":mean",lambda:self.op("/",self.op("*",operand,operand),str(self.width)))
                magnitude="Float64(U64And(Bits64("+operand+"),9223372036854775807))"
                own=source_bounds[dominant]
                minimum=max(own.minimum_magnitude,own.minimum if own.minimum>0 else -own.maximum if own.maximum<0 else 0)
                magnitude_key=self.conversions.key(syntax(magnitude))
                self.conversions.completed[magnitude_key]=FiniteSource(minimum,max(abs(own.minimum),abs(own.maximum)),own.quantum,minimum)
                self.conversions.half_values.add(magnitude_key)
                self.conversions.f32_values.add(magnitude_key)
                self.conversions.no_negative_zero_values.add(magnitude_key)
                root="R32("+magnitude+" * "+("1.0" if self.width==1 else "0.7071067811865476")+")"
            inverse=self.producer(key+":inverse",lambda:self.op("/","1.0",root))
            product=self.op("*",input_value(coordinate),inverse)
            normalized="R16("+product+")"
            if self.conversions is not None:
                # The correlation proof assumes finite Half operands. A
                # later residual may overflow even when the original inputs
                # are finite; never carry the bound across that frontier.
                finite_half=len(source_bounds)==self.width
                if finite_half:
                    raw,raw_keys=self.conversions.analyze_expression(product)
                    half,half_keys=self.conversions.analyze_expression(normalized)
                    for node,keys in ((raw,raw_keys),(raw.args[0],raw_keys),(half,half_keys)):
                        existing=self.conversions.bounds(node)
                        q=existing.quantum if existing is not None else (-149 if node is raw else -1074)
                        if node is not raw and isinstance(node,ast.Call) and node.func.id=="R16":q=-24
                        low=-bound if existing is None else max(-bound,existing.minimum)
                        high=bound if existing is None else min(bound,existing.maximum)
                        magnitude=existing.minimum_magnitude if existing is not None else 0
                        if components is not None:
                            own,minimum=components[1 if node is half else 0]
                            low,high=max(low,own[0]),min(high,own[1])
                            magnitude=max(magnitude,minimum)
                        if low>high:raise ValueError('RMS correlation contradicts existing enclosure')
                        enclosure=FiniteSource(low,high,q,magnitude)
                        virtual=self.conversions.key(node)
                        self.conversions.completed[virtual]=enclosure
                        self.conversions.completed[keys.get(node,virtual)]=enclosure
                    # This is the inverse constructed immediately above,
                    # not a guessed relation between positive cached values.
                    self.conversions.remember_rms_guard(product,input_value(coordinate),mean,inverse)
            return "R16("+self.op("*",normalized,self.weight(name,coordinate))+")"
        result=self.producer(key+":"+str(coordinate),build)
        if certificate:
            vector_key,floor,metadata=certificate[0]
            vector=self.norm_vectors.setdefault(vector_key,{'width':self.width,'floor':floor,'components':{},'magnitudes':{},**metadata})
            bound=self.conversions.bounds(syntax(result))
            if bound is not None:
                vector['components'][coordinate]=result
                vector['magnitudes'][coordinate]=F(max(abs(bound.minimum),abs(bound.maximum)))
        return result

    def gated(self,prefix,neuron,input_value,*,input_bounds=None):
        # Each scalar projection owns its rounding frontier. Close it before
        # substituting it into the activation/product; composing both raw
        # reductions first postpones numerical simplification incorrectly.
        gate=self.producer(prefix+"gate:"+str(neuron),lambda:
            self.linear(prefix+"mlp.gate_proj.weight",neuron,input_value,**({"input_bounds":input_bounds} if input_bounds is not None else {})))
        activation=self.producer(prefix+"activation:"+str(neuron),lambda:"Silu16("+gate+")")
        up=self.producer(prefix+"up:"+str(neuron),lambda:
            self.linear(prefix+"mlp.up_proj.weight",neuron,input_value,**({"input_bounds":input_bounds} if input_bounds is not None else {})))
        if self.conversions is not None and all(value in self.conversions.closed_literals and value in self.conversions.closed_literal_keys for value in (activation,up)):
            return self.conversions.compose_closed(
                "R16(R32(X999999998 * X999999999))",
                {"X999999998":activation,"X999999999":up})
        return "R16("+self.op("*",activation,up)+")"

    def invisible_layer_update(self,layer,prefix,kind,coordinate,value):
        if self.conversions is None:return False
        # The RMS enclosure needs every previous scalar to be finite Half.
        previous=[syntax(self.hidden(layer-1,i)) for i in range(self.width)]
        if any(self.conversions.value_kind(x)!="half" or self.conversions.bounds(x) is None for x in previous):return False
        radius=self.conversions.half_cell_radius(syntax(value))
        if radius is None:return False
        if prefix not in self.layer_bound_cache:
            from direct_sympy_layer_bounds import layer as layer_bounds
            self.layer_bound_cache[prefix]=layer_bounds(self,prefix)
        bounds=self.layer_bound_cache[prefix]
        if bounds is None or coordinate>=len(bounds[kind]):return False
        if kind=="mlp":
            # Post-attention RMS cannot inherit a finite-Half bound when
            # any residual could overflow. Prove every residual enclosure
            # without expanding its dependencies first.
            for i,node in enumerate(previous):
                domain=self.conversions.bounds(node)
                if max(abs(domain.minimum),abs(domain.maximum))+bounds["attention"][i]>=65520:return False
        bound=bounds[kind][coordinate]
        if bound is None or not bound<radius:return False
        self.elided_updates.append((prefix,kind,coordinate,bound,radius))
        return True

    def hidden(self,layer,coordinate):
        if layer<0:return f"X{coordinate+1}"
        prefix=f"model.layers.{layer}."
        def pre(column):return self.norm(prefix+"pre",prefix+"input_layernorm.weight",column,lambda i:self.hidden(layer-1,i))
        def residual(column):
            original=self.hidden(layer-1,column)
            if self.invisible_layer_update(layer,prefix,"attention",column,original):return original
            def value(i):return self.producer(prefix+"v:"+str(i),lambda:self.linear(prefix+"self_attn.v_proj.weight",i,pre,input_bounds=lambda:self.norm_intervals(
                prefix+"input_layernorm.weight",lambda j:self.hidden(layer-1,j))))
            heads=self.config["num_attention_heads"];kv_heads=self.config["num_key_value_heads"]
            head_dim=self.config.get("head_dim",self.width//heads)
            if heads%kv_heads:raise ValueError("Invalid grouped-query geometry")
            # Position zero has one causal key: its probability is exactly 1.
            # This is a coordinate proof, never prompt/length specialization.
            context=lambda i:self.producer(prefix+"context:"+str(i),lambda:self.reduction([
                self.op("*","1.0",value((i//head_dim//(heads//kv_heads))*head_dim+i%head_dim))]))
            attention=self.linear(prefix+"self_attn.o_proj.weight",column,context)
            return "R16("+self.op("+",original,attention)+")"
        def post(column):return self.norm(prefix+"post",prefix+"post_attention_layernorm.weight",column,
            lambda i:self.producer(prefix+"residual:"+str(i),lambda:residual(i)))
        def build_hidden():
            value=self.producer(prefix+"residual:"+str(coordinate),lambda:residual(coordinate))
            if self.invisible_layer_update(layer,prefix,"mlp",coordinate,value):return value
            return "R16("+self.op("+",value,self.linear(prefix+"mlp.down_proj.weight",coordinate,
                lambda i:self.producer(prefix+"gated:"+str(i),lambda:self.gated(prefix,i,post,input_bounds=lambda:self.norm_intervals(
                    prefix+"post_attention_layernorm.weight",lambda j:self.producer(prefix+"residual:"+str(j),lambda:residual(j)))))))+")"
        return self.producer(prefix+"hidden:"+str(coordinate),build_hidden)

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
    parser.add_argument("--compressed-savepoints",action="store_true",help="Store exact producer strings with streaming lossless gzip")
    parser.add_argument("--workers",type=int,default=1)
    parser.add_argument("--memory-mib",type=int,default=2048)
    parser.add_argument("--block-size",type=int,default=64)
    args=parser.parse_args()
    if args.max_characters<1 or args.max_seconds<1:parser.error("Resource budgets must be positive")
    if args.resume and not args.savepoint_directory:parser.error("--resume requires --savepoint-directory")
    def timeout(*_):raise TimeoutError("Compilation wall-clock budget exceeded")
    signal.signal(signal.SIGALRM,timeout);signal.alarm(args.max_seconds)
    compiler=StringCompiler(max_characters=args.max_characters)
    from direct_sympy_parallel import ParallelBudget
    budget=ParallelBudget(args.workers,args.memory_mib*1024*1024,args.block_size)
    if args.workers>1:
        import torch
        torch.set_num_threads(1)
    with ExitStack() as resources:
        model=resources.enter_context(CheckpointStrings(args.checkpoint,compiler,lower_conversions=not args.reference_boundaries,parallel_budget=budget if args.workers>1 else None))
        if args.savepoint_directory:
            from direct_sympy_savepoints import ProducerSavepoints
            store=resources.enter_context(ProducerSavepoints(args.savepoint_directory,model,args.dimension,compressed=args.compressed_savepoints))
            if args.resume:print(f"Restored completed dependencies: {store.restore(model)}")
            elif (store.directory/'frontier.json').exists():raise ValueError("Existing savepoint requires --resume; refusing to overwrite")
            model.on_completed=store.save
        try:expression=model.coordinate(args.dimension)
        except (ValueError,TimeoutError) as error:
            signal.alarm(0)
            Path(str(args.output)+".parallel.json").parent.mkdir(parents=True,exist_ok=True)
            Path(str(args.output)+".parallel.json").write_text(json.dumps(model.parallel_events,indent=2)+"\n")
            path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
            Path(str(path)+".growth.tsv").write_text("dependency\tbeforeCharacters\tafterCharacters\tCASPasses\n"+
                "\n".join("\t".join(map(str,e)) for e in model.events)+"\n")
            Path(str(path)+".substitutions.tsv").write_text("variable\ttemplateCharacters\treplacementCharacters\toccurrences\testimatedCharacters\tfinalCharacters\tstatus\n"+
                "\n".join("\t".join(map(str,e)) for e in compiler.substitution_events)+"\n")
            Path(str(path)+".budget-simplifications.tsv").write_text("virtualCharacters\tstabilizedVirtualCharacters\texpandedCharacters\tcompletedRegions\tCASPasses\tadmitted\n"+
                "\n".join("\t".join(map(str,e)) for e in compiler.budget_events)+"\n")
            Path(str(path)+".conditions.tsv").write_text("path\tarm\taction\n"+
                "\n".join("\t".join(map(str,e)) for e in compiler.condition_events)+"\n")
            Path(str(path)+".synchronization.tsv").write_text("beforeCharacters\tcandidateCharacters\tafterCharacters\tlifts\tsynchronizedOperations\tstatus\n"+
                "\n".join("\t".join(map(str,e)) for e in compiler.synchronization_events)+"\n")
            if model.conversions is not None:
                Path(str(path)+".frontier-closures.tsv").write_text("producerAlias\tbeforeRestoredCharacters\tcandidateRestoredCharacters\tarms\tadmitted\n"+
                    "\n".join("\t".join(map(str,e)) for e in model.conversions.frontier_events)+"\n")
                Path(str(path)+".numeric-envelopes.tsv").write_text("inputCharacters\tvirtualCharacters\tclosedVirtualCharacters\texpandedCharacters\tcompletedRegions\n"+
                    "\n".join("\t".join(map(str,e)) for e in model.conversions.numeric_envelopes)+"\n")
            if compiler.failed_substitution is not None:
                template,name,replacement=compiler.failed_substitution
                write_expression(Path(str(path)+".failed-template.work.expr"),template)
                write_expression(Path(str(path)+".failed-replacement.work.expr"),replacement)
            last=next(reversed(model.memo),None)
            if last is not None:write_expression(Path(str(path)+".prefix.work.expr"),model.memo[last])
            print(f"Compilation stopped: {error}; completedDependencies={len(model.events)} lastDependency={last} closedConversions={model.conversions.closed if model.conversions else 0} redundantConversions={model.conversions.redundant if model.conversions else 0} reusedConvertedRegions={model.conversions.reused_regions if model.conversions else 0} visitedConversionNodes={model.conversions.visited_nodes if model.conversions else 0} earlyHalfProductsThisRun={model.early_half_products}; no coordinate artifact admitted")
            return 1
        signal.alarm(0)
        Path(str(args.output)+".parallel.json").parent.mkdir(parents=True,exist_ok=True)
        Path(str(args.output)+".parallel.json").write_text(json.dumps(model.parallel_events,indent=2)+"\n")
        path=Path(args.output)
        path.parent.mkdir(parents=True,exist_ok=True)
        # Working-expression suffix prevents mistaking residual numerical
        # primitives for a fully lowered final artifact.
        working=Path(str(path)+".work.expr")
        write_expression(working,expression)
        growth=Path(str(path)+".growth.tsv")
        growth.write_text("dependency\tbeforeCharacters\tafterCharacters\tCASPasses\n"+
            "\n".join("\t".join(map(str,e)) for e in model.events)+"\n")
        print(f"SymPy working coordinate: position=0 dimension={args.dimension} characters={len(expression)} weightReads={model.read_weights} CASPasses={len(compiler.events)}")
        print(f"Working string: {working}; no final artifact admitted: R32/R16/sqrt/Silu16 lowering and emitted-artifact parity remain pending")
        return 2


if __name__=="__main__":
    raise SystemExit(main())
