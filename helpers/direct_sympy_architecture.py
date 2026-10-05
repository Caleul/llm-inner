"""Checkpoint-weighted vector templates for the complete causal Llama forward.

Operators are compiler interfaces, not a runtime architecture. Pairwise
composition eliminates every internal port before an expression is emitted.
Numerical primitives remain explicit until the separate closure gate succeeds.
"""
from dataclasses import dataclass
import math
import struct

from direct_sympy_checkpoint import CheckpointStrings,f32,rms_half_bound
from direct_sympy_operators import OperatorBlock,compose_operators
from direct_sympy_strings import Domain,StringCompiler
from fractions import Fraction as F


@dataclass(frozen=True)
class ArchitecturePlan:
    blocks:tuple[OperatorBlock,...]
    names:tuple[str,...]
    length:int
    width:int
    vocab:int
    weight_reads:int
    finite_certificate:dict


def finite_architecture(model,length):
    """Prove all stage outputs finite on the entire finite-Half input domain.

    Softmax positivity and its unit maximum term bound each stored probability
    by one. A residual update below 15 cannot cross the Half overflow midpoint
    65520, even after the preceding F32 addition. No sampled data enters this
    certificate; projection enclosures use exact rational gamma bounds.
    """
    from direct_sympy_layer_bounds import dot,half
    from fractions import Fraction
    config=model.config;d=model.width;heads=config['num_attention_heads'];kv=config['num_key_value_heads']
    hd=config.get('head_dim',d//heads);records=[]
    def stored(value):
        result=half(math.nextafter(value,math.inf))
        if result is None:raise ValueError('Finite architecture certificate failed at Half storage')
        return result
    def normalized(name):
        bound=rms_half_bound(d,f32(config['rms_norm_eps']))
        if bound is None:raise ValueError('RMS finite certificate unavailable')
        inner=stored(bound)
        return [stored(inner*abs(float(model.weight(name,i)))) for i in range(d)]
    def project(name,bounds):
        result=dot(model,name,bounds)
        if result is None:raise ValueError('Finite architecture projection certificate unavailable: '+name)
        return result
    for layer in range(model.layers):
        prefix=f'model.layers.{layer}.';pre=normalized(prefix+'input_layernorm.weight')
        v=project(prefix+'self_attn.v_proj.weight',pre)
        # Rotary products each store Half before their original sum. Trig
        # coefficients have magnitude <=1; retain both storage boundaries.
        score=0.0
        if length>1:
            q=project(prefix+'self_attn.q_proj.weight',pre);k=project(prefix+'self_attn.k_proj.weight',pre)
            qr=[stored(2*x) for x in q];kr=[stored(2*x) for x in k]
            score_bound=Fraction(max(qr,default=0))*Fraction(max(kr,default=0))*hd*Fraction(2**24,2**24-(hd+3))
            score=stored(float(score_bound))
            stored(score*f32(hd**-0.5))
        gamma=Fraction(2**24,2**24-(length+3))
        context=[stored(float(Fraction(length)*Fraction(v[(head//(heads//kv))*hd+i])*gamma)) for head in range(heads) for i in range(hd)]
        attention=project(prefix+'self_attn.o_proj.weight',context)
        if max(attention,default=0)>=15:raise ValueError('Finite attention residual certificate unavailable')
        post=normalized(prefix+'post_attention_layernorm.weight')
        gate=project(prefix+'mlp.gate_proj.weight',post);up=project(prefix+'mlp.up_proj.weight',post)
        gated=[stored(g*u) for g,u in zip(gate,up)]
        down=project(prefix+'mlp.down_proj.weight',gated)
        if max(down,default=0)>=15:raise ValueError('Finite MLP residual certificate unavailable')
        records.append({'layer':layer,'scoreAbsMaximum':score,'attentionUpdateAbsMaximum':max(attention,default=0),
            'mlpUpdateAbsMaximum':max(down,default=0)})
    final=normalized('model.norm.weight')
    output='model.embed_tokens.weight' if config.get('tie_word_embeddings') else 'lm_head.weight'
    logits=project(output,final)
    return {'inputDomain':'All finite Half values, including both signed zeros','layers':records,'logitAbsMaximum':logits}


def architecture_plan(checkpoint,length,*,max_characters=512*1024**2):
    compiler=StringCompiler(max_characters=max_characters)
    with CheckpointStrings(checkpoint,compiler,lower_conversions=False) as model:
        config=model.config;d=model.width;heads=config['num_attention_heads'];kv=config['num_key_value_heads']
        hd=config.get('head_dim',d//heads);middle=config['intermediate_size'];vocab=config['vocab_size']
        if type(length) is not int or not 1<=length<=config['max_position_embeddings']:raise ValueError('Invalid token length')
        if heads%kv or hd%2:raise ValueError('Unsupported grouped-query/RoPE geometry')
        if config.get('attention_bias') or config.get('mlp_bias') or config.get('hidden_act')!='silu':
            raise ValueError('Architecture template requires bias-free SiLU Llama')
        rope=config.get('rope_parameters',{})
        if rope.get('rope_type','default')!='default' or config.get('rope_scaling'):raise ValueError('Unsupported RoPE scaling')
        theta=rope.get('rope_theta',config.get('rope_theta',10000.0))
        certificate=finite_architecture(model,length)
        # Match the checkpoint's CPU F32 trigonometric constants. They are
        # coefficients discovered from geometry, not prompt specialization.
        import torch
        frequency=1.0/(float(theta)**(torch.arange(0,hd,2,dtype=torch.float32)/hd))
        angles=torch.arange(length,dtype=torch.float32)[:,None]*frequency[None,:]
        cos=torch.cat((angles,angles),dim=1).cos().half().tolist()
        sin=torch.cat((angles,angles),dim=1).sin().half().tolist()
        blocks=[];names=[];next_port=1;size=length*d;finite_half_ports=set()
        model.domains={}
        def stage(name,build):
            nonlocal next_port,size
            ports=tuple(f'X{next_port+i}' for i in range(size));next_port+=size
            # finite_architecture has proved all stored interfaces finite;
            # every stage below stores Half outputs. This is a provenance
            # proof for these ports, never a domain inherited from another
            # arbitrary block or a mere assumption from an identifier.
            finite_half_ports.clear();finite_half_ports.update(ports)
            model.memo.clear();model.events.clear()
            outputs=tuple(build(ports))
            blocks.append(OperatorBlock(ports,outputs));names.append(name);size=len(outputs)
        def product(a,b):
            def half_operand(text):
                if text in finite_half_ports:return True
                try:
                    value=float(text)
                    return math.isfinite(value) and struct.unpack('e',struct.pack('e',value))[0]==value
                except (ValueError,OverflowError):return False
            if half_operand(a) and half_operand(b):
                # Two finite Half operands have <=22 product bits and
                # nonzero exponents [-48,32]: F32 multiplication is exact,
                # including either sign of zero. Keep every subsequent
                # addition/reduction/storage boundary in original order.
                left=model.compiler.substitute('(X999999998 * X999999999)','X999999998',a,model.domains)
                return model.compiler.substitute(left,'X999999999',b,model.domains)
            return model.op('*',a,b)
        def dot(weight,row,inputs):
            rows,columns=model.shape(weight)
            if not 0<=row<rows or len(inputs)!=columns:raise ValueError('Projection geometry mismatch')
            terms=[]
            for column in range(columns):
                coefficient=model.weight(weight,row,column)
                if not math.isfinite(float(coefficient)):raise ValueError('Finite checkpoint weights required')
                terms.append(product(inputs[column],coefficient))
            return model.reduction(terms)
        def norm(name,ports,key):
            return [model.norm(key+':'+str(row),name,i,lambda j:ports[row*d+j]) for row in range(length) for i in range(d)]
        def add(a,b):return 'R16('+model.op('+',a,b)+')'
        qsize=length*heads*hd;ksize=length*kv*hd;state=length*d
        for layer in range(model.layers):
            prefix=f'model.layers.{layer}.'
            stage(prefix+'pre-normalization',lambda p: list(p)+norm(prefix+'input_layernorm.weight',p,'pre'))
            roles=('v',) if length==1 else ('q','k','v')
            def project(p):
                result=list(p[:state]);normalized=p[state:]
                for role in roles:
                    count=model.shape(prefix+'self_attn.'+role+'_proj.weight')[0]
                    result.extend(dot(prefix+'self_attn.'+role+'_proj.weight',i,normalized[row*d:(row+1)*d]) for row in range(length) for i in range(count))
                return result
            stage(prefix+'qkv-projections',project)
            if length>1:
                def rotate(p):
                    result=list(p[:state]);offset=state
                    for role,count in [('q',heads),('k',kv)]:
                        for row in range(length):
                            for head in range(count):
                                base=offset+(row*count+head)*hd
                                for i in range(hd):
                                    other=p[base+(i+hd//2)%hd]
                                    a='R16('+product(p[base+i],repr(cos[row][i]))+')'
                                    b='R16('+product(other,repr((-1 if i<hd//2 else 1)*sin[row][i]))+')'
                                    result.append(add(a,b))
                        offset+=length*count*hd
                    return result+list(p[offset:])
                stage(prefix+'rotary',rotate)
                triples=[(row,head,key) for row in range(length) for head in range(heads) for key in range(row+1)]
                def scores(p):
                    result=list(p[:state])+list(p[state+qsize+ksize:])
                    for row,head,key in triples:
                        qbase=state+(row*heads+head)*hd
                        kbase=state+qsize+(key*kv+head//(heads//kv))*hd
                        value=model.reduction([product(p[qbase+i],p[kbase+i]) for i in range(hd)])
                        result.append('R16('+model.op('*',value,repr(f32(hd**-0.5)))+')')
                    return result
                stage(prefix+'attention-scores',scores)
                def probabilities(p):
                    result=list(p[:state+ksize]);offset=state+ksize
                    for row in range(length):
                        for head in range(heads):
                            scores=p[offset:offset+row+1];offset+=row+1
                            if row==0:result.append('1.0');continue
                            # Under the finite certificate, the final maximum
                            # has only row+1 possible producers. Direct mutually
                            # exclusive winner guards preserve the first tie,
                            # avoiding recursive copies of every prior maximum.
                            arms=[]
                            for i,score in enumerate(scores[:-1]):
                                conditions=[f'{score} '+('>' if j<i else '>=')+f' {other}' for j,other in enumerate(scores) if j!=i]
                                guard=conditions[0] if len(conditions)==1 else 'And('+', '.join(conditions)+')'
                                arms.append(f'({score}, {guard})')
                            maximum='Piecewise('+', '.join(arms+[f'({scores[-1]}, True)'])+')'
                            terms=['Exp32('+model.op('-',score,maximum)+')' for score in scores]
                            if length<4:
                                denominator=terms[0]
                                for term in terms[1:]:denominator=model.op('+',denominator,term)
                            else:
                                lanes=[]
                                for lane in range(4):
                                    terms_lane=terms[lane::4];value=terms_lane[0] if terms_lane else '0.0'
                                    for term in terms_lane[1:]:value=model.op('+',value,term)
                                    lanes.append(value)
                                denominator=model.op('+',model.op('+',lanes[0],lanes[2]),model.op('+',lanes[1],lanes[3]))
                            reciprocal=model.op('/','1.0',denominator)
                            result.extend('R16('+model.op('*',term,reciprocal)+')' for term in terms)
                    return result
                stage(prefix+'softmax',probabilities)
                def context(p):
                    result=list(p[:state]);prob_offset=state+ksize
                    for row in range(length):
                        for head in range(heads):
                            probabilities=p[prob_offset:prob_offset+row+1];prob_offset+=row+1
                            for i in range(hd):
                                result.append(model.reduction([product(probabilities[key],p[state+(key*kv+head//(heads//kv))*hd+i]) for key in range(row+1)]))
                    return result
                stage(prefix+'attention-context',context)
            else:
                # The sole causal key has exactly unit probability. Remove Q/K
                # and their rotary/softmax dependencies before reading weights.
                stage(prefix+'attention-context',lambda p:list(p[:state])+[model.reduction([product('1.0',p[state+(head//(heads//kv))*hd+i])]) for head in range(heads) for i in range(hd)])
            stage(prefix+'attention-output',lambda p:list(p[:state])+[dot(prefix+'self_attn.o_proj.weight',i,p[state+row*heads*hd:state+(row+1)*heads*hd]) for row in range(length) for i in range(d)])
            stage(prefix+'attention-residual',lambda p:[add(p[i],p[state+i]) for i in range(state)])
            stage(prefix+'post-normalization',lambda p:list(p)+norm(prefix+'post_attention_layernorm.weight',p,'post'))
            stage(prefix+'gate-up',lambda p:list(p[:state])+[dot(prefix+'mlp.'+role+'_proj.weight',i,p[state+row*d:state+(row+1)*d]) for role in ['gate','up'] for row in range(length) for i in range(middle)])
            stage(prefix+'activation-product',lambda p:list(p[:state])+['R16('+model.op('*','Silu16('+p[state+i]+')',p[state+length*middle+i])+')' for i in range(length*middle)])
            stage(prefix+'down-projection',lambda p:list(p[:state])+[dot(prefix+'mlp.down_proj.weight',i,p[state+row*middle:state+(row+1)*middle]) for row in range(length) for i in range(d)])
            stage(prefix+'mlp-residual',lambda p:[add(p[i],p[state+i]) for i in range(state)])
        stage('final-normalization',lambda p:norm('model.norm.weight',p,'final'))
        output='model.embed_tokens.weight' if config.get('tie_word_embeddings') else 'lm_head.weight'
        stage('output-logits',lambda p:[dot(output,i,p[row*d:(row+1)*d]) for row in range(length) for i in range(vocab)])
        return ArchitecturePlan(tuple(blocks),tuple(names),length,d,vocab,model.read_weights,certificate)


def last_position_indices(plan):
    """Only these logits predict the next token after the whole input."""
    return tuple(range((plan.length-1)*plan.vocab,plan.length*plan.vocab))


def prune_plan(plan,output_indices=None):
    """Remove unused scalar producers before their interfaces are substituted."""
    import re
    selected=tuple(range(len(plan.blocks[-1].outputs))) if output_indices is None else tuple(output_indices)
    needed=set(selected)
    if any(type(i) is not int or not 0<=i<len(plan.blocks[-1].outputs) for i in needed):raise ValueError('Invalid output coordinate')
    if not selected or selected!=tuple(sorted(needed)):raise ValueError('Select distinct output coordinates in original order')
    result=[]
    for block in reversed(plan.blocks):
        outputs=tuple(value for i,value in enumerate(block.outputs) if i in needed)
        used=set(name for value in outputs for name in re.findall(r'\bX[1-9][0-9]*\b',value))
        needed={i for i,name in enumerate(block.inputs) if name in used}
        inputs=tuple(name for i,name in enumerate(block.inputs) if i in needed)
        if outputs:result.append(OperatorBlock(inputs,outputs))
    return tuple(reversed(result))


def compose_architecture(plan,compiler,budget,*,output_indices=None,on_wave=None):
    compiler.reuse_branch_free_regions=True
    domains={f'X{i+1}':Domain(F(-65504),F(65504),-24,False) for i in range(plan.length*plan.width)}
    blocks=prune_plan(plan,output_indices)
    outputs,stats=compose_operators(blocks,compiler,domains,budget,max_accumulated_characters=compiler.max_characters,on_wave=on_wave)
    stats.update({'stageNames':plan.names,'length':plan.length,'width':plan.width,'vocab':plan.vocab,'weightReads':plan.weight_reads,
        'originalInterfaceOutputSlots':sum(len(b.outputs) for b in plan.blocks),
        'reachableInterfaceOutputSlots':sum(len(b.outputs) for b in blocks),
        'originalComputedOutputs':sum(value not in b.inputs for b in plan.blocks for value in b.outputs),
        'reachableComputedOutputs':sum(value not in b.inputs for b in blocks for value in b.outputs)})
    return outputs,stats
