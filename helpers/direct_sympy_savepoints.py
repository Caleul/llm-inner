"""Atomic compiler-only frontiers; expressions remain separate mathematical strings."""
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import tempfile

import sympy
import torch
from direct_sympy_strings import syntax, StringCompiler
from direct_sympy_conversions import FiniteSource

SOURCES=('direct_sympy_layer_bounds.py','direct_sympy_savepoints.py','direct_sympy_checkpoint.py','direct_sympy_strings.py','direct_sympy_conversions.py','direct_sympy_words.py','direct_sympy_arithmetic.py','direct_sympy_tandem.py','direct_sympy_signatures.py','direct_sympy_conditions.py','direct_sympy_synchronize.py','direct_sympy_silu.py','direct_sympy_sqrt.py','direct_sympy_parallel.py')


def digest_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':')).encode()


def atomic(path,data):
    path=Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as stream:
        temporary=Path(stream.name)
        try:
            stream.write(data);stream.flush();os.fsync(stream.fileno())
            os.replace(temporary,path)
            descriptor=os.open(path.parent,os.O_RDONLY)
            try:os.fsync(descriptor)
            finally:os.close(descriptor)
        finally:temporary.unlink(missing_ok=True)


def save_expression(directory,expression):
    """Hash/write literal strings in chunks without a full encoded copy."""
    h=hashlib.sha256()
    with tempfile.NamedTemporaryFile(dir=directory,delete=False) as stream:
        temporary=Path(stream.name)
        try:
            for offset in range(0,len(expression),1024*1024):
                block=expression[offset:offset+1024*1024].encode('utf-8')
                h.update(block);stream.write(block)
            stream.flush();os.fsync(stream.fileno())
            digest=h.hexdigest();target=Path(directory)/(digest+'.expr')
            if not target.exists():os.replace(temporary,target)
            descriptor=os.open(directory,os.O_RDONLY)
            try:os.fsync(descriptor)
            finally:os.close(descriptor)
            return digest
        finally:temporary.unlink(missing_ok=True)


class ProducerSavepoints:
    def __init__(self,directory,model,dimension):
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        (self.directory/'objects').mkdir(exist_ok=True)
        self.lock=(self.directory/'writer.lock').open('a')
        try:
            fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.identity={
                'schema':3,'position':0,'dimension':dimension,'dtype':model.compiler.dtype,
                'lowerConversions':model.conversions is not None,
                'domains':{k:[str(v.minimum),str(v.maximum),v.quantum,v.excludes_negative_zero] for k,v in model.domains.items()},
                'checkpoint':{p.name:digest_file(p) for p in [model.directory/'config.json',*sorted(model.directory.glob('*.safetensors'))]},
                'sources':{name:digest_file(Path(__file__).parent/name) for name in SOURCES},
                'python':sys.version,'sympy':sympy.__version__,'platform':platform.platform(),'byteOrder':sys.byteorder,
                'referenceBackend':{'name':'torch CPU','version':str(torch.__version__),'gitVersion':torch.version.git_version,'capability':torch.backends.cpu.get_cpu_capability()},
            }
        except BaseException:self.lock.close();raise

    def __enter__(self):return self
    def __exit__(self,*_):self.lock.close()

    def save(self,model):
        if self.lock.closed:raise ValueError('Savepoint writer is closed')
        records=[]
        previous=getattr(self,'_record_cache',{}) if getattr(self,'_record_session',None) is model.conversions else {}
        for name,expression in model.memo.items():
            entry=previous.get(name)
            if entry is not None and entry[0] is expression and (self.directory/'objects'/(entry[1]['digest']+'.expr')).exists():
                # Immutable literal, same conversion session and input domain.
                # Do not parse/sign every old producer after each new one.
                record=entry[1].copy();records.append(record)
                continue
            digest=save_expression(self.directory/'objects',expression)
            session=model.conversions
            if session and expression in session.closed_literals:
                bounds,kind,positive_zero=session.closed_literals[expression]
                key=session.closed_literal_keys[expression]
                bounds=session.completed.get(key,bounds)
                if key in session.half_values:kind='half'
                elif key in session.f32_values:kind='f32'
                positive_zero=positive_zero or key in session.no_negative_zero_values
                records.append({'name':name,'digest':digest,'characters':len(expression),
                    'bounds':None if bounds is None else [bounds.minimum.hex(),bounds.maximum.hex(),bounds.quantum],
                    'kind':kind,'noNegativeZero':positive_zero,'castsClosed':True})
                continue
            node=syntax(expression)
            bounds=session.bounds(node) if session else None
            records.append({'name':name,'digest':digest,'characters':len(expression),
                'bounds':None if bounds is None else [bounds.minimum.hex(),bounds.maximum.hex(),bounds.quantum],
                'kind':session.value_kind(node) if session else None,
                'noNegativeZero':session.no_negative_zero(node) if session else False,
                'castsClosed':session is not None and session.key(node) in session.converted_regions})
        for record,expression in zip(records,model.memo.values()):
            stored=model.conversions.selector_literals.get(expression) if model.conversions else None
            record['selectorArmBounds']=[] if stored is None else [None if b is None else [b.minimum.hex(),b.maximum.hex(),b.quantum] for b in stored[3]]
        payload={'identity':self.identity,'records':records,'events':model.events,'weightReads':model.read_weights}
        envelope={'payload':payload,'integrity':hashlib.sha256(canonical(payload)).hexdigest()}
        atomic(self.directory/'frontier.json',canonical(envelope))
        # Only a successfully published frontier enters this compiler cache.
        self._record_cache={name:(expression,record.copy()) for (name,expression),record in zip(model.memo.items(),records)}
        self._record_session=model.conversions

    def restore(self,model):
        if self.lock.closed:raise ValueError('Savepoint writer is closed')
        envelope=json.loads((self.directory/'frontier.json').read_text())
        payload=envelope['payload']
        if hashlib.sha256(canonical(payload)).hexdigest()!=envelope['integrity']:raise ValueError('Savepoint manifest integrity mismatch')
        if payload['identity']!=self.identity:raise ValueError('Incompatible savepoint: checkpoint, domain, dimension or compiler semantics differ')
        loaded=[];names=set();arm_proofs={}
        validation=StringCompiler(dtype=model.compiler.dtype,max_characters=model.compiler.max_characters)
        for record in payload['records']:
            name=record['name'];digest=record['digest']
            if name in names or not re.fullmatch('[0-9a-f]{64}',digest):raise ValueError('Invalid savepoint producer identity')
            names.add(name)
            target=self.directory/'objects'/(digest+'.expr')
            if digest_file(target)!=digest:raise ValueError('Savepoint expression integrity mismatch')
            expression=target.read_text()
            if len(expression)!=record['characters'] or len(expression)>model.compiler.max_characters:raise ValueError('Savepoint expression exceeds current budget or recorded length')
            if re.search(r'\bCAS(?:Boundary|StableRegion|NumericRegion)[0-9]+\b',expression):raise ValueError('Savepoint retains a compiler placeholder')
            # Grammar validation only: do not transform or simplify an arm.
            # Global producer proofs already hold under this identical domain.
            compact,regions=validation.compact_regions(expression,validation.context(model.domains),validate_context=False)
            node=syntax(compact)
            for child in ast.walk(node):
                if isinstance(child,ast.Name) and child.id not in regions and (child.id.startswith('CASBoundary') or (re.fullmatch('X[0-9]+',child.id) and child.id not in model.domains)):raise ValueError('Savepoint retains a non-input dependency')
            if record['kind'] not in (None,'half','f32'):raise ValueError('Invalid saved dtype proof')
            if record['castsClosed'] and re.search(r'\bR(?:16|32)\s*\(',expression):raise ValueError('Savepoint conversion proof contradicts expression')
            bounds=None if record['bounds'] is None else FiniteSource(float.fromhex(record['bounds'][0]),float.fromhex(record['bounds'][1]),record['bounds'][2])
            values=record.get('selectorArmBounds')
            if not isinstance(values,list) or len(values)>16:raise ValueError('Invalid saved selector arm proofs')
            try:arm_proofs[name]=tuple(None if b is None else FiniteSource(float.fromhex(b[0]),float.fromhex(b[1]),b[2]) for b in values)
            except (ValueError,TypeError,IndexError):raise ValueError('Invalid saved selector arm proofs') from None
            loaded.append((record,expression,node,bounds,regions))
            # Prior dependencies are validated before serving as opaque literals.
            # Only closed producers are compacted; unclosed ones keep full parsing.
            if record['castsClosed']:validation.register_completed_region(expression,model.domains,node,word_closed=True)
        if len(payload['events'])!=len(loaded) or [e[0] for e in payload['events']]!=[r['name'] for r,_,_,_,_ in loaded]:raise ValueError('Savepoint frontier order mismatch')
        if model.memo:raise ValueError('Restore requires a fresh compiler frontier')
        # No model state is mutated until all files and proofs have passed.
        restored_keys={};restored_purity={}
        for record,expression,node,bounds,regions in loaded:
            model.memo[record['name']]=expression
            model.compiler.register_completed_region(expression,model.domains,node,word_closed=record['castsClosed'])
            session=model.conversions
            if session:
                arm_bounds=arm_proofs[record['name']]
                original,pure=session.restore_compact_literal(expression,node,regions,bounds,record['kind'],record['noNegativeZero'],record['castsClosed'],restored_keys,restored_purity,arm_bounds)
                restored_keys[expression]=original;restored_purity[expression]=pure
        model.events=[tuple(event) for event in payload['events']]
        model.read_weights=payload['weightReads']
        self._record_cache={record['name']:(expression,record.copy()) for record,expression,_,_,_ in loaded}
        self._record_session=model.conversions
        return len(loaded)
