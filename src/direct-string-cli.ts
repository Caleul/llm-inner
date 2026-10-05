import {spawn} from 'node:child_process';

const [python,checkpoint,output,...flags]=process.argv.slice(2);
if(!python||!checkpoint||!output)throw new Error('Usage: direct-string-cli PYTHON CHECKPOINT OUTPUT_DIRECTORY [--workers N] [--memory-mib N] [--max-characters N] [--reference PATH] [--seconds-per-length N] [--lower-seconds N]; --dimension N selects a coordinate diagnostic');
const diagnosticCoordinate=flags.includes('--dimension');
const helper=new URL(diagnosticCoordinate?'../../helpers/direct_sympy_checkpoint.py':'../../helpers/direct_sympy_architecture_run.py',import.meta.url).pathname;
const forwarded=diagnosticCoordinate?flags:flags.filter(flag=>flag!=='--reference-boundaries').map(flag=>flag==='--max-seconds'?'--seconds-per-length':flag);
if(!diagnosticCoordinate&&!flags.includes('--reference-boundaries')&&!forwarded.includes('--lower'))forwarded.push('--lower');
const child=spawn(python,[helper,checkpoint,output,...forwarded],{stdio:'inherit'});
child.on('error',error=>{console.error(error.message);process.exitCode=1;});
child.on('exit',(code,signal)=>{
  if(signal)console.error(`String compiler terminated: ${signal}`);
  process.exitCode=code??1;
});
