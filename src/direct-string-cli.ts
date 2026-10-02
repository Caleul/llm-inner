import {spawn} from 'node:child_process';

const [python,checkpoint,output,...flags]=process.argv.slice(2);
if(!python||!checkpoint||!output)throw new Error('Usage: direct-string-cli PYTHON CHECKPOINT OUTPUT [--dimension N] [--max-characters N]');
const helper=new URL('../../helpers/direct_sympy_checkpoint.py',import.meta.url).pathname;
const child=spawn(python,[helper,checkpoint,output,...flags],{stdio:'inherit'});
child.on('error',error=>{console.error(error.message);process.exitCode=1;});
child.on('exit',(code,signal)=>{
  if(signal)console.error(`String compiler terminated: ${signal}`);
  process.exitCode=code??1;
});
