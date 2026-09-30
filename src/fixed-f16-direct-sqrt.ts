/** Direct F32 square-root iteration, rounded before a consumer takes its reciprocal. */
export function directSqrtPrefix(): string {
  return `((value)=>{` +
    `if(value!==value||value<0)return NaN;` +
    `if(value===0)return value;` +
    `if(value>1.7976931348623157e308)return Infinity;` +
    `let scaled=value,scale=1;` +
    `while(scaled>=4){scaled/=4;scale+=scale;}` +
    `while(scaled<1){scaled*=4;scale/=2;}` +
    `let root=scaled>=2?2:1;` +
    `for(let i=0;i<9;i++)root=(root+scaled/root)/2;` +
    `return Math.fround(root*scale);` +
    `})(`;
}
