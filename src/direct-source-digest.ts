import { createHash } from 'node:crypto';

/** Bounded lexical batching for SHA256. Hashes exactly the bytes that separate
 * update calls would hash; non-ASCII chunks bypass batching so even split UTF16
 * surrogate pairs preserve the original per-chunk UTF8 encoding.
 */
export class DirectSourceDigest {
  private readonly hash=createHash('sha256');
  private chunks:string[]=[];
  private characters=0;
  update(chunk:string):this{
    if(chunk.length>=8192||/[^\x00-\x7f]/.test(chunk)){
      this.flush();this.hash.update(chunk);return this;
    }
    if(this.characters+chunk.length>8192||this.chunks.length>=1024)this.flush();
    this.chunks.push(chunk);this.characters+=chunk.length;return this;
  }
  private flush():void{
    if(!this.chunks.length)return;
    this.hash.update(this.chunks.join(''));this.chunks=[];this.characters=0;
  }
  digest(encoding:'hex'):string{this.flush();return this.hash.digest(encoding);}
}
