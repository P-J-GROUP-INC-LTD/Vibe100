#!/usr/bin/env python3
"""tools/ds41_gguf_remote_headers.py - read GGUF headers (metadata + tensor table) over HTTP range requests.

The DeepSeek-V4.1 GGUFs are 411 GB (MXFP4) to 1.5 TB; their headers are a few MB.  This fetches only the header
of each URL (growing the range until it parses), and prints {file: {"kv": ..., "tensors": [[name, type, dims,
offset], ...]}} as JSON.  Arrays longer than 64 entries (the tokenizer, Engram's token map) are summarised.
The outputs for the two published GGUF families are kept in third_party/deepseek-v41-flash-reference/.

    python3 tools/ds41_gguf_remote_headers.py https://huggingface.co/<repo>/resolve/main/<shard>.gguf ... > h.json
"""
import json, struct, subprocess, sys
TYPES = {0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",6:"Q5_0",7:"Q5_1",8:"Q8_0",9:"Q8_1",10:"Q2_K",11:"Q3_K",12:"Q4_K",13:"Q5_K",14:"Q6_K",15:"Q8_K",16:"IQ2_XXS",17:"IQ2_XS",18:"IQ3_XXS",19:"IQ1_S",20:"IQ4_NL",21:"IQ3_S",22:"IQ2_S",23:"IQ4_XS",24:"I8",25:"I16",26:"I32",27:"I64",28:"F64",29:"IQ1_M",30:"BF16",34:"TQ1_0",35:"TQ2_0",39:"MXFP4"}
def fetch(url, n):
    return subprocess.run(["curl","-sSL","-r",f"0-{n-1}",url],capture_output=True,check=True).stdout
class R:
    def __init__(s,b): s.b=b; s.o=0
    def u(s,f): v=struct.unpack_from(f,s.b,s.o); s.o+=struct.calcsize(f); return v[0]
    def str(s): n=s.u("<Q"); v=s.b[s.o:s.o+n].decode("utf-8","replace"); s.o+=n; return v
    def val(s,t):
        if t in (0,1): return s.u("<B" if t==0 else "<b")
        if t in (2,3): return s.u("<H" if t==2 else "<h")
        if t in (4,5): return s.u("<I" if t==4 else "<i")
        if t==6: return s.u("<f")
        if t==7: return bool(s.u("<B"))
        if t==8: return s.str()
        if t==9:
            et=s.u("<I"); n=s.u("<Q"); return [s.val(et) for _ in range(n)]
        if t in (10,11): return s.u("<Q" if t==10 else "<q")
        if t==12: return s.u("<d")
        raise ValueError(t)
def parse(url):
    n=1<<20
    while True:
        b=fetch(url,n)
        try:
            r=R(b); assert r.b[:4]==b"GGUF"; r.o=4; ver=r.u("<I"); nt=r.u("<Q"); nk=r.u("<Q")
            kv={}
            for _ in range(nk):
                k=r.str(); t=r.u("<I"); v=r.val(t)
                kv[k]=v if not (isinstance(v,list) and len(v)>64) else f"<array len {len(v)}>"
            ts=[]
            for _ in range(nt):
                name=r.str(); nd=r.u("<I"); dims=[r.u("<Q") for _ in range(nd)]; ty=r.u("<I"); off=r.u("<Q")
                ts.append((name,TYPES.get(ty,str(ty)),dims,off))
            return kv,ts
        except (struct.error, IndexError, UnicodeDecodeError, AssertionError) as e:
            if n>=1<<27: raise
            n*=4
if __name__=="__main__":
    out={}
    for url in sys.argv[1:]:
        kv,ts=parse(url); out[url.rsplit("/",1)[1]]={"kv":kv,"tensors":ts}
        print(url.rsplit("/",1)[1], len(ts), "tensors", file=sys.stderr)
    json.dump(out,sys.stdout)
