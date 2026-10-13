"""List missing inputs for test diagnostics; never connects to Sherlock.

Planar tests: 128² histories and endpoint states at all grids.
3D: full-volume 64³ histories only, keeping the requested transfer manageable.
Other high-resolution 3D error panels remain unavailable until fetched separately.
"""
import gzip
import re
from pathlib import Path
import test_diagnostics as td


def main():
    data=td.Data();wanted=set();results=data.sw.results
    def add(path):
        if not (results/path).is_file():wanted.add(path)
    for (test,method,n,seed),planned in data.sw.by.items():
        if method not in td.METHODS+td.sd.REFERENCES or test.startswith("gate_") or test=="throughput_ot3d":continue
        part=data.sw.part(test,method,n,seed)
        base=str(Path(part).parent)
        add(base+"/environment.toml")
        meta=data.sw.meta(test,method,n,seed)
        if not meta:continue
        planar=bool(meta.get("planar"))
        target_n=128 if planar else 64
        for f in meta.get("frames",[]):
            endpoint=abs(f["time"]-data.sw.configs[test]["t_end"])<=1e-6*max(1,data.sw.configs[test]["t_end"])
            if method in td.sd.REFERENCES:
                if method!="athenak_reference":continue
                for grid,name in f.get("restricted",{}).items():
                    if int(grid)==target_n or (planar and endpoint):add(part+"/"+name)
            elif f.get("file") and (n==target_n or (planar and endpoint)):
                add(part+"/"+f["file"])
    for test in td.ALE_TESTS:
        c=data.sw.configs[test]
        for n in c["grids"]:
            base=f"{test}-mhdale-ale-run-N{n}-s42/data"
            for name in ("run.toml","diagnostics.csv","mhdale.log"):
                add(base+"/mhdale_ale_r1/"+name)
            add(base+"/environment.toml")
            for frame in (range(c["frames"]) if n==128 else [c["frames"]-1]):
                add(base+f"/mhdale_ale_r1/state_{frame:03d}.bin")
    out=td.ROOT/"scripts/test_analysis_fetch_needed.txt"
    out.write_text("\n".join(sorted(wanted))+"\n")
    sizes={}
    inventory=td.SITE/"sweep_status/inventory_20261007T181743Z.tsv.gz"
    if inventory.is_file():
        with gzip.open(inventory,"rt") as f:
            for line in f:
                cols=line.rstrip("\n").split("\t",3)
                if len(cols)!=4:continue
                root,size,_,path=cols
                if root=="mhd-results-sweep":sizes[path]=int(size)
    known=sum(sizes[p] for p in wanted if p in sizes)
    print(f"{len(wanted)} missing paths; {sum(p in sizes for p in wanted)} in the saved inventory ({known/1e9:.2f} GB uncompressed). New MHD-ALE records may add to this.")
    print(out)


if __name__=="__main__":main()
