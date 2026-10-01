"""memplan.py - where the bytes of the DeepSeek-V4.1-Flash GGUF go on one GPU box (docs/deepseek/PLAN.md section 2).

All sizes come from the GGUF's own tensor table (manifest.py passes the per-group byte totals); the box is described
by flags.  Placement decided in the plan:

    GPU   dense weights (attention, shared experts, router, mHC, indexer/compressor, Engram projections, output head),
          KV cache (fp16), activations / prompt buffers / CUDA context, and the hot-expert cache = whatever is left.
    RAM   ALL routed experts, resident, split in halves across the NUMA nodes (CONTRACTS.md); token_embd (rows are
          looked up per token); a few GiB of engine buffers; the OS; optionally the DSpark sidecar.
    SSD   Engram tables, mapped in place; the RAM that is left over is their page cache.

Nothing here is a measurement: the reserves are flags with the plan's defaults.
"""
from __future__ import annotations

GIB = 2 ** 30

# groups that are NOT "dense" (see ds41_spec.TSpec.group)
NON_DENSE = ("expert", "engram_embed", "embd")


def _gib(b) -> float:
    return b / GIB


def plan(groups: dict, *, expert_bytes: int, n_experts: int, engram_row_bytes: int | None, engram_rows_per_token: int,
         dspark_bytes: int = 0, vram_gib: float = 32.0, ram_gib: float = 384.0, numa_nodes: int = 2,
         ctx_tokens: int = 131072, kv_bytes_per_token: int = 3200, gpu_reserve_gib: float = 3.0,
         os_gib: float = 8.0, host_buffers_gib: float = 3.0, kernel_overhead_pct: float = 1.8,
         embd_on_gpu: bool = False, dspark_on_cpu: bool = False, cache_gib: float | None = None,
         cache_sizes_gib=(16, 18, 20, 22, 24)) -> dict:
    out: dict = {"inputs": {"vram_gib": vram_gib, "ram_gib": ram_gib, "numa_nodes": numa_nodes,
                            "ctx_tokens": ctx_tokens, "kv_bytes_per_token": kv_bytes_per_token,
                            "gpu_reserve_gib": gpu_reserve_gib, "os_gib": os_gib,
                            "host_buffers_gib": host_buffers_gib, "kernel_overhead_pct": kernel_overhead_pct,
                            "embd_on_gpu": embd_on_gpu, "dspark_on_cpu": dspark_on_cpu}}
    warn: list = []

    dense = {g: b for g, b in groups.items() if g not in NON_DENSE}
    dense_bytes = sum(dense.values())
    embd_bytes = groups.get("embd", 0)
    expert_total = groups.get("expert", 0)
    engram_bytes = groups.get("engram_embed", 0)

    # ---- GPU
    gpu_dense = dense_bytes + (embd_bytes if embd_on_gpu else 0)
    kv_bytes = ctx_tokens * kv_bytes_per_token
    reserve = int(gpu_reserve_gib * GIB)
    vram = int(vram_gib * GIB)
    cache_budget = vram - gpu_dense - kv_bytes - reserve
    gpu = {"vram_bytes": vram, "dense_bytes": gpu_dense, "dense_breakdown": dense, "kv_bytes": kv_bytes,
           "reserve_bytes": reserve, "cache_budget_bytes": cache_budget, "expert_bytes": expert_bytes}
    out["gpu"] = gpu

    def fit(budget_bytes):
        if expert_bytes <= 0 or budget_bytes <= 0:
            return 0
        return min(n_experts, int(budget_bytes // expert_bytes))

    gpu["experts_fit"] = fit(cache_budget)
    gpu["fraction_of_all"] = gpu["experts_fit"] / n_experts if n_experts else 0.0
    if cache_budget <= 0:
        warn.append(f"GPU: dense weights + KV + reserve ({_gib(gpu_dense + kv_bytes + reserve):.2f} GiB) leave nothing "
                    f"of {vram_gib:g} GiB for an expert cache")
    sizes = list(cache_sizes_gib)
    if cache_gib is not None and cache_gib not in sizes:
        sizes.append(cache_gib)
    out["cache_table"] = [{"cache_gib": s, "experts": fit(int(s * GIB)),
                           "fraction_of_all": fit(int(s * GIB)) / n_experts if n_experts else 0.0}
                          for s in sorted(sizes)]
    if cache_gib is not None:
        out["explicit_cache"] = {"cache_gib": cache_gib, "experts": fit(int(cache_gib * GIB)),
                                 "fits_in_vram": int(cache_gib * GIB) <= cache_budget}
        if int(cache_gib * GIB) > cache_budget:
            warn.append(f"--cache-gib {cache_gib:g} exceeds the {_gib(cache_budget):.2f} GiB the GPU has left")

    # ---- RAM
    ram = int(ram_gib * GIB)
    usable = int(ram * (1 - kernel_overhead_pct / 100.0))
    host_embd = 0 if embd_on_gpu else embd_bytes
    items = {
        "routed_experts": expert_total,
        "os_services": int(os_gib * GIB),
        "engine_host_buffers": int(host_buffers_gib * GIB),
        "token_embd_host": host_embd,
        "dspark_sidecar": dspark_bytes if dspark_on_cpu else 0,
    }
    page_cache = usable - sum(items.values())
    cached = max(0, min(page_cache, engram_bytes))
    n = max(1, int(numa_nodes))
    ramd = {"ram_bytes": ram, "usable_bytes": usable, "items": items, "page_cache_bytes": page_cache,
            "engram_bytes": engram_bytes, "engram_cached_bytes": cached,
            "engram_cached_fraction": (cached / engram_bytes) if engram_bytes else 1.0,
            "numa_nodes": n, "dspark_bytes": dspark_bytes}
    # experts are row-split across the nodes (CONTRACTS.md: halves); other layouts are only an even split here
    per_node_experts = expert_total // n
    ramd["per_node"] = [{"node": i, "ram_bytes": ram // n, "experts_bytes": per_node_experts,
                         "free_bytes_after_experts": ram // n - per_node_experts} for i in range(n)]
    out["ram"] = ramd
    if n not in (1, 2):
        warn.append(f"--numa-nodes {n}: the port's expert split is defined for 1 or 2 nodes (halves); "
                    "the per-node numbers below are an even split")
    if per_node_experts > ram // n:
        warn.append(f"RAM: a node's share of the experts ({_gib(per_node_experts):.1f} GiB) exceeds its RAM "
                    f"({_gib(ram // n):.1f} GiB)")
    if page_cache < 0:
        warn.append(f"RAM: experts + reserves ({_gib(sum(items.values())):.1f} GiB) exceed the usable "
                    f"{_gib(usable):.1f} GiB; the experts do not fit in RAM")
    elif engram_bytes and ramd["engram_cached_fraction"] < 0.9:
        warn.append(f"RAM: only {100 * ramd['engram_cached_fraction']:.0f} % of the Engram tables fit in the page "
                    "cache; the rest is read from the SSD on demand")
    out["engram"] = {"tables_bytes": engram_bytes, "row_bytes": engram_row_bytes,
                     "rows_per_token": engram_rows_per_token,
                     "bytes_per_token": (engram_row_bytes or 0) * engram_rows_per_token}
    out["warnings"] = warn
    return out


def format_plan(p: dict, n_layers: int = 40) -> str:
    i = p["inputs"]
    g, r = p["gpu"], p["ram"]
    L: list = []
    w = L.append
    w(f"Memory plan: GPU {i['vram_gib']:g} GiB, RAM {i['ram_gib']:g} GiB, {i['numa_nodes']} NUMA node(s)")
    w(f"  GPU ({i['vram_gib']:g} GiB)")
    br = ", ".join(f"{k} {_gib(v):.2f}" for k, v in sorted(g["dense_breakdown"].items(), key=lambda kv: -kv[1]))
    w(f"    dense weights, GPU-resident        {_gib(g['dense_bytes']):8.2f} GiB   ({br})")
    w(f"    KV cache, fp16, {i['ctx_tokens']} tokens x {i['kv_bytes_per_token']} B   {_gib(g['kv_bytes']):8.2f} GiB")
    w(f"    activations / prompt buffers / CUDA  {_gib(g['reserve_bytes']):8.2f} GiB   (--gpu-reserve-gib)")
    w(f"    hot-expert cache budget            {_gib(g['cache_budget_bytes']):8.2f} GiB   -> "
      f"{g['experts_fit']} experts of {g['expert_bytes']:,} B  "
      f"({g['experts_fit'] / n_layers:.1f} per layer, {100 * g['fraction_of_all']:.1f} % of all)")
    w("    cache size -> experts (per layer, share of all):  " + "   ".join(
        f"{t['cache_gib']:g} GiB -> {t['experts']} ({t['experts'] / n_layers:.1f}, {100 * t['fraction_of_all']:.1f} %)"
        for t in p["cache_table"]))
    if "explicit_cache" in p:
        e = p["explicit_cache"]
        w(f"    --cache-gib {e['cache_gib']:g}: {e['experts']} experts" + ("" if e["fits_in_vram"] else
                                                                           "  (does not fit the GPU budget above)"))
    w(f"  RAM ({i['ram_gib']:g} GiB, {_gib(r['usable_bytes']):.1f} GiB usable after {i['kernel_overhead_pct']:g} % "
      "kernel/struct-page overhead)")
    it = r["items"]
    n = r["numa_nodes"]
    w(f"    routed experts, all resident        {_gib(it['routed_experts']):8.2f} GiB   "
      f"({_gib(it['routed_experts'] // n):.2f} per node: row-split halves)")
    w(f"    OS, services                        {_gib(it['os_services']):8.2f} GiB   (--os-gib)")
    w(f"    engine host buffers                 {_gib(it['engine_host_buffers']):8.2f} GiB   (--host-buffers-gib)")
    w(f"    token_embd (host lookup)            {_gib(it['token_embd_host']):8.2f} GiB")
    if it["dspark_sidecar"]:
        w(f"    DSpark sidecar on the CPU           {_gib(it['dspark_sidecar']):8.2f} GiB")
    elif r.get("dspark_bytes"):
        w(f"    (DSpark sidecar {_gib(r['dspark_bytes']):.2f} GiB not counted; --dspark-on-cpu adds it)")
    w(f"    left for Engram's page cache        {_gib(r['page_cache_bytes']):8.2f} GiB   of "
      f"{_gib(r['engram_bytes']):.2f} GiB of Engram tables -> {100 * r['engram_cached_fraction']:.1f} % cached")
    for nd in r["per_node"]:
        w(f"    node {nd['node']}: {_gib(nd['ram_bytes']):.0f} GiB RAM, experts {_gib(nd['experts_bytes']):.2f} GiB, "
          f"{_gib(nd['free_bytes_after_experts']):.2f} GiB free")
    e = p["engram"]
    if e["row_bytes"]:
        w(f"  Engram: {e['rows_per_token']} rows/token x {e['row_bytes']} B = {e['bytes_per_token']:,} B/token "
          f"(+ 4 KiB page granularity on a cache miss)")
    for t in p["warnings"]:
        w(f"  WARNING: {t}")
    return "\n".join(L)
