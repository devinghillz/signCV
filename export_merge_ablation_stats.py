import csv
import re
import torch
from pathlib import Path

def analyze_state_dict(d: dict) -> dict:
    tot_elems = 0
    nnz = 0
    abs_sum = 0.0
    sq_sum = 0.0
    mx = 0.0
    for v in d.values():
        x = v.float().flatten()
        n = x.numel()
        tot_elems += n
        abs_sum += float(x.abs().sum())
        sq_sum += float((x * x).sum())
        mx = max(mx, float(x.abs().max()))
        nnz += int((x.abs() > 1e-12).sum())
    l2 = sq_sum ** 0.5
    mean_abs = abs_sum / max(tot_elems, 1)
    frac_nnz = nnz / max(tot_elems, 1)
    return dict(
        n_keys=len(d),
        tot_elems=tot_elems,
        l2=l2,
        mean_abs=mean_abs,
        max_abs=mx,
        nnz=nnz,
        frac_nnz=frac_nnz,
    )

def parse_ur_xs(name: str):
    if "_xs" not in name or not name.startswith("ur"):
        return "", ""
    left, xs = name.split("_xs", 1)
    ur = left[2:] if left.startswith("ur") else ""
    return ur, xs

def main() -> None:
    out_path = Path("checkpoints/merge_ablation_lowmem_strict_intersection_stats.csv")
    root = Path("checkpoints/merge_ablation_lowmem")
    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        p = d / "delta_star.pt"
        ur, xs = parse_ur_xs(d.name)
        row = {
            "folder": d.name,
            "unanimity_ratio": ur,
            "delta_extra_scale": xs,
            "cross_axis_mode": "intersection",
            "delta_star_path": str(p.resolve()),
            "file_exists": p.exists(),
            "bytes": p.stat().st_size if p.exists() else 0,
        }
        if p.exists():
            t = torch.load(p, map_location="cpu")
            if isinstance(t, dict):
                row.update(analyze_state_dict(t))
            else:
                row["error"] = repr(type(t))
        rows.append(row)

    fieldnames = [
        "folder", "unanimity_ratio", "delta_extra_scale", "cross_axis_mode",
        "l2", "mean_abs", "max_abs", "nnz", "frac_nnz",
        "n_keys", "tot_elems", "bytes", "delta_star_path", "file_exists",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("Wrote", out_path.resolve())
    for r in rows:
        print(r["folder"], "UR=", r["unanimity_ratio"], "XS=", r["delta_extra_scale"], "L2=", r.get("l2", "n/a"))

if __name__ == "__main__":
    main()
