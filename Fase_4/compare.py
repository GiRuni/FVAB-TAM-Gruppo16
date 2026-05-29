import sys
import pandas as pd

if len(sys.argv) != 4:
    print("Usage: python compare.py <mode_a.csv> <mode_b.csv> <output.csv>")
    sys.exit(1)

mode_a = pd.read_csv(sys.argv[1])
mode_b = pd.read_csv(sys.argv[2])
output_path = sys.argv[3]

metric_cols = ["obj_iou", "iou_hard", "io_ratio", "wdp", "func_iou", "f1_iou"]
header = ["image_id", "mask_id", "token_group"] + metric_cols

out_rows = []

for i, r in mode_b.iterrows():
    if i > 0:
        out_rows.append({c: "" for c in header})

    image_id = r["image"]

    out_rows.append({
        "image_id": image_id,
        "mask_id": r["query_mask"],
        "token_group": r["query_pair"],
        "obj_iou": r["obj_iou"],
        "iou_hard": r["iou_hard"],
        "io_ratio": r["io_ratio"],
        "wdp": r["wdp"],
        "func_iou": r["func_iou"],
        "f1_iou": r["f1_iou"],
    })

    matching = mode_a[(mode_a["image"] == image_id) & (mode_a["step"] == r["step"]) & (mode_a["target"] == str(r["query_mask"]))]

    for _, r2 in matching.iterrows():
        out_rows.append({
            "image_id": image_id,
            "mask_id": r2["target"],
            "token_group": r2["token"],
            "obj_iou": r2["obj_iou"],
            "iou_hard": r2["iou_hard"],
            "io_ratio": r2["io_ratio"],
            "wdp": r2["wdp"],
            "func_iou": r2["func_iou"],
            "f1_iou": r2["f1_iou"],
        })

df_out = pd.DataFrame(out_rows, columns=header)
df_out.to_csv(output_path, index=False)
print("Done. Wrote", len(df_out), "rows.")
