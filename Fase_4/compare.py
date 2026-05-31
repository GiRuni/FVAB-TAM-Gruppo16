import sys
import pandas as pd

if len(sys.argv) != 4:
    print("Usage: python compare.py <mode_a.csv> <mode_b.csv> <output.csv>")
    sys.exit(1)

mode_a = pd.read_csv(sys.argv[1])
mode_b = pd.read_csv(sys.argv[2])
output_path = sys.argv[3]

metric_cols = ["obj_iou", "iou_hard", "io_ratio", "wdp", "func_iou", "f1_iou"]
header = ["image_id", "mask_id", "step", "token_group"] + metric_cols

out_rows = []

def _split_query_components(row) -> list[str]:
    if "query_object" in row and "query_word" in row:
        obj = str(row["query_object"]).strip() if pd.notna(row["query_object"]) else ""
        word = str(row["query_word"]).strip() if pd.notna(row["query_word"]) else ""
        if obj and word:
            return [obj, word]

    pair = str(row.get("query_pair", "")).strip()
    return [part.strip() for part in pair.split("+") if part.strip()]

for i, r in mode_b.iterrows():
    if i > 0:
        out_rows.append({c: "-"*60 for c in header})

    image_id = r["image"]

    components = _split_query_components(r)
    pair_label = " + ".join(components) if components else str(r.get("query_pair", ""))

    out_rows.append({
        "image_id": image_id,
        "mask_id": r["query_mask"],
        "step": r["target_step_end"],
        "token_group": pair_label,
        "obj_iou": r["obj_iou"],
        "iou_hard": r["iou_hard"],
        "io_ratio": r["io_ratio"],
        "wdp": r["wdp"],
        "func_iou": r["func_iou"],
        "f1_iou": r["f1_iou"],
    })

    matching = mode_a[(mode_a["image"] == image_id) & (mode_a["step"] == r["target_step_end"]) & (mode_a["target"] == str(r["query_mask"]))]
    matching_aux_fw = mode_a[(mode_a["image"] == image_id) & (mode_a["step"] >= int(r["firstword_step_start"])) & (mode_a["step"] <= int(r["firstword_step_end"])) & (mode_a["target"] == str(r["query_mask"]))]
    matching_aux_target = mode_a[(mode_a["image"] == image_id) & (mode_a["step"] >= int(r["target_step_start"])) & (mode_a["step"] <= int(r["target_step_end"])) & (mode_a["target"] == str(r["query_mask"]))]

    for _, r2 in matching.iterrows():
        out_rows.append({
            "image_id": image_id,
            "mask_id": r2["target"],
            "step": r2["step"],
            "token_group": r2["token"],
            "obj_iou": r2["obj_iou"],
            "iou_hard": r2["iou_hard"],
            "io_ratio": r2["io_ratio"],
            "wdp": r2["wdp"],
            "func_iou": r2["func_iou"],
            "f1_iou": r2["f1_iou"],
        })

    out_rows.append({c: "" for c in header})

    for _, r_fw in matching_aux_fw.iterrows():
        out_rows.append({
            "image_id": image_id,
            "mask_id": r_fw["target"],
            "step": r_fw["step"],
            "token_group": r_fw["token"],
            "obj_iou": r_fw["obj_iou"],
            "iou_hard": r_fw["iou_hard"],
            "io_ratio": r_fw["io_ratio"],
            "wdp": r_fw["wdp"],
            "func_iou": r_fw["func_iou"],
            "f1_iou": r_fw["f1_iou"],
        })
    
    out_rows.append({c: "" for c in header})

    for _, r_target in matching_aux_target.iterrows():
        out_rows.append({
            "image_id": image_id,
            "mask_id": r_target["target"],
            "step": r_target["step"],
            "token_group": r_target["token"],
            "obj_iou": r_target["obj_iou"],
            "iou_hard": r_target["iou_hard"],
            "io_ratio": r_target["io_ratio"],
            "wdp": r_target["wdp"],
            "func_iou": r_target["func_iou"],
            "f1_iou": r_target["f1_iou"],
        })

df_out = pd.DataFrame(out_rows, columns=header)
df_out.to_csv(output_path, index=False)
print("Done. Wrote", len(df_out), "rows.")
