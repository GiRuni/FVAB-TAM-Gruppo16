import re
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


def _norm_text(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _clean_token_piece(token: str) -> str:
    text = str(token).replace("▁", "").replace("Ġ", "")
    if text.startswith("##"):
        text = text[2:]
    return text.strip()


def _split_query_components(row) -> list[str]:
    if "query_object" in row and "query_word" in row:
        obj = str(row["query_object"]).strip() if pd.notna(row["query_object"]) else ""
        word = str(row["query_word"]).strip() if pd.notna(row["query_word"]) else ""
        if obj and word:
            return [obj, word]

    pair = str(row.get("query_pair", "")).strip()
    return [part.strip() for part in pair.split("+") if part.strip()]


def _group_key(row) -> tuple:
    if "word_id" in row and pd.notna(row["word_id"]):
        return ("word_id", row["word_id"])
    if "word_step_start" in row and pd.notna(row["word_step_start"]):
        return ("step", int(row["word_step_start"]))
    if "step" in row and pd.notna(row["step"]):
        return ("step", int(row["step"]))
    return ("row", int(row.name))


def _group_label(group: pd.DataFrame) -> str:
    for column in ("word", "token", "token_group"):
        if column in group.columns:
            values = [str(v).strip() for v in group[column].tolist() if pd.notna(v) and str(v).strip()]
            if values:
                norms = [_norm_text(v) for v in values if _norm_text(v)]
                if norms and len(set(norms)) == 1:
                    return values[0]

    if "token" in group.columns:
        joined = "".join(_clean_token_piece(v) for v in group["token"].tolist() if pd.notna(v))
        if joined:
            return joined

    if "word" in group.columns:
        joined = "".join(_clean_token_piece(v) for v in group["word"].tolist() if pd.notna(v))
        if joined:
            return joined

    return ""


def _group_sort_key(group: pd.DataFrame):
    step_col = group["step"] if "step" in group.columns else pd.Series([0])
    step_end_col = group["step_end"] if "step_end" in group.columns else step_col
    return (
        int(step_col.min()),
        int(step_end_col.min()),
        len(group),
    )


def _pick_first_occurrence_rows(frame: pd.DataFrame, component: str) -> pd.DataFrame:
    target = _norm_text(component)
    if not target or frame.empty:
        return frame.iloc[0:0]

    grouped_rows = []
    for _, group in frame.groupby(frame.apply(_group_key, axis=1), sort=False):
        grouped_rows.append(group)

    grouped_rows.sort(key=_group_sort_key)

    for group in grouped_rows:
        label = _group_label(group)
        if _norm_text(label) == target:
            sort_cols = [col for col in ("step", "step_end", "word_id") if col in group.columns]
            if sort_cols:
                return group.sort_values(sort_cols, kind="stable")
            return group

    return frame.iloc[0:0]


def _row_label(row, fallback: str) -> str:
    for column in ("token", "word", "token_group"):
        if column in row and pd.notna(row[column]):
            text = str(row[column]).strip()
            if text:
                return text
    return fallback

for i, r in mode_b.iterrows():
    if i > 0:
        out_rows.append({c: "" for c in header})

    image_id = r["image"]
    mask_id = str(r["query_mask"])
    components = _split_query_components(r)
    pair_label = " + ".join(components) if components else str(r.get("query_pair", ""))

    out_rows.append({
        "image_id": image_id,
        "mask_id": mask_id,
        "token_group": pair_label,
        "obj_iou": r["obj_iou"],
        "iou_hard": r["iou_hard"],
        "io_ratio": r["io_ratio"],
        "wdp": r["wdp"],
        "func_iou": r["func_iou"],
        "f1_iou": r["f1_iou"],
    })

    image_col = mode_a["image"].astype(str)
    if "target" in mode_a.columns:
        target_col = mode_a["target"].astype(str)
        matching = mode_a[(image_col == str(image_id)) & (target_col == mask_id)]
    else:
        matching = mode_a[image_col == str(image_id)]

    seen_components = set()
    for component in components:
        component_norm = _norm_text(component)
        if not component_norm or component_norm in seen_components:
            continue
        seen_components.add(component_norm)

        first_rows = _pick_first_occurrence_rows(matching, component)
        if first_rows.empty:
            continue

        for _, r2 in first_rows.iterrows():
            out_rows.append({
                "image_id": image_id,
                "mask_id": r2.get("target", mask_id),
                "token_group": _row_label(r2, component),
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
