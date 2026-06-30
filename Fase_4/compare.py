"""
Token Activation Map - Comparison Script
========================================

Input format:

proposed row
last_token row

(blank row)

primary_token row

(blank row)

secondary_token row

------------------------------------------------------------

Output:
    confronto.csv
    proposed_vs_last_token_pie.png
    proposed_vs_primary_token_pie.png
    proposed_vs_secondary_token_pie.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------------------------
# 1. FILES
# ---------------------------------------------------------------------------

INPUT_FILE = "results.csv"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILES = {
    "all": "confronto_all_relations.csv",
    1: "confronto_object_attribute.csv",
    2: "confronto_object_action.csv",
    3: "confronto_spatial_relation.csv",
}

# ---------------------------------------------------------------------------
# 2. METRICS
# ---------------------------------------------------------------------------

METRICS = [
    "obj_iou",
    "iou_hard",
    "io_ratio",
    "wdp",
    "func_iou",
    "f1_iou",
]

# ---------------------------------------------------------------------------
# 3. LOAD CSV
# ---------------------------------------------------------------------------

df = pd.read_csv(INPUT_FILE, dtype=str)

# ---------------------------------------------------------------------------
# 4. SPLIT INTO RELATION BLOCKS
# ---------------------------------------------------------------------------

blocks = []
current = []

for _, row in df.iterrows():

    image_value = str(row["image_id"])

    # dashed separator row
    if image_value.startswith("-"):
        if current:
            blocks.append(pd.DataFrame(current))
            current = []
        continue

    # blank separator row
    if pd.isna(row["image_id"]) or image_value == "nan":
        continue

    current.append(row)

if current:
    blocks.append(pd.DataFrame(current))

print(f"Found {len(blocks)} relation blocks")

# ---------------------------------------------------------------------------
# 5. EXTRACT PROPOSED / BASELINES
# ---------------------------------------------------------------------------

row_labels = [
    "proposed",
    "last_token",
    "primary_token",
    "secondary_token",
]

frames = {
    "all": {label: [] for label in row_labels},
    1: {label: [] for label in row_labels},
    2: {label: [] for label in row_labels},
    3: {label: [] for label in row_labels},
}

for block_idx, block in enumerate(blocks):

    # Need at least:
    # proposed
    # last token
    # primary token
    # secondary token
    if len(block) < 4:
        print(
            f"[WARNING] block {block_idx} has "
            f"{len(block)} rows (< 4) - skipping."
        )
        continue

    # -----------------------------------------------------------------------
    # BLOCK STRUCTURE
    #
    # row 0 = proposed
    # row 1 = last token baseline
    # row 2 = primary token baseline
    # row N = last secondary subtoken baseline
    #
    # Example:
    #
    # proposed
    # last_token
    # primary_token
    # fr
    # is
    # bee
    #
    # -> secondary_token = bee
    # -----------------------------------------------------------------------

    proposed = block.iloc[0]
    last_token = block.iloc[1]
    primary = block.iloc[2]

    # keep ONLY the last subtoken
    secondary = block.iloc[-1]

    relation_type = int(proposed["mask_id"])

    if relation_type not in (1, 2, 3):
        print(f"[WARNING] Unknown relation type {relation_type}")
        continue

    sample_id = (
        f"{proposed['image_id']}_"
        f"{proposed['mask_id']}"
    )

    rows = {
        "proposed": proposed,
        "last_token": last_token,
        "primary_token": primary,
        "secondary_token": secondary,
    }

    for label, row in rows.items():

        record = {
            "sample_id": sample_id,
            "image_id": proposed["image_id"],
            "mask_id": proposed["mask_id"],
        }

        for metric in METRICS:
            record[metric] = pd.to_numeric(
                row[metric],
                errors="coerce"
            )

        frames[relation_type][label].append(record)
        frames["all"][label].append(record)

comparisons = {
    "proposed_vs_last_token": (
        "proposed",
        "last_token",
    ),
    "proposed_vs_primary_token": (
        "proposed",
        "primary_token",
    ),
    "proposed_vs_secondary_token": (
        "proposed",
        "secondary_token",
    ),
}

relation_names = {
    "all": "all_relations",
    1: "object_attribute",
    2: "object_action",
    3: "spatial_relation",
}

for relation_type in relation_names:

    relation_dir = os.path.join(
    OUTPUT_DIR,
    relation_names[relation_type]
    )

    pie_dir = os.path.join(relation_dir, "pie")
    hist_dir = os.path.join(relation_dir, "histograms")

    os.makedirs(pie_dir, exist_ok=True)
    os.makedirs(hist_dir, exist_ok=True)

    os.makedirs(relation_dir, exist_ok=True)

    print()
    print("=" * 70)
    print(f"Processing {relation_names[relation_type]}")
    print("=" * 70)

    dfs = {}

    empty = False

    for label in row_labels:

        if len(frames[relation_type][label]) == 0:
            empty = True
            break

        dfs[label] = (
            pd.DataFrame(frames[relation_type][label])
            .set_index("sample_id")
        )

    if empty:
        print("No samples.")
        continue

    output_rows = []
    q1_rows = []
    median_rows = []
    q3_rows = []

    for comp_name, (a, b) in comparisons.items():

        diff = dfs[a][METRICS] - dfs[b][METRICS]

        q1_row = {"comparison": comp_name}
        median_row = {"comparison": comp_name}
        q3_row = {"comparison": comp_name}

        # relazione con il massimo gain di F1
        best_idx = diff["f1_iou"].idxmax()

        best_gain = diff.loc[best_idx, "f1_iou"]

        best_image = dfs[a].loc[best_idx, "image_id"]
        best_mask = dfs[a].loc[best_idx, "mask_id"]

        print(
            f"[{relation_names[relation_type]}] "
            f"{comp_name}: "
            f"best F1 gain = {best_gain:.4f} "
            f"(image_id={best_image}, mask_id={best_mask})"
        )

        row = {"comparison": comp_name}

        for metric in METRICS:
            values = diff[metric].dropna()

            row[f"avg_gain_{metric}"] = values.mean()

            q1_row[metric] = values.quantile(0.25)
            median_row[metric] = values.median()
            q3_row[metric] = values.quantile(0.75)

        total_relations = len(diff)

# ------------------------------------------------------------------
# Equal relations:
# all metrics must be exactly equal
# ------------------------------------------------------------------

        equal_mask = (diff[METRICS] == 0).all(axis=1)
        equal_relations = equal_mask.sum()

        row["equal_relations"] = equal_relations
        row["total_relations"] = total_relations

        # ------------------------------------------------------------------
        # Statistics for every metric
        # ------------------------------------------------------------------

        for metric in METRICS:

            better_relations = (diff[metric] > 0).sum()
            worse_relations = (diff[metric] < 0).sum()

            row[f"proposed_{metric}_better"] = (
                f"{100 * better_relations / total_relations:.2f}%"
                if total_relations > 0 else "0.00%"
            )

            row[f"proposed_{metric}_equal"] = (
                f"{100 * equal_relations / total_relations:.2f}%"
                if total_relations > 0 else "0.00%"
            )

            row[f"proposed_{metric}_worse"] = (
                f"{100 * worse_relations / total_relations:.2f}%"
                if total_relations > 0 else "0.00%"
            )

            row[f"{metric}_better_relations"] = better_relations
            row[f"{metric}_worse_relations"] = worse_relations

            plt.figure(figsize=(6, 6))

            plt.pie(
                [
                    better_relations,
                    equal_relations,
                    worse_relations,
                ],
                labels=[
                    f"Better\n({better_relations})",
                    f"Equal\n({equal_relations})",
                    f"Worse\n({worse_relations})",
                ],
                autopct="%1.1f%%",
            )

            plt.title(
                f"{relation_names[relation_type]}\n"
                f"{comp_name.replace('_', ' ').title()}\n"
                f"{metric}"
            )

            filename = f"{comp_name}_{metric}_pie.png"

            filepath = os.path.join(
                relation_dir,
                filename
            )

            plt.savefig(
                filepath,
                bbox_inches="tight"
            )

            print(f"Saved {filepath}")

            plt.close()

            print(f"Saved {filename}")

            plt.figure(figsize=(8,5))

            plt.hist(
                values,
                bins=20,
            )

            plt.xlabel(f"Gain ({metric})")
            plt.ylabel("Number of relations")

            plt.title(
                f"{relation_names[relation_type]}\n"
                f"{comp_name.replace('_',' ').title()}\n"
                f"{metric}"
            )

            plt.grid(axis="y", alpha=0.3)

            hist_filename = (
                f"{comp_name}_{metric}_hist.png"
            )

            hist_filepath = os.path.join(
                hist_dir,
                hist_filename,
            )

            plt.savefig(
                hist_filepath,
                bbox_inches="tight",
            )

            plt.close()

            print(f"Saved {hist_filepath}")

        output_rows.append(row)
        q1_rows.append(q1_row)
        median_rows.append(median_row)
        q3_rows.append(q3_row)

    output = pd.DataFrame(output_rows)

    output.set_index("comparison", inplace=True)

    output_path = os.path.join(
    relation_dir,
    OUTPUT_FILES[relation_type]
)

    output.to_csv(output_path)
    q1_df = pd.DataFrame(q1_rows).set_index("comparison")
    median_df = pd.DataFrame(median_rows).set_index("comparison")
    q3_df = pd.DataFrame(q3_rows).set_index("comparison")

    q1_df.to_csv(os.path.join(relation_dir, "q1.csv"))
    median_df.to_csv(os.path.join(relation_dir, "median.csv"))
    q3_df.to_csv(os.path.join(relation_dir, "q3.csv"))

    print(f"Saved {os.path.join(relation_dir, 'q1.csv')}")
    print(f"Saved {os.path.join(relation_dir, 'median.csv')}")
    print(f"Saved {os.path.join(relation_dir, 'q3.csv')}")

    print(f"Saved {output_path}")
    print(output)
