from pathlib import Path
from collections import Counter

# ---------------------------------------------------------
# DATASET PATH
# ---------------------------------------------------------

DATASET_DIR = Path(
    r"D:\BTP_Marker_Project\IMU_WordPredict_BTP\Dataset"
)


# ---------------------------------------------------------
# LENGTH BUCKETS
# ---------------------------------------------------------

def get_bucket(n):
    if 1 <= n <= 10:
        return "1–10"
    elif 11 <= n <= 20:
        return "11–20"
    elif 21 <= n <= 30:
        return "21–30"
    elif 31 <= n <= 39:
        return "31–39"
    elif 40 <= n <= 50:
        return "40–50"
    elif 51 <= n <= 60:
        return "51–60"
    elif 61 <= n <= 70:
        return "61–70"
    elif 71 <= n <= 80:
        return "71–80"
    elif 81 <= n <= 100:
        return "81–100"
    elif 101 <= n <= 150:
        return "101–150"
    elif 151 <= n <= 200:
        return "151–200"
    elif 201 <= n <= 300:
        return "201–300"
    else:
        return "300+"


# ---------------------------------------------------------
# COUNT FILE LENGTHS
# ---------------------------------------------------------

bucket_counts = Counter()
total_files = 0
total_rows = 0

lengths = []


for filepath in DATASET_DIR.rglob("*.txt"):

    try:
        with open(
            filepath,
            "r",
            encoding="utf-8",
            errors="replace"
        ) as f:

            # Count non-empty lines
            n_rows = sum(
                1
                for line in f
                if line.strip()
            )

        bucket_counts[get_bucket(n_rows)] += 1
        lengths.append(n_rows)

        total_files += 1
        total_rows += n_rows

    except Exception as e:
        print(f"[WARNING] Could not read {filepath}: {e}")


# ---------------------------------------------------------
# PRINT RESULT
# ---------------------------------------------------------

print()
print("=" * 45)
print("DATASET FILE LENGTH ANALYSIS")
print("=" * 45)

print(f"Dataset: {DATASET_DIR}")
print(f"Total files: {total_files}")
print(f"Total sensor rows: {total_rows}")

if lengths:
    print(f"Minimum rows/file: {min(lengths)}")
    print(f"Maximum rows/file: {max(lengths)}")
    print(f"Average rows/file: {sum(lengths) / len(lengths):.2f}")

print()
print("Length range       Number of files")
print("-----------------------------------")

buckets = [
    "1–10",
    "11–20",
    "21–30",
    "31–39",
    "40–50",
    "51–60",
    "61–70",
    "71–80",
    "81–100",
    "101–150",
    "151–200",
    "201–300",
    "300+",
]

for bucket in buckets:
    print(f"{bucket:<18} {bucket_counts[bucket]:>8}")


# ---------------------------------------------------------
# TRAINING FILTER SUMMARY
# ---------------------------------------------------------

valid_40_80 = sum(
    count
    for bucket, count in bucket_counts.items()
    if bucket in {"40–50", "51–60", "61–70", "71–80"}
)

invalid_below_40 = sum(
    count
    for bucket, count in bucket_counts.items()
    if bucket in {"1–10", "11–20", "21–30", "31–39"}
)

invalid_above_80 = sum(
    count
    for bucket, count in bucket_counts.items()
    if bucket in {"81–100", "101–150", "151–200", "201–300", "300+"}
)

print()
print("=" * 45)
print("CURRENT TRAINING FILTER: 40–80 ROWS")
print("=" * 45)

print(f"Valid (40–80):       {valid_40_80}")
print(f"Below 40:            {invalid_below_40}")
print(f"Above 80:            {invalid_above_80}")
print(f"Total:               {total_files}")

if total_files:
    print()
    print(
        f"Percentage retained: "
        f"{100 * valid_40_80 / total_files:.2f}%"
    )

    print(
        f"Percentage discarded: "
        f"{100 * (total_files - valid_40_80) / total_files:.2f}%"
    )