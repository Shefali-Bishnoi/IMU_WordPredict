from pathlib import Path
from collections import defaultdict
import csv

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path(
    r"D:\BTP_Marker_Project\IMU_WordPredict_BTP\Dataset"
)

# Candidate ranges to compare
RANGES = [
    (20, 100),
    (30, 100),
    (40, 80),      # current
    (30, 120),
    (40, 100),
    (20, 150),
]

# File extensions to consider
VALID_EXTENSIONS = {".txt"}

# ============================================================
# CHARACTER EXTRACTION
# ============================================================

def get_character(filepath):
    """
    Expected structure:

    Dataset/
        capital letters/
            A/
                S01/
                    A-01.txt

        small letters/
            a/
                S01/
                    a-01.txt

    We take the character directory immediately
    above the subject directory.
    """

    parts = filepath.parts

    # Find S01, S02, etc.
    for i, part in enumerate(parts):

        if part.startswith("S") and part[1:].isdigit():

            if i >= 1:
                return parts[i - 1]

    return None


# ============================================================
# COUNT RAW SENSOR ROWS
# ============================================================

def count_rows(filepath):
    """
    Count non-empty data lines.

    Assumes each non-empty line corresponds to
    one sensor row.
    """

    try:

        with open(
            filepath,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:

            count = 0

            for line in f:

                line = line.strip()

                if line:
                    count += 1

            return count

    except Exception as e:

        print(f"[WARNING] Could not read {filepath}")
        print(f"         {e}")

        return None


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("COMPARING DATASET LENGTH FILTERS")
print("=" * 70)

print(f"Dataset: {DATASET_DIR}")
print()

# ------------------------------------------------------------
# Find files
# ------------------------------------------------------------

files = [
    p
    for p in DATASET_DIR.rglob("*")
    if p.is_file()
    and p.suffix.lower() in VALID_EXTENSIONS
]

print(f"Total files found: {len(files):,}")
print()

# ============================================================
# COLLECT DATA
# ============================================================

# character -> list of lengths
lengths_by_char = defaultdict(list)

# character -> number of files
total_by_char = defaultdict(int)

total_files = 0
total_rows = 0

print("Scanning files...")

for n, filepath in enumerate(files, start=1):

    char = get_character(filepath)

    if char is None:
        continue

    rows = count_rows(filepath)

    if rows is None:
        continue

    lengths_by_char[char].append(rows)

    total_by_char[char] += 1

    total_files += 1
    total_rows += rows

    if n % 10000 == 0:
        print(f"Processed: {n:,}/{len(files):,}")


# ============================================================
# SORT CHARACTERS
# ============================================================

characters = sorted(
    lengths_by_char.keys(),
    key=lambda x: (
        x.islower(),
        x.lower()
    )
)

# ============================================================
# BASIC DATASET STATISTICS
# ============================================================

all_lengths = []

for char in characters:
    all_lengths.extend(lengths_by_char[char])

print()
print("=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print(f"Total files:        {total_files:,}")
print(f"Total sensor rows:  {total_rows:,}")

if all_lengths:

    print(f"Minimum rows/file:  {min(all_lengths):,}")
    print(f"Maximum rows/file:  {max(all_lengths):,}")
    print(f"Average rows/file:  {sum(all_lengths) / len(all_lengths):.2f}")

print()

# ============================================================
# OVERALL RANGE COMPARISON
# ============================================================

print("=" * 90)
print("OVERALL FILTER COMPARISON")
print("=" * 90)

print(
    f"{'Range':<15}"
    f"{'Files':>12}"
    f"{'Retained':>12}"
    f"{'Retention %':>14}"
    f"{'Discarded':>12}"
    f"{'Discarded %':>14}"
)

print("-" * 90)

overall_results = []

for low, high in RANGES:

    retained = sum(
        1
        for length in all_lengths
        if low <= length <= high
    )

    discarded = total_files - retained

    retention_pct = (
        retained / total_files * 100
        if total_files
        else 0
    )

    discarded_pct = (
        discarded / total_files * 100
        if total_files
        else 0
    )

    print(
        f"{f'{low}–{high}':<15}"
        f"{total_files:>12,}"
        f"{retained:>12,}"
        f"{retention_pct:>13.2f}%"
        f"{discarded:>12,}"
        f"{discarded_pct:>13.2f}%"
    )

    overall_results.append({
        "range": f"{low}-{high}",
        "low": low,
        "high": high,
        "total": total_files,
        "retained": retained,
        "retention_pct": retention_pct,
        "discarded": discarded,
        "discarded_pct": discarded_pct,
    })


# ============================================================
# PER CHARACTER COMPARISON
# ============================================================

print()
print("=" * 110)
print("RETENTION % BY CHARACTER")
print("=" * 110)

# Header

header = f"{'Char':>5}"

for low, high in RANGES:
    header += f"{f'{low}-{high}':>13}"

header += f"{'Total':>10}"

print(header)
print("-" * 110)

per_char_results = []

for char in characters:

    lengths = lengths_by_char[char]

    row = f"{char:>5}"

    result = {
        "character": char,
        "total": len(lengths),
    }

    for low, high in RANGES:

        retained = sum(
            1
            for length in lengths
            if low <= length <= high
        )

        pct = (
            retained / len(lengths) * 100
            if lengths
            else 0
        )

        row += f"{pct:>12.2f}%"

        result[f"{low}-{high}_count"] = retained
        result[f"{low}-{high}_pct"] = pct

    row += f"{len(lengths):>10,}"

    print(row)

    per_char_results.append(result)


# ============================================================
# PER CHARACTER COUNTS
# ============================================================

print()
print("=" * 110)
print("RETAINED FILE COUNT BY CHARACTER")
print("=" * 110)

header = f"{'Char':>5}"

for low, high in RANGES:
    header += f"{f'{low}-{high}':>13}"

header += f"{'Total':>10}"

print(header)
print("-" * 110)

for char in characters:

    lengths = lengths_by_char[char]

    row = f"{char:>5}"

    for low, high in RANGES:

        retained = sum(
            1
            for length in lengths
            if low <= length <= high
        )

        row += f"{retained:>13,}"

    row += f"{len(lengths):>10,}"

    print(row)


# ============================================================
# BEST / WORST CHARACTER RETENTION
# ============================================================

print()
print("=" * 110)
print("CHARACTER-LEVEL IMPACT")
print("=" * 110)

for low, high in RANGES:

    values = []

    for char in characters:

        lengths = lengths_by_char[char]

        retained = sum(
            1
            for length in lengths
            if low <= length <= high
        )

        pct = retained / len(lengths) * 100

        values.append((char, pct))

    values.sort(key=lambda x: x[1])

    worst_char, worst_pct = values[0]
    best_char, best_pct = values[-1]

    avg_pct = sum(x[1] for x in values) / len(values)

    print()
    print(f"Range: {low}–{high}")
    print(f"Average character retention : {avg_pct:.2f}%")
    print(
        f"Lowest retention            : "
        f"{worst_char} ({worst_pct:.2f}%)"
    )
    print(
        f"Highest retention           : "
        f"{best_char} ({best_pct:.2f}%)"
    )


# ============================================================
# CURRENT FILTER VS ALTERNATIVES
# ============================================================

CURRENT = (40, 80)

current_low, current_high = CURRENT

current_retained = sum(
    1
    for length in all_lengths
    if current_low <= length <= current_high
)

print()
print("=" * 110)
print("CURRENT FILTER: 40–80")
print("=" * 110)

print(
    f"Current retained: "
    f"{current_retained:,} / {total_files:,}"
)

print(
    f"Current retention: "
    f"{current_retained / total_files * 100:.2f}%"
)

print()

print("Comparison with alternatives:")
print()

for result in overall_results:

    if (
        result["low"] == 40
        and result["high"] == 80
    ):
        continue

    difference = (
        result["retained"]
        - current_retained
    )

    print(
        f"{result['range']:>10} : "
        f"{result['retained']:>8,} files "
        f"({result['retention_pct']:.2f}%) "
        f"| "
        f"{difference:+,} vs 40–80"
    )


# ============================================================
# SAVE CSV - OVERALL
# ============================================================

overall_csv = Path("length_filter_overall_comparison.csv")

with open(
    overall_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "range",
            "low",
            "high",
            "total",
            "retained",
            "retention_pct",
            "discarded",
            "discarded_pct",
        ]
    )

    writer.writeheader()
    writer.writerows(overall_results)


# ============================================================
# SAVE CSV - PER CHARACTER
# ============================================================

char_csv = Path("length_filter_character_comparison.csv")

fieldnames = ["character", "total"]

for low, high in RANGES:

    fieldnames.append(
        f"{low}-{high}_count"
    )

    fieldnames.append(
        f"{low}-{high}_pct"
    )

with open(
    char_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(per_char_results)


# ============================================================
# SAVE RAW LENGTH DATA
# ============================================================

raw_csv = Path("dataset_file_lengths.csv")

with open(
    raw_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "character",
        "length"
    ])

    for char in characters:

        for length in lengths_by_char[char]:

            writer.writerow([
                char,
                length
            ])


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 110)
print("FILES SAVED")
print("=" * 110)

print(f"1. {overall_csv}")
print(f"2. {char_csv}")
print(f"3. {raw_csv}")

print()
print("Analysis complete.")