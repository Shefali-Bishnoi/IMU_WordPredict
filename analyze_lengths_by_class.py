from pathlib import Path
from collections import defaultdict

DATASET_DIR = Path(
    r"D:\BTP_Marker_Project\IMU_WordPredict_BTP\Dataset"
)


def classify_length(n):
    if n < 40:
        return "<40"
    elif n <= 80:
        return "40-80"
    else:
        return ">80"


# ---------------------------------------------------------
# character -> category -> count
# ---------------------------------------------------------

counts = defaultdict(lambda: {
    "<40": 0,
    "40-80": 0,
    ">80": 0,
})


total = 0


for filepath in DATASET_DIR.rglob("*.txt"):

    # Dataset structure:
    #
    # Dataset/
    #   capital letters/
    #       X/
    #           S01/
    #               X-01.txt
    #
    # filepath.parts[-3] = character directory

    try:
        char = filepath.parent.parent.name

        with open(
            filepath,
            "r",
            encoding="utf-8",
            errors="replace"
        ) as f:

            n_rows = sum(
                1
                for line in f
                if line.strip()
            )

        bucket = classify_length(n_rows)

        counts[char][bucket] += 1
        total += 1

    except Exception as e:
        print(f"[WARNING] {filepath}: {e}")


# ---------------------------------------------------------
# PRINT
# ---------------------------------------------------------

print()
print("=" * 65)
print("LENGTH DISTRIBUTION BY CHARACTER")
print("=" * 65)

print(
    f"{'Char':>4} "
    f"{'<40':>10} "
    f"{'40-80':>10} "
    f"{'>80':>10} "
    f"{'Total':>10}"
)

print("-" * 65)


for char in sorted(counts.keys()):

    below = counts[char]["<40"]
    valid = counts[char]["40-80"]
    above = counts[char][">80"]

    total_char = below + valid + above

    print(
        f"{char:>4} "
        f"{below:>10} "
        f"{valid:>10} "
        f"{above:>10} "
        f"{total_char:>10}"
    )


print("-" * 65)

overall_below = sum(v["<40"] for v in counts.values())
overall_valid = sum(v["40-80"] for v in counts.values())
overall_above = sum(v[">80"] for v in counts.values())

print(
    f"{'ALL':>4} "
    f"{overall_below:>10} "
    f"{overall_valid:>10} "
    f"{overall_above:>10} "
    f"{total:>10}"
)