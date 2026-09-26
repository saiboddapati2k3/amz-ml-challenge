"""Generate a tiny FAKE dataset in the official layout, for smoke tests only.

python scripts/make_synthetic.py --out dataset_synth --n 400
Never mix this with real data; it only exists so the code can run before the
real dataset is copied in.
"""
import argparse
import os
import random

WORDS = ["royal", "star", "green", "apex", "zyphora", "metro", "sai", "om", "golden",
         "prime", "blue", "sunrise", "national", "global", "shree", "lotus"]
KINDS = ["bakery", "traders", "hospital", "electronics", "motors", "textiles", "pharma"]
LEGAL = ["pvt ltd", "private limited", "inc", "llc", "corp", "ltd", ""]
STREETS = ["mg road", "main street", "park avenue", "station rd", "lake view st"]
CITIES = {"India": ["mumbai", "delhi", "pune"], "US": ["austin", "boston", "denver"]}


def noisy(s, rng):
    s = s.replace(" road", " rd") if rng.random() < 0.4 else s
    if rng.random() < 0.3 and len(s) > 4:
        i = rng.randrange(len(s) - 1)
        s = s[:i] + s[i + 1] + s[i] + s[i + 2:]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset_synth")
    ap.add_argument("--n", type=int, default=400)
    args = ap.parse_args()
    rng = random.Random(0)
    for split in ("train", "test"):
        d = os.path.join(args.out, split)
        os.makedirs(d, exist_ok=True)
        s1, s2, s3, gt = [], [], [], []
        c2 = c3 = 0
        for i in range(args.n):
            country = rng.choice(["India", "US"] + (["France"] if split == "test" else []))
            city = rng.choice(CITIES.get(country, ["paris", "lyon"]))
            name = f"{rng.choice(WORDS)} {rng.choice(KINDS)}"
            addr = f"{rng.randint(1, 200)} {rng.choice(STREETS)}, {city} {rng.randint(10000, 999999)}"
            sid = f"S1-{i:05d}"
            s1.append((sid, f"{name} {rng.choice(LEGAL)}".strip().title(), addr, country))
            matches = []
            for _ in range(rng.choice([0, 0, 1, 1, 1, 2, 3])):
                if rng.random() < 0.5:
                    c2 += 1; mid = f"S2-{c2:05d}"; s2.append((mid, noisy(name, rng).upper(), noisy(addr, rng), country))
                else:
                    c3 += 1; mid = f"S3-{c3:05d}"; s3.append((mid, noisy(name, rng) + " " + rng.choice(LEGAL), noisy(addr, rng), country))
                matches.append(mid)
            gt.append((sid, ",".join(matches)))
        for _ in range(args.n // 2):  # distractors
            c2 += 1
            s2.append((f"S2-{c2:05d}", f"{rng.choice(WORDS)} {rng.choice(KINDS)}", f"{rng.randint(1, 200)} {rng.choice(STREETS)}", rng.choice(["India", "US"])))
        rng.shuffle(s2); rng.shuffle(s3)
        hdr = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        for n, rows in ((1, s1), (2, s2), (3, s3)):
            with open(os.path.join(d, f"{split}_source{n}.tsv"), "w", encoding="utf-8") as f:
                f.write(hdr + "".join("\t".join(r) + "\n" for r in rows))
        if split == "train":
            with open(os.path.join(d, "train_ground_truth.tsv"), "w", encoding="utf-8") as f:
                f.write("source1_entity_id\tmatched_entity_ids\n" + "".join(f"{a}\t{b}\n" for a, b in gt))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
