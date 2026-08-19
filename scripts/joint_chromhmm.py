"""Concatenated ChromHMM input for two replicates.

concat - copy the binarized files of both replicates into a single directory,
         keeping the mark names as they are and only renaming the cell to the
         replicate name.  ChromHMM LearnModel over that directory learns a
         single model on the concatenated signal of both replicates (every
         replicate chromosome stays its own sequence, so no artificial
         transitions are introduced) and writes a segmentation per replicate in
         the shared state space directly, so no model splitting is needed
         afterwards.

Only marks present in both replicates are kept, in a single common column
order, since one model requires the same columns in every input file.
"""
import sys
import os
import argparse
import gzip

def open_text(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")

def binarized_files(binarized_dir):
    """Binarized file name -> path of a single replicate."""
    return {f: os.path.join(binarized_dir, f) for f in os.listdir(binarized_dir)
            if f.endswith("_binary.txt") or f.endswith("_binary.txt.gz")}

def collect(binarized_dir):
    """Chromosome -> (path, marks) of a single replicate."""
    result = {}
    for f_name, path in sorted(binarized_files(binarized_dir).items()):
        with open_text(path) as fh:
            cell_chrom = fh.readline().rstrip("\n").split("\t")
            marks = fh.readline().rstrip("\n").split("\t")
        if len(cell_chrom) < 2 or not marks:
            print(f"Skipping malformed binarized file {path}")
            continue
        chrom = cell_chrom[1]
        if chrom in result:
            sys.exit(f"Chromosome {chrom} is present twice in {binarized_dir}")
        result[chrom] = (path, marks)
    if not result:
        sys.exit(f"No binarized files found in {binarized_dir}")
    return result

def concat(args):
    os.makedirs(args.outdir, exist_ok=True)

    reps = {rep: collect(binarized) for rep, binarized in
            (("rep1", args.rep1), ("rep2", args.rep2))}

    # Marks in the order of the first replicate, restricted to marks binarized
    # in both replicates.
    marks1 = next(iter(reps["rep1"].values()))[1]
    marks2 = set(next(iter(reps["rep2"].values()))[1])
    marks = [m for m in marks1 if m in marks2]
    if not marks:
        sys.exit(f"No common marks in {args.rep1} and {args.rep2}")
    dropped = sorted((set(marks1) | marks2) - set(marks))
    if dropped:
        print(f"Marks {','.join(dropped)} are missing in one of the replicates and are dropped")

    chroms = sorted(set(reps["rep1"]) & set(reps["rep2"]))
    if not chroms:
        sys.exit(f"No common chromosomes in {args.rep1} and {args.rep2}")
    skipped = sorted((set(reps["rep1"]) | set(reps["rep2"])) - set(chroms))
    if skipped:
        print(f"Chromosomes {','.join(skipped)} are missing in one of the replicates and are skipped")

    for rep, files in reps.items():
        for chrom in chroms:
            path, file_marks = files[chrom]
            missing = [m for m in marks if m not in file_marks]
            if missing:
                sys.exit(f"Marks {','.join(missing)} are not present in {path}")
            columns = [file_marks.index(m) for m in marks]

            out_p = os.path.join(args.outdir, f"{rep}_{chrom}_binary.txt")
            with open_text(path) as fh, open(out_p, "w") as out:
                fh.readline()  # Cell Chrom
                fh.readline()  # Marks
                out.write(f"{rep}\t{chrom}\n")
                out.write("\t".join(marks) + "\n")
                for line in fh:
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) < len(file_marks):
                        continue  # trailing empty line
                    out.write("\t".join(fields[c] for c in columns) + "\n")

    print(f"Concatenated {len(reps)} replicates x {len(chroms)} chromosomes x "
          f"{len(marks)} marks into {args.outdir}")

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_concat = subparsers.add_parser(
        "concat", help="Concatenate the binarized files of both replicates for the same marks")
    p_concat.add_argument("--rep1", required=True, help="Replicate 1 binarized directory")
    p_concat.add_argument("--rep2", required=True, help="Replicate 2 binarized directory")
    p_concat.add_argument("--outdir", required=True, help="Output directory")
    p_concat.set_defaults(func=concat)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
