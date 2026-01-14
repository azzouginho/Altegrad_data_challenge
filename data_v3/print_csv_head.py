import argparse
import csv
from pathlib import Path
import pandas as pd


def print_head(csv_path: Path, n_rows: int, split=None) -> None:
    df = pd.read_csv(csv_path)
    if split == 'test':
        for desc in df[:10]["description"]:
            print(f'description : {desc}', end='\n\n')
    elif split == 'train':
        for desc, ground_truth in zip(df[:10]["description"], df[:10]["ground_truth"]):
            print(f'description : {desc}\nground_truth : {ground_truth}', end='\n\n')

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print the first N rows of a CSV file."
    )
    parser.add_argument("path", type=Path, help="Path to the CSV file")
    parser.add_argument(
        "-n",
        "--rows",
        type=int,
        default=10,
        help="Number of rows to print (default: 10)",
    )
    parser.add_argument("--split", default='test')
    args = parser.parse_args()

    if not args.path.exists():
        raise SystemExit(f"File not found: {args.path}")

    print_head(args.path, args.rows, args.split)


if __name__ == "__main__":
    main()