import os
import re
import csv
import argparse


def extract_accuracies_exp1(folder, output_csv='temp.csv'):
    pattern = re.compile(r'\[Online Eval\].*\[Accuracy:([\d.]+)\]')

    txt_files = sorted(
        [f for f in os.listdir(folder) if f.endswith('.txt')],
        key=lambda f: [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', f)]
    )

    accuracies = []
    for fname in txt_files:
        fpath = os.path.join(folder, fname)
        last_acc = None
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    last_acc = float(m.group(1))
        accuracies.append((fname, last_acc))
        status = f'{last_acc:.4f}' if last_acc is not None else 'NOT FOUND'
        print(f'{fname}: {status}')

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([fname for fname, _ in accuracies])
        writer.writerow([f'{acc:.4f}' if acc is not None else 'NOT FOUND' for _, acc in accuracies])

    acc_list = [acc for _, acc in accuracies]
    print(f'\nAccuracy list:\n{acc_list}')
    print(f'\nResults saved to {output_csv}')
    return acc_list


def extract_accuracies_exp2(folder, output_csv='temp.csv'):
    pattern = re.compile(r'\[PerCorruption\]\[(\w+)\]\[NumSamples:\d+\]\[Accuracy:([\d.]+)\]')

    txt_files = sorted(
        [f for f in os.listdir(folder) if f.endswith('.txt')],
        key=lambda f: [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', f)]
    )

    # corruption names in order (used as column headers)
    corruption_names = None
    rows = []

    for fname in txt_files:
        fpath = os.path.join(folder, fname)
        per_corruption = {}
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    per_corruption[m.group(1)] = float(m.group(2))
        if corruption_names is None and per_corruption:
            corruption_names = list(per_corruption.keys())
        acc_row = [f'{per_corruption.get(c, None):.4f}' if per_corruption.get(c) is not None else 'NOT FOUND'
                   for c in (corruption_names or [])]
        rows.append((fname, acc_row))
        print(f'{fname}: {acc_row}')

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['file'] + (corruption_names or []))
        for fname, acc_row in rows:
            writer.writerow([fname] + acc_row)

    print(f'\nResults saved to {output_csv}')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str, help='Folder containing txt log files')
    parser.add_argument('--output', type=str, default='temp.csv', help='Output CSV file path')
    parser.add_argument('--exp', type=int, default=1, choices=[1, 2], help='Experiment type: 1 or 2')
    args = parser.parse_args()
    if args.exp == 2:
        extract_accuracies_exp2(args.folder, args.output)
    else:
        extract_accuracies_exp1(args.folder, args.output)
