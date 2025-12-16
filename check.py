#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import pandas as pd


def analyze_csv(path: Path, sep: str, encoding: str, chunksize: int) -> pd.DataFrame:
    """
    Zwraca tabelę z brakami danych per kolumna:
    - missing_count: liczba braków (NaN + puste/whitespace w kolumnach tekstowych)
    - missing_pct: procent braków
    - dtype: typ kolumny (z pierwszego chunku lub całego pliku)
    """
    total_rows = 0
    missing_counts = None
    dtypes = None

    def process_chunk(chunk: pd.DataFrame):
        nonlocal total_rows, missing_counts, dtypes

        if dtypes is None:
            dtypes = chunk.dtypes.astype(str)

        total_rows += len(chunk)

        # Braki "klasyczne" (NaN)
        miss = chunk.isna()

        # Dodatkowo: puste stringi / same spacje w kolumnach tekstowych
        obj_cols = chunk.select_dtypes(include=["object", "string"]).columns
        if len(obj_cols) > 0:
            # Uwaga: .astype("string") daje bezpieczniejszą obsługę braków
            stripped_empty = (
                chunk[obj_cols]
                .astype("string")
                .apply(lambda s: s.str.strip().eq(""))
            )
            # Nie podwajaj, jeśli już jest NaN
            miss.loc[:, obj_cols] = miss.loc[:, obj_cols] | stripped_empty

        miss_sum = miss.sum(axis=0)

        if missing_counts is None:
            missing_counts = miss_sum
        else:
            missing_counts = missing_counts.add(missF if False else miss_sum, fill_value=0)

    # Czytanie w chunkach (dla dużych plików) albo całościowo
    if chunksize and chunksize > 0:
        reader = pd.read_csv(path, sep=sep, encoding=encoding, chunksize=chunksize, low_memory=False)
        for chunk in reader:
            process_chunk(chunk)
    else:
        df = pd.read_csv(path, sep=sep, encoding=encoding, low_memory=False)
        process_chunk(df)

    # Raport
    missing_counts = missing_counts.astype(int)
    report = pd.DataFrame({
        "column": missing_counts.index,
        "missing_count": missing_counts.values,
    })
    report["missing_pct"] = (report["missing_count"] / max(total_rows, 1) * 100.0).round(4)

    if dtypes is not None:
        report["dtype"] = report["column"].map(dtypes.to_dict())
    else:
        report["dtype"] = None

    report = report.sort_values(["missing_count", "column"], ascending=[False, True]).reset_index(drop=True)
    report.attrs["total_rows"] = total_rows
    report.attrs["file"] = str(path)
    return report


def iter_csv_files(input_path: Path):
    if input_path.is_file() and input_path.suffix.lower() == ".csv":
        yield input_path
        return
    if input_path.is_dir():
        yield from sorted(input_path.glob("*.csv"))
        return
    raise FileNotFoundError(f"Nie znaleziono pliku CSV ani katalogu: {input_path}")


def main():
    ap = argparse.ArgumentParser(description="Sprawdza braki danych w CSV (NaN + puste/whitespace).")
    ap.add_argument("input", help="Ścieżka do pliku CSV albo katalogu z plikami CSV.")
    ap.add_argument("--sep", default=",", help="Separator CSV (domyślnie ',').")
    ap.add_argument("--encoding", default="utf-8", help="Kodowanie pliku (domyślnie utf-8).")
    ap.add_argument("--chunksize", type=int, default=200_000,
                    help="Rozmiar chunka (0 = wczytaj całość). Domyślnie 200000.")
    ap.add_argument("--out", default=None,
                    help="Opcjonalnie: ścieżka do zapisu raportu CSV (dla katalogu dopisze nazwy plików).")
    args = ap.parse_args()

    input_path = Path(args.input)

    for csv_path in iter_csv_files(input_path):
        report = analyze_csv(csv_path, sep=args.sep, encoding=args.encoding, chunksize=args.chunksize)

        total_rows = report.attrs.get("total_rows", 0)
        cols_with_missing = int((report["missing_count"] > 0).sum())
        total_cols = len(report)

        print("\n" + "=" * 80)
        print(f"Plik: {csv_path}")
        print(f"Wiersze: {total_rows}, Kolumny: {total_cols}, Kolumny z brakami: {cols_with_missing}")

        if cols_with_missing == 0:
            print("Brak braków danych (NaN/puste/whitespace) w kolumnach.")
        else:
            print("\nTop kolumny z brakami:")
            print(report.loc[report["missing_count"] > 0, ["column", "missing_count", "missing_pct", "dtype"]].head(20).to_string(index=False))

        # Zapis raportu
        if args.out:
            out_path = Path(args.out)
            if input_path.is_dir():
                out_file = out_path.parent / f"{csv_path.stem}_missing_report.csv" if out_path.suffix.lower() != ".csv" else out_path.parent / f"{csv_path.stem}_{out_path.name}"
            else:
                out_file = out_path
            out_file.parent.mkdir(parents=True, exist_ok=True)
            report.to_csv(out_file, index=False, encoding="utf-8")
            print(f"\nZapisano raport: {out_file}")

    print("\nGotowe.")


if __name__ == "__main__":
    main()
