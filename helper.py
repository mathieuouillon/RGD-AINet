import re
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import polars as pl
import matplotlib.pyplot as plt


def _read_columns(filepath: str, columns: Tuple[str, ...], skip_rows: int = 1, sep: str = ",") -> Tuple[np.ndarray, ...]:
    """
    Generic helper to read specific columns from a whitespace-delimited file
    using Polars, skipping `skip_rows` header lines.

    Args:
    -------
    filepath (str): Path to the file.
    columns (Tuple[str, ...]): Column names to read.
    skip_rows (int): Number of header rows to skip.
    sep (str): Column separator.

    Returns:
    -------
    Tuple[np.ndarray, ...]: Tuple of numpy arrays for each column.
    """
    df = pl.read_csv(filepath, separator=sep, skip_rows=skip_rows)
    return tuple(df[col].to_numpy() for col in columns)


def read_data_points(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read x, y, and yerror columns from a data-points file.

    Args:
    -------
    filepath (str): Path to the file.

    Returns:
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]: x, y, and yerror arrays.

    """
    return _read_columns(filepath, ("#x", "y", "yerror"))


def read_lines(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read x and y columns from a lines file.

    Args:
    -------
    filepath (str): Path to the file.

    Returns:
    -------
    Tuple[np.ndarray, np.ndarray]: x and y arrays.
    """
    return _read_columns(filepath, ("#x", "y"))


def read_parameters(filepath: str) -> Tuple[float, float, float, float]:
    """
    Extract p0 and p1 (and their uncertainties) from a parameter file.

    Expects lines like:
        p0: 1.234e-1 ± 5.6e-3
        p1: 7.890e+0 ± 1.2e-1

    Args:
    -------
    filepath (str): Path to the parameter file.

    Returns
    -------
    p0 : float - Intercept
    p1 : float - Slope
    err_p0 : float - Uncertainty on p0
    err_p1 : float - Uncertainty on p1
    """
    with open(filepath, "r") as file:
        lines = file.readlines()

    # Filter lines that start with 'p0' or 'p1' and extract numbers
    results = {}
    for line in lines:
        if line.strip().startswith("p0") or line.strip().startswith("p1"):
            key = line.strip().split(":")[0]
            numbers = re.findall(r"[-+]?\d*\.\d+e[+-]?\d+|[-+]?\d*\.\d+|[-+]?\d+", line)
            results[key] = [float(num) for num in numbers]

    p0 = results["p0 "][1]
    p1 = results["p1 "][1]
    err_p0 = results["p0 "][2]
    err_p1 = results["p1 "][2]
    return p0, p1, err_p0, err_p1


def read_values(base: str, kind: str) -> Dict[str, Tuple[np.ndarray, ...]]:
    """
    Read three categories of data (conventional, ai, ratio) for a given base path and kind.

    Files for each category are:
      - conventional:
          data:     `{base}_0_{kind}conventional.txt`
          line:     `{base}_1_f{kind}conventional.txt`
          params:   `{base}_1_f{kind}conventional_pars.txt`
      - ai:
          data:     `{base}_2_{kind}ai.txt`
          line:     `{base}_3_f{kind}ai.txt`
          params:   `{base}_3_f{kind}ai_pars.txt`
      - ratio:
          data:     `{base}_4_{kind}.txt`
          line:     `{base}_5_f{kind}.txt`
          params:   `{base}_5_f{kind}_pars.txt`

    Returns
    -------
    results : dict
        Keys are "conventional", "ai", "ratio". Each value is a tuple of:
           (x_data, y_data, y_error,
            x_line, y_line,
            p0, p1, err_p0, err_p1)
    """

    suffixes = {
        "conventional": ("_0_{k}conventional.txt", "_1_f{k}conventional.txt", "_1_f{k}conventional_pars.txt"),
        "ai": ("_2_{k}ai.txt", "_3_f{k}ai.txt", "_3_f{k}ai_pars.txt"),
        "ratio": ("_4_{k}.txt", "_5_f{k}.txt", "_5_f{k}_pars.txt"),
    }

    results: Dict[str, Tuple[np.ndarray, ...]] = {}
    for category, (d_suf, l_suf, p_suf) in suffixes.items():
        data_path = f"{base}{d_suf.format(k=kind)}"
        line_path = f"{base}{l_suf.format(k=kind)}"
        pars_path = f"{base}{p_suf.format(k=kind)}"

        x_data, y_data, y_err = read_data_points(data_path)
        x_line, y_line = read_lines(line_path)
        p0, p1, err_p0, err_p1 = read_parameters(pars_path)

        results[category] = (x_data, y_data, y_err, x_line, y_line, p0, p1, err_p0, err_p1)

    return results


def find_base(directory: str, kind: str) -> str:
    """
    Locate one '*_{kind}conventional.txt' file and strip its suffix
    to obtain the common base path for helper.read_values().
    """
    dirp = Path(directory)
    suffix = f"_0_{kind}conventional.txt"
    f = next(dirp.glob(f"*{suffix}"))
    stem = f.stem  # removes ".txt"
    base = stem[: -len(suffix.replace(".txt", ""))]
    return str(dirp / base)


def plot_comparison(
    path1: str,
    path2: str,
    save_name: str,
    label1: str,
    label2: str,
    title: str,
    kinds: Tuple[str, str] = ("pos", "neg"),
    x_limits: Tuple[float, float] = (-5, 140),
    y_limits: Tuple[float, float] = (0.5, 1.5),
    annot_xpos: Tuple[float, float] = (0, 80),
) -> None:
    """
    Generate side by side efficiency comparisons for 'pos' and 'neg' slices from two directories, and save as PNG + PDF.

    Parameters
    ----------
    path1, path2 : str - Directory paths containing the data files for each condition.
    save_name : str - Output basename (without extension) for the saved figures.
    label1, label2 : str - Legend labels for the two datasets.
    title : str - Supertitle for the figure.
    kinds : tuple of str - Subdirectories or filename keywords to distinguish "positive" vs "negative" plots.
    """

    # Configuration
    categories = [
        ("CV", "conventional", "black"),
        ("AI", "ai", "tab:red"),
        ("Ratio", "ratio", "tab:green"),
    ]
    styles = {
        label1: dict(marker="o", fillstyle="full", linestyle="-"),
        label2: dict(marker="s", fillstyle="none", linestyle="--"),
    }
    annot_x = {label1: annot_xpos[0], label2: annot_xpos[1]}
    label_y = 0.4
    annot_y = {"conventional": 0.30, "ai": 0.35}

    # Resolve base paths for 'pos' and 'neg'
    bases = {k: (find_base(path1, k), find_base(path2, k)) for k in kinds}

    # Create figure with two panels
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    for ax, kind in zip(axes, kinds):
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_xlabel("I [nA]")
        ax.set_ylabel("Efficiency")
        ax.set_title("Positive" if kind == "pos" else "Negative")
        # Add dataset labels in-panel
        ax.text(annot_x[label1], label_y, label1, fontsize=16)
        ax.text(annot_x[label2], label_y, label2, fontsize=16)

        # Read results for both datasets
        res1 = read_values(bases[kind][0], kind)
        res2 = read_values(bases[kind][1], kind)

        # Plot each category for both datasets
        for short, key, color in categories:
            for lbl, res in [(label1, res1), (label2, res2)]:
                style = styles[lbl]
                x_data, y_data, y_err, x_line, y_line, p0, p1, err_p0, err_p1 = res[key]
                ax.errorbar(
                    x_data, y_data, yerr=y_err, label=f"{short} {lbl}", marker=style["marker"], linestyle="None", markerfacecolor=color if style["fillstyle"] == "full" else "none", color=color
                )
                ax.plot(x_line, y_line, linestyle=style["linestyle"], color=color, label=f"{short} {lbl}")

            # Annotate fit for CV and AI only
            if key in annot_y:
                y0 = annot_y[key]
                # Dataset1 annotation
                p0_1, p1_1, err_p1_1 = res1[key][5], res1[key][6], res1[key][8]
                ax.text(annot_x[label1], y0, f"{p0_1:.5f} + {p1_1:.6f} x (±{err_p1_1:.6f})", fontsize=14, color=color)
                # Dataset2 annotation
                p0_2, p1_2, err_p1_2 = res2[key][5], res2[key][6], res2[key][8]
                ax.text(annot_x[label2], y0, f"{p0_2:.5f} + {p1_2:.6f} x (±{err_p1_2:.6f})", fontsize=14, color=color)

        ax.legend(fontsize=14, ncol=2)

    # Finalize and save
    fig.suptitle(title, fontsize=24, y=0.95)
    for ext in (".png", ".pdf"):
        fig.savefig(f"{save_name}{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
