#!/usr/bin/env python
"""Stage 7: turn result CSVs into LaTeX table fragments and number macros for the report.

    python scripts/07_report_assets.py

Writes report/generated/*.tex. The report never hard-codes a number: every figure
in the text comes from these macros, so re-running the pipeline updates the PDF.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from milan_forecast.config import PROJECT_ROOT, load_config  # noqa: E402

OUT = PROJECT_ROOT / "report" / "generated"
MODEL_NAMES = {"baselines": "Baselines", "naive": "Naive", "seasonal_naive": "Seasonal naive (24 h)",
               "weekly_naive": "Seasonal naive (168 h)", "sarima": "SARIMAX", "lightgbm": "LightGBM",
               "lstm": "LSTM", "gru": "GRU"}


HORIZON_WORDS = {"1": "One", "6": "Six", "24": "Day"}


def macro_name(s: str) -> str:
    # TeX macro names can't contain digits
    parts = re.sub(r"[^a-zA-Z0-9]+", " ", s).split()
    return "".join(HORIZON_WORDS.get(p, p).capitalize() for p in parts)


def write(name: str, text: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(text)


def fmt(x, nd=2) -> str:
    if pd.isna(x):
        return "--"
    return f"{x:,.{nd}f}" if abs(x) >= 1000 else f"{x:.{nd}f}"


def results_table(tables: Path, macros: list[str]) -> None:
    path = tables / "results_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df["horizon"] = df["horizon"].astype(str)
    rows = []
    for model in [m for m in MODEL_NAMES if m in set(df["model"])]:
        sub = df[df["model"] == model].set_index("horizon")
        cells = []
        for h in ["1", "6", "24"]:
            if h in sub.index:
                cells.append(f"{fmt(sub.loc[h, 'mase'], 3)} & {fmt(sub.loc[h, 'smape'], 1)}")
                macros.append(f"\\newcommand{{\\mase{macro_name(model)}H{HORIZON_WORDS[h]}}}{{{fmt(sub.loc[h, 'mase'], 3)}}}")
                macros.append(f"\\newcommand{{\\smape{macro_name(model)}H{HORIZON_WORDS[h]}}}{{{fmt(sub.loc[h, 'smape'], 1)}}}")
            else:
                cells.append("-- & --")
        allrow = sub.loc["all"] if "all" in sub.index else None
        overall = f"{fmt(allrow['mase'], 3)} & {fmt(allrow['smape'], 1)} & {fmt(allrow['mae'], 1)}" if allrow is not None else "-- & -- & --"
        if allrow is not None:
            macros.append(f"\\newcommand{{\\mase{macro_name(model)}All}}{{{fmt(allrow['mase'], 3)}}}")
            macros.append(f"\\newcommand{{\\smape{macro_name(model)}All}}{{{fmt(allrow['smape'], 1)}}}")
        rows.append(f"{MODEL_NAMES[model]} & " + " & ".join(cells) + f" & {overall} \\\\")
    body = "\n".join(rows)
    write("results_table.tex", r"""\begin{tabular}{l rr rr rr rrr}
\toprule
 & \multicolumn{2}{c}{$h=1$ h} & \multicolumn{2}{c}{$h=6$ h} & \multicolumn{2}{c}{$h=24$ h} & \multicolumn{3}{c}{All horizons} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-10}
Model & MASE & sMAPE & MASE & sMAPE & MASE & sMAPE & MASE & sMAPE & MAE \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
""")
    # best learned model overall
    learned = df[(df["horizon"] == "all") & df["model"].isin(["sarima", "lightgbm", "lstm", "gru"])]
    if len(learned):
        best = learned.sort_values("mase").iloc[0]
        macros.append(f"\\newcommand{{\\bestModel}}{{{MODEL_NAMES[best['model']]}}}")
        macros.append(f"\\newcommand{{\\bestModelMase}}{{{fmt(best['mase'], 3)}}}")
    overall = df[df["horizon"] == "all"]
    if len(overall):
        best = overall.sort_values("mase").iloc[0]
        macros.append(f"\\newcommand{{\\bestOverallModel}}{{{MODEL_NAMES[best['model']]}}}")
        macros.append(f"\\newcommand{{\\bestOverallMase}}{{{fmt(best['mase'], 3)}}}")


def mase_by_cell_table(tables: Path) -> None:
    path = tables / "results_mase_pivot.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    models = [m for m in MODEL_NAMES if m in df.columns]
    short = {"naive": "Naive", "seasonal_naive": "SN-24", "weekly_naive": "SN-168", "sarima": "SARIMAX",
             "lightgbm": "LightGBM", "lstm": "LSTM", "gru": "GRU"}
    header = " & ".join(short[m] for m in models)
    rows = []
    for (series, h), g in df.groupby(["series", "horizon"]):
        vals = g.iloc[0][models].astype(float)
        best = vals.idxmin()
        cells = [f"\\textbf{{{fmt(v, 3)}}}" if m == best else fmt(v, 3) for m, v in vals.items()]
        rows.append(f"{series.replace('_', ' ')} & {h} & " + " & ".join(cells) + r" \\")
    write("mase_by_cell_table.tex", "\\begin{tabular}{l r " + "r" * len(models) + "}\n\\toprule\nCell & $h$ & " + header
          + " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def dm_table(tables: Path) -> None:
    path = tables / "dm_vs_seasonal_naive.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df[df["model"].isin(["sarima", "lightgbm", "lstm", "gru"])]
    summary = (df.assign(sig_better=(df["p_value"] < 0.05) & (df["better"] == "model"),
                         sig_worse=(df["p_value"] < 0.05) & (df["better"] == "reference"))
               .groupby("model")[["sig_better", "sig_worse"]].sum().assign(total=df.groupby("model").size()))
    rows = [f"{MODEL_NAMES[m]} & {int(r.sig_better)} & {int(r.sig_worse)} & {int(r.total)} \\\\" for m, r in summary.iterrows()]
    write("dm_table.tex", "\\begin{tabular}{l rrr}\n\\toprule\nModel & Sig. better & Sig. worse & Comparisons \\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def cost_table(tables: Path) -> None:
    path = tables / "compute_cost.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df[df["part"] == "test"]
    rows = [f"{MODEL_NAMES.get(r.model, r.model)} & {fmt(r.seconds, 1)} & {fmt(r.peak_rss_mb, 0)} \\\\" for r in df.itertuples()]
    write("cost_table.tex", "\\begin{tabular}{l rr}\n\\toprule\nModel & Fit + forecast time (s) & Peak RSS (MB) \\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def memory_table(tables: Path, macros: list[str]) -> None:
    mem_path, naive_path = tables / "memory_log.csv", tables / "naive_cost_estimate.csv"
    if not mem_path.exists():
        return
    mem = pd.read_csv(mem_path)
    ingest = mem[mem["stage"].str.startswith("ingest:")]
    rows = []
    if naive_path.exists():
        naive = pd.read_csv(naive_path).iloc[0]
        rows.append(f"Naive pandas load, one day (est.) & {fmt(naive['naive_pandas_mb'], 0)} & -- \\\\")
        rows.append(f"Naive pandas load, all {int(naive['n_files'])} days (est.) & {fmt(naive['naive_all_days_gb'] * 1024, 0)} & -- \\\\")
        macros.append(f"\\newcommand{{\\naiveDayMB}}{{{fmt(naive['naive_pandas_mb'], 0)}}}")
        macros.append(f"\\newcommand{{\\naiveAllGB}}{{{fmt(naive['naive_all_days_gb'], 1)}}}")
        macros.append(f"\\newcommand{{\\rawDayMB}}{{{fmt(naive['raw_size_mb'], 0)}}}")
        macros.append(f"\\newcommand{{\\rawRowsM}}{{{fmt(naive['est_rows'] / 1e6, 1)}}}")
    if len(ingest):
        rows.append(f"Streaming aggregation, per day (median) & {fmt(ingest['rss_peak_mb'].median(), 0)} & {fmt(ingest['seconds'].median(), 1)} \\\\")
        rows.append(f"Streaming aggregation, all days (max, total) & {fmt(ingest['rss_peak_mb'].max(), 0)} & {fmt(ingest['seconds'].sum(), 0)} \\\\")
        macros.append(f"\\newcommand{{\\ingestPeakMB}}{{{fmt(ingest['rss_peak_mb'].max(), 0)}}}")
        macros.append(f"\\newcommand{{\\ingestTotalMin}}{{{fmt(ingest['seconds'].sum() / 60, 1)}}}")
        macros.append(f"\\newcommand{{\\ingestDays}}{{{len(ingest)}}}")
    for stage in ["wide_hourly:internet", "citywide_10min", "eda", "tsa"]:
        s = mem[mem["stage"] == stage]
        if len(s):
            rows.append(f"{stage.replace('_', ' ').replace(':', ' ')} & {fmt(s['rss_peak_mb'].max(), 0)} & {fmt(s['seconds'].max(), 1)} \\\\")
    if len(mem):
        macros.append(f"\\newcommand{{\\pipelinePeakMB}}{{{fmt(mem['rss_peak_mb'].max(), 0)}}}")
    write("memory_table.tex", "\\begin{tabular}{p{4.6cm} rr}\n\\toprule\nStage & RSS (MB) & Time (s) \\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def stationarity_table(tables: Path) -> None:
    path = tables / "tsa_stationarity.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df[df["series"] == "citywide"]
    rows = [f"{r.transform} & {fmt(r.adf_stat, 2)} & {fmt(r.adf_p, 3)} & {fmt(r.kpss_stat, 2)} & {fmt(r.kpss_p, 3)} & {r.verdict} \\\\"
            for r in df.itertuples()]
    write("stationarity_table.tex", "\\begin{tabular}{l rr rr l}\n\\toprule\nTransform & ADF & $p$ & KPSS & $p$ & Verdict \\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def decomposition_macros(tables: Path, macros: list[str]) -> None:
    path = tables / "tsa_decomposition.csv"
    if not path.exists():
        return
    df = pd.read_csv(path).set_index("series")
    if "citywide" in df.index:
        r = df.loc["citywide"]
        for col, name in [("stl24_seasonal_share", "stlDailyShare"), ("stl24_resid_share", "stlResidShare"),
                          ("mstl_daily_share", "mstlDailyShare"), ("mstl_weekly_share", "mstlWeeklyShare")]:
            if col in r and pd.notna(r[col]):
                macros.append(f"\\newcommand{{\\{name}}}{{{100 * r[col]:.0f}}}")


def eda_macros(tables: Path, macros: list[str]) -> None:
    conc = tables / "spatial_concentration_internet.csv"
    if conc.exists():
        c = pd.read_csv(conc).set_index("metric")["value"]
        macros.append(f"\\newcommand{{\\giniInternet}}{{{c['gini']:.2f}}}")
        macros.append(f"\\newcommand{{\\topTenShare}}{{{100 * c['top_10pct_share']:.0f}}}")
        macros.append(f"\\newcommand{{\\topOneShare}}{{{100 * c['top_1pct_share']:.0f}}}")
    stats = tables / "summary_statistics.csv"
    if stats.exists():
        s = pd.read_csv(stats, index_col=0)
        macros.append(f"\\newcommand{{\\internetShare}}{{{s.loc['internet', 'share_pct']:.0f}}}")
        macros.append(f"\\newcommand{{\\nHours}}{{{int(s.loc['n_hours', 'total'])}}}")
    sel = tables / "selected_cells.json"
    if sel.exists():
        cells = json.load(open(sel))["cells"]
        items = ", ".join(f"{k.replace('_', ' ')} (\\#{v})" for k, v in cells.items())
        macros.append(f"\\newcommand{{\\selectedCells}}{{{items}}}")
        macros.append(f"\\newcommand{{\\nSelected}}{{{len(cells)}}}")
    sil = tables / "cluster_silhouette.csv"
    if sil.exists():
        s = pd.read_csv(sil).set_index("k")["silhouette"]
        macros.append(f"\\newcommand{{\\bestK}}{{{int(s.idxmax())}}}")
        macros.append(f"\\newcommand{{\\bestSilhouette}}{{{s.max():.2f}}}")
        clusters = tables / "cell_clusters.csv"
        if clusters.exists():
            chosen_k = pd.read_csv(clusters)["cluster"].nunique()
            macros.append(f"\\newcommand{{\\chosenK}}{{{chosen_k}}}")
            if chosen_k in s.index:
                macros.append(f"\\newcommand{{\\chosenSilhouette}}{{{s.loc[chosen_k]:.2f}}}")


def experiment_log_table(cfg, macros: list[str]) -> None:
    """Condensed tuning trail: manual rounds only (Optuna trials summarised in one line)."""
    runs_dir = cfg.paths.experiments_dir / "runs"
    if not runs_dir.exists():
        return
    rows = []
    for model in ["sarima", "lightgbm", "lstm"]:
        recs = []
        for p in sorted((runs_dir / model).glob("*_val.json")) if (runs_dir / model).exists() else []:
            recs.append(json.loads(p.read_text()))
        manual = [r for r in recs if not r["run_id"].startswith("opt")]
        optuna = [r for r in recs if r["run_id"].startswith("opt")]
        for r in manual:
            note = r["note"].replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
            if len(note) > 110:
                note = note[:107] + "..."
            family_name = "RNN" if model == "lstm" else MODEL_NAMES[model]
            rows.append(f"{family_name} & {r['run_id'].removesuffix('_val')} & {fmt(r['metrics']['mase'], 3)} & "
                        f"{fmt(r['metrics']['smape'], 1)} & {fmt(r['train_seconds'], 0)} & {note} \\\\")
        if optuna:
            best = min(optuna, key=lambda r: r["metrics"]["mase"])
            family_name = "RNN" if model == "lstm" else MODEL_NAMES[model]
            rows.append(f"{family_name} & Optuna ({len(optuna)} trials) & {fmt(best['metrics']['mase'], 3)} & "
                        f"{fmt(best['metrics']['smape'], 1)} & {fmt(sum(r['train_seconds'] for r in optuna), 0)} & "
                        f"best trial {best['run_id'].removesuffix('_val')}: " +
                        ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in best["params"].items()
                                  if k in ("num_leaves", "learning_rate", "min_child_samples", "feature_fraction", "lambda_l2")).replace("_", "\\_") + r" \\")
            macros.append(f"\\newcommand{{\\optunaTrials}}{{{len(optuna)}}}")
            macros.append(f"\\newcommand{{\\optunaBestMase}}{{{fmt(best['metrics']['mase'], 3)}}}")
    write("experiment_log_table.tex", "\\begin{tabular}{l l rr r p{8.2cm}}\n\\toprule\nModel & Run & val MASE & val sMAPE & s & Rationale for the run \\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    tables = cfg.paths.tables_dir
    macros: list[str] = []
    results_table(tables, macros)
    mase_by_cell_table(tables)
    dm_table(tables)
    cost_table(tables)
    memory_table(tables, macros)
    stationarity_table(tables)
    decomposition_macros(tables, macros)
    eda_macros(tables, macros)
    experiment_log_table(cfg, macros)
    # relative path from report/ to the figures directory, so \includegraphics works from any config
    fig_rel = Path("..") / cfg.paths.figures_dir.relative_to(PROJECT_ROOT)
    macros.append(f"\\newcommand{{\\figdir}}{{{fig_rel.as_posix()}}}")
    write("numbers.tex", "% generated by scripts/07_report_assets.py - do not edit\n" + "\n".join(macros) + "\n")
    logging.info("wrote %d macros and tables to %s", len(macros), OUT)


if __name__ == "__main__":
    main()
