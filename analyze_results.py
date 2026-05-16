"""
analyze_results.py — Hệ thống phân tích toàn diện cho thực nghiệm FL + Blockchain.

Sử dụng:
    python analyze_results.py
    python analyze_results.py --log-dir results/logs --out-dir results/plots --dataset mnist
    python analyze_results.py --smooth 5 --ci --report
    python analyze_results.py --report --latex
"""
import argparse
import logging
import sys
from pathlib import Path

from analysis.loader import load_all_logs
from analysis.plots import (
    plot_baseline_comparison,
    plot_fairness_analysis,
    plot_reputation_analysis,
    plot_attack_analysis,
    plot_alpha_sensitivity,
    plot_convergence_comparison,
)
from analysis.stats import (
    compute_summary_metrics,
    compute_fairness_metrics,
    statistical_tests,
)
from analysis.report import generate_markdown_report, export_latex_tables


# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze FL + Blockchain experiment results."
    )
    p.add_argument(
        "--log-dir", default="results/logs",
        help="Directory containing CSV log files (default: results/logs)"
    )
    p.add_argument(
        "--out-dir", default="results/plots",
        help="Output directory for plots (default: results/plots)"
    )
    p.add_argument(
        "--result-dir", default="results",
        help="Output directory for CSV summaries (default: results)"
    )
    p.add_argument(
        "--dataset", default=None,
        help="Filter by dataset: mnist | fashion_mnist | cifar10 (default: all)"
    )
    p.add_argument(
        "--smooth", type=int, default=0,
        help="Moving-average window size for line plots (0 = disabled)"
    )
    p.add_argument(
        "--ci", action="store_true",
        help="Show 95%% confidence interval bands on line plots"
    )
    p.add_argument(
        "--report", action="store_true",
        help="Generate Markdown report after analysis"
    )
    p.add_argument(
        "--latex", action="store_true",
        help="Export LaTeX tables for paper inclusion"
    )
    p.add_argument(
        "--dpi", type=int, default=200,
        help="DPI for saved PNG files (default: 200)"
    )
    return p.parse_args()


def main():
    args = parse_args()

    log_dir    = Path(args.log_dir)
    out_dir    = Path(args.out_dir)
    result_dir = Path(args.result_dir)

    # Create output directories
    out_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    log.info("Loading CSV logs from: %s", log_dir)
    df = load_all_logs(log_dir)
    if df is None or df.empty:
        log.error("No data found in %s — aborting.", log_dir)
        sys.exit(1)

    if args.dataset:
        df = df[df["dataset"] == args.dataset]
        if df.empty:
            log.error("No data for dataset='%s'", args.dataset)
            sys.exit(1)
        log.info("Filtered to dataset='%s': %d rows", args.dataset, len(df))

    log.info("Total rows loaded: %d", len(df))
    log.info("Datasets   : %s", sorted(df["dataset"].unique()))
    log.info("Scenarios  : %s", sorted(df["scenario"].unique()))
    if "scenario_variant" in df.columns:
        log.info("Variants   : %s", sorted(df["scenario_variant"].unique()))
    log.info("Configs    : %s", sorted(df["config"].unique()))
    log.info("Alpha vals : %s", sorted(df["alpha"].unique()))
    if "attack_label" in df.columns:
        attack_counts = df.groupby("attack_label")["run_id"].nunique().to_dict()
        log.info("Conditions : %s", attack_counts)
    if "run_rounds_observed" in df.columns:
        run_rounds = df.groupby("run_id")["run_rounds_observed"].first()
        if run_rounds.nunique() > 1:
            log.warning("Mixed run lengths detected: %s", run_rounds.value_counts().sort_index().to_dict())

    # Plot kwargs passed to all plotting functions
    plot_cfg = dict(out_dir=out_dir, dpi=args.dpi, smooth=args.smooth, ci=args.ci)

    # 2. Baseline comparison
    log.info("--- [1/6] Baseline Comparison ---")
    plot_baseline_comparison(df, **plot_cfg)

    # 3. Fairness analysis
    log.info("--- [2/6] Fairness Analysis ---")
    plot_fairness_analysis(df, **plot_cfg)

    # 4. Reputation analysis
    log.info("--- [3/6] Reputation Analysis ---")
    plot_reputation_analysis(df, **plot_cfg)

    # 5. Attack analysis
    log.info("--- [4/6] Attack Analysis ---")
    plot_attack_analysis(df, **plot_cfg)

    # 6. Alpha sensitivity
    log.info("--- [5/6] Alpha Sensitivity ---")
    plot_alpha_sensitivity(df, **plot_cfg)

    # 7. Convergence comparison
    log.info("--- [6/6] Convergence Comparison ---")
    plot_convergence_comparison(df, **plot_cfg)

    # 8. Export summary CSVs
    log.info("Exporting summary CSVs...")
    summary = compute_summary_metrics(df)
    summary.to_csv(result_dir / "summary_metrics.csv", index=False)
    log.info("Saved: %s/summary_metrics.csv (%d rows)", result_dir, len(summary))

    fairness = compute_fairness_metrics(df)
    fairness.to_csv(result_dir / "fairness_metrics.csv", index=False)
    log.info("Saved: %s/fairness_metrics.csv (%d rows)", result_dir, len(fairness))

    # 9. Statistical significance tests
    log.info("Running statistical significance tests...")
    stat_df = df
    if "attack_label" in df.columns and (df["attack_label"] == "clean").any():
        stat_df = df[df["attack_label"] == "clean"]
    stat_results = statistical_tests(stat_df, metric_col="global_accuracy", group_col="config")
    if not stat_results.empty:
        stat_results.to_csv(result_dir / "stat_tests.csv", index=False)
        log.info("Saved: %s/stat_tests.csv (%d pairs)", result_dir, len(stat_results))
        # Log significant results
        sig = stat_results[stat_results["significant"]]
        if not sig.empty:
            log.info("Significant differences (p < 0.05):")
            for _, row in sig.iterrows():
                log.info("  %s vs %s: diff=%.4f, p=%.6f, effect=%.4f",
                         row["config_a"], row["config_b"], row["diff"], row["p_value"], row["effect_size"])
        else:
            log.info("No statistically significant differences found.")
    else:
        log.info("Insufficient runs for statistical tests.")

    # 10. LaTeX tables
    if args.latex:
        log.info("Exporting LaTeX tables...")
        latex_dir = result_dir / "latex"
        export_latex_tables(summary, fairness, latex_dir)

    # 11. Markdown report
    if args.report:
        log.info("Generating Markdown report...")
        report_path = result_dir / "analysis_report.md"
        generate_markdown_report(df, summary, fairness, report_path, out_dir,
                                 stat_tests=stat_results if not stat_results.empty else None)
        log.info("Saved: %s", report_path)

    log.info("Analysis complete. Outputs in: %s / %s", out_dir, result_dir)


if __name__ == "__main__":
    main()
