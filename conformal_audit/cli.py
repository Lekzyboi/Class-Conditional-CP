"""Command-line entry points for the conformal audit framework."""

from __future__ import annotations

import argparse

from conformal_audit.audit.duplicates import run_duplicate_audit_from_config
from conformal_audit.audit.overlap import run_overlap_audit_from_config, run_overlap_filter_from_config
from conformal_audit.audit.splits import run_split_audit_from_config
from conformal_audit.config import load_experiment_config
from conformal_audit.evaluation import run_experiment
from conformal_audit.reporting.figures import generate_figures_from_results
from conformal_audit.reporting.tables import generate_tables_from_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="conformal-audit",
        description="Audit class-conditional conformal coverage in biomedical classifiers.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run an experiment from a config file.")
    run_parser.add_argument("--config", required=True, help="Path to a YAML/JSON config file.")

    audit_parser = subparsers.add_parser("audit", help="Run dataset or split audits.")
    audit_parser.add_argument("kind", choices=["overlap", "overlap-filter", "duplicates", "splits", "distribution-shift"])
    audit_parser.add_argument("--config", required=True, help="Path to an audit config file.")

    report_parser = subparsers.add_parser("report", help="Generate tables or figures.")
    report_parser.add_argument("kind", choices=["tables", "figures", "confusion", "manuscript-checks"])
    report_parser.add_argument("--results", required=True, help="Path to a result JSON file.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        config = load_experiment_config(args.config)
        result = run_experiment(config)
        print(f"Finished {result['name']} -> {result['output_dir']}")
        return 0
    if args.command == "audit":
        config = load_experiment_config(args.config)
        if args.kind == "overlap":
            result = run_overlap_audit_from_config(config)
        elif args.kind == "overlap-filter":
            result = run_overlap_filter_from_config(config)
        elif args.kind == "duplicates":
            result = run_duplicate_audit_from_config(config)
        elif args.kind == "splits":
            result = run_split_audit_from_config(config)
        else:
            parser.error(f"Audit kind '{args.kind}' is scaffolded but not implemented yet.")
        print(f"Finished {args.kind} audit {result.get('name', config.name)} -> {result['output_dir']}")
        return 0
    if args.command == "report":
        if args.kind == "tables":
            manifest = generate_tables_from_results(args.results)
            print(f"Finished table report -> {manifest['output_dir']}")
            return 0
        if args.kind == "figures":
            manifest = generate_figures_from_results(args.results)
            print(f"Finished figure report -> {manifest['output_dir']}")
            return 0
        parser.error(f"Report kind '{args.kind}' is scaffolded but not implemented yet.")
    parser.error(f"Command '{args.command}' is scaffolded but not implemented yet.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
