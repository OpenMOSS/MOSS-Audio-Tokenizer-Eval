from __future__ import annotations

import argparse
import logging


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def _split_metrics(value: str | None, cfg: dict | None = None) -> list[str] | None:
    if value:
        return [m.strip() for m in value.split(",") if m.strip()]
    if cfg:
        from moss_eval.config import metric_config

        enabled = metric_config(cfg).get("enabled")
        if enabled:
            return list(enabled)
    return None


def cmd_validate(args) -> int:
    from moss_eval.config import load_config
    from moss_eval.dataset import discover_datasets, load_jsonl_dataset, validate_dataset

    if args.config:
        cfg = load_config(args.config)
        datasets = discover_datasets(cfg, check_exists=not args.no_check_audio)
    elif args.jsonl:
        datasets = [load_jsonl_dataset(args.jsonl, check_exists=not args.no_check_audio)]
    else:
        raise SystemExit("validate requires --config or --jsonl")
    failed = False
    for dataset in datasets:
        errors = validate_dataset(dataset)
        if errors:
            failed = True
            logging.error("Dataset %s has %d errors", dataset.name, len(errors))
            for err in errors[:20]:
                logging.error("  %s", err)
        else:
            logging.info("OK dataset=%s jsonl=%s items=%d", dataset.name, dataset.jsonl_path, len(dataset))
    return 1 if failed else 0


def cmd_reconstruct(args) -> int:
    from moss_eval.config import load_config
    from moss_eval.reconstruct import run_reconstruct

    cfg = load_config(args.config)
    outputs = run_reconstruct(cfg, force=args.force, device=args.device, nq_override=args.nq)
    for path in outputs:
        print(path)
    return 0


def cmd_metrics(args) -> int:
    from moss_eval.config import load_config, metric_config
    from moss_eval.metrics import DEFAULT_METRICS, evaluate_output_dir, write_results

    cfg = load_config(args.config) if args.config else {}
    mcfg = metric_config(cfg)
    metrics = _split_metrics(args.metrics, cfg) or DEFAULT_METRICS
    options = mcfg.get("options", {}) if isinstance(mcfg, dict) else {}
    results, _, errors = evaluate_output_dir(
        args.output_dir,
        metrics,
        device=args.device or cfg.get("device", "cpu"),
        metric_options=options,
    )
    result_path = write_results(args.output_dir, results, metric_errors=errors)
    logging.info("Results written to %s", result_path)
    return 0


def cmd_run(args) -> int:
    from moss_eval.config import load_config, metric_config
    from moss_eval.metrics import DEFAULT_METRICS, evaluate_output_dir, write_results
    from moss_eval.reconstruct import run_reconstruct

    cfg = load_config(args.config)
    outputs = run_reconstruct(cfg, force=args.force, device=args.device, nq_override=args.nq)
    mcfg = metric_config(cfg)
    metrics = _split_metrics(args.metrics, cfg) or DEFAULT_METRICS
    options = mcfg.get("options", {}) if isinstance(mcfg, dict) else {}
    for out_dir in outputs:
        results, _, errors = evaluate_output_dir(
            out_dir,
            metrics,
            device=args.device or cfg.get("device", "cpu"),
            metric_options=options,
        )
        result_path = write_results(out_dir, results, metric_errors=errors)
        logging.info("Results written to %s", result_path)
    return 0


def cmd_summarize(args) -> int:
    from moss_eval.summary import summarize

    metrics = _split_metrics(args.metrics) or None
    written = summarize(args.exp_dir, args.output_dir, metrics=metrics)
    for path in written:
        print(path)
    if not written:
        logging.warning("No results.json files found under %s", args.exp_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="moss-eval", description="Audio tokenizer/codec/VAE reconstruction evaluation toolkit")
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="validate config or JSONL datasets")
    p.add_argument("--config")
    p.add_argument("--jsonl")
    p.add_argument("--no-check-audio", action="store_true", help="validate JSON syntax without requiring local audio files")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("reconstruct", help="generate gt_audios and syn_audios")
    p.add_argument("--config", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--nq", default=None, help="override config nq, e.g. 1, 1,2,4, 1..8, all")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_reconstruct)

    p = sub.add_parser("metrics", help="run metrics on an existing output_dir")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--config")
    p.add_argument("--device", default="cpu")
    p.add_argument("--metrics", help="comma-separated metric list")
    p.set_defaults(func=cmd_metrics)

    p = sub.add_parser("run", help="reconstruct then evaluate metrics")
    p.add_argument("--config", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--nq", default=None)
    p.add_argument("--metrics", help="comma-separated metric list")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("summarize", help="merge results.json files into CSV files")
    p.add_argument("--exp-dir", default="exp")
    p.add_argument("--output-dir", default="csv_results")
    p.add_argument("--metrics", help="comma-separated metric list")
    p.set_defaults(func=cmd_summarize)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
