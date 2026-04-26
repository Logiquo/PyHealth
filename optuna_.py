import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, cast

import numpy as np

from default import (
    EARLY_STOPPING_PATIENCE,
    SEEDS,
    SPLIT_RATIOS,
    TASKS,
    get_device,
    load_sample_dataset,
    set_global_seed,
    validate_pair,
)
from pyhealth.datasets import get_dataloader, split_by_patient
from pyhealth.models import AdaCare, GAMENet, RETAIN, RNN
from pyhealth.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Optuna-tuned MIMIC-IV PyHealth baselines."
    )
    parser.add_argument("--task", required=True, choices=sorted(TASKS.keys()))
    parser.add_argument(
        "--model",
        required=True,
        choices=["rnn", "retain", "adacare", "gamenet"],
    )
    parser.add_argument(
        "--exp",
        required=True,
        type=int,
        help="Number of Optuna experiments/trials to run.",
    )
    return parser.parse_args()


def import_optuna() -> Any:
    try:
        return importlib.import_module("optuna")
    except ModuleNotFoundError:
        print(
            "Optuna is not installed in this environment. Install it before "
            "running optuna_.py, for example: pip install optuna"
        )
        sys.exit(1)


def suggest_train_config(trial: Any, model_code: str) -> Dict[str, Any]:
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    if model_code == "gamenet":
        return {
            "batch_size": batch_size,
            "epochs": trial.suggest_categorical("epochs", [30, 40, 50]),
            "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
            "weight_decay": trial.suggest_float(
                "weight_decay", 1e-7, 1e-3, log=True
            ),
            "model_cls": GAMENet,
            "model_kwargs": {
                "embedding_dim": trial.suggest_categorical(
                    "embedding_dim", [32, 64, 128]
                ),
                "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128]),
                "num_layers": trial.suggest_int("num_layers", 1, 2),
                "dropout": trial.suggest_float("dropout", 0.0, 0.7),
            },
        }

    common = {
        "batch_size": batch_size,
        "epochs": trial.suggest_categorical("epochs", [20, 30, 40]),
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True),
        "model_kwargs": {},
    }

    if model_code == "rnn":
        return {
            **common,
            "model_cls": RNN,
            "model_kwargs": {
                "embedding_dim": trial.suggest_categorical(
                    "embedding_dim", [64, 128, 256]
                ),
                "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
                "rnn_type": trial.suggest_categorical(
                    "rnn_type", ["RNN", "LSTM", "GRU"]
                ),
                "num_layers": trial.suggest_int("num_layers", 1, 3),
                "dropout": trial.suggest_float("dropout", 0.0, 0.7),
                "bidirectional": trial.suggest_categorical(
                    "bidirectional", [False, True]
                ),
            },
        }

    if model_code == "retain":
        return {
            **common,
            "model_cls": RETAIN,
            "model_kwargs": {
                "embedding_dim": trial.suggest_categorical(
                    "embedding_dim", [64, 128, 256]
                ),
                "dropout": trial.suggest_float("dropout", 0.0, 0.8),
            },
        }

    if model_code == "adacare":
        return {
            **common,
            "model_cls": AdaCare,
            "model_kwargs": {
                "embedding_dim": trial.suggest_categorical(
                    "embedding_dim", [64, 128, 256]
                ),
                "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
                "kernel_size": trial.suggest_int("kernel_size", 2, 5),
                "kernel_num": trial.suggest_categorical("kernel_num", [32, 64, 128]),
                "r_v": trial.suggest_categorical("r_v", [2, 4, 8, 16]),
                "r_c": trial.suggest_categorical("r_c", [2, 4, 8, 16]),
                "activation": trial.suggest_categorical(
                    "activation", ["sigmoid", "softmax", "sparsemax"]
                ),
                "rnn_type": trial.suggest_categorical("rnn_type", ["gru", "lstm"]),
                "dropout": trial.suggest_float("dropout", 0.0, 0.7),
            },
        }

    raise ValueError(f"Unsupported model: {model_code}")


def run_seed(
    sample_dataset: Any,
    task_code: str,
    model_code: str,
    seed: int,
    device: str,
    train_cfg: Mapping[str, Any],
    experiment: int,
) -> Dict[str, float]:
    task_cfg = TASKS[task_code]

    set_global_seed(seed)
    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset,
        SPLIT_RATIOS,
        seed=seed,
    )
    print(
        f"Experiment {experiment}, seed {seed}: "
        f"train={len(train_dataset)}, val={len(val_dataset)}, "
        f"test={len(test_dataset)}"
    )

    train_loader = get_dataloader(
        train_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
    )
    val_loader = get_dataloader(
        val_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
    )
    test_loader = get_dataloader(
        test_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
    )

    model = train_cfg["model_cls"](
        dataset=sample_dataset,
        **train_cfg["model_kwargs"],
    )
    trainer = Trainer(
        model=model,
        metrics=task_cfg["metrics"],
        device=device,
        output_path=str(Path.cwd() / "output" / "optuna" / task_code / model_code),
        exp_name=f"experiment-{experiment}-seed-{seed}",
    )
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=int(train_cfg["epochs"]),
        monitor=task_cfg["monitor"],
        patience=EARLY_STOPPING_PATIENCE,
        optimizer_params={"lr": float(train_cfg["lr"])},
        weight_decay=float(train_cfg["weight_decay"]),
    )

    results = trainer.evaluate(test_loader)
    metric_output_keys = cast(
        Mapping[str, str],
        task_cfg["metric_output_keys"],
    )
    return {
        output_key: float(results[metric_name])
        for metric_name, output_key in metric_output_keys.items()
    }


def build_run(
    task_code: str,
    experiment: int,
    seed_results: List[Dict[str, Any]],
    params: Mapping[str, Any],
) -> Dict[str, Any]:
    task_cfg = TASKS[task_code]
    metric_output_keys = cast(
        Mapping[str, str],
        task_cfg["metric_output_keys"],
    )
    run: Dict[str, Any] = {"experiment": experiment}

    for metric_key in metric_output_keys.values():
        values = [float(seed_result[metric_key]) for seed_result in seed_results]
        run[metric_key] = {
            "avg": float(np.mean(values)),
            "std": float(np.std(values)),
        }
    run["seeds"] = seed_results
    run["params"] = dict(params)
    return run


def write_output(task_code: str, model_code: str, runs: List[Dict[str, Any]]) -> None:
    output = {
        "task": task_code,
        "model": model_code,
        "runs": runs,
    }
    output_path = Path.cwd() / f"optuna-{task_code}-{model_code}.json"
    with output_path.open("w") as f:
        json.dump(output, f, indent=4)
        f.write("\n")
    print(f"Wrote results to {output_path}")


def main() -> None:
    args = parse_args()
    if args.exp <= 0:
        raise ValueError("--exp must be a positive integer")
    validate_pair(args.task, args.model)

    optuna = import_optuna()
    device = get_device()
    sample_dataset = load_sample_dataset(args.task)
    runs: List[Dict[str, Any]] = []
    storage_path = Path.cwd() / "output" / "optuna" / "optuna.db"
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{storage_path}"
    study_name = f"{args.task}-{args.model}"

    def objective(trial: Any) -> float:
        train_cfg = suggest_train_config(trial, args.model)
        seed_results = []

        print("=" * 80)
        print(f"Starting experiment {trial.number}: {trial.params}")
        for seed in SEEDS:
            metrics = run_seed(
                sample_dataset=sample_dataset,
                task_code=args.task,
                model_code=args.model,
                seed=seed,
                device=device,
                train_cfg=train_cfg,
                experiment=trial.number,
            )
            seed_result = {"seed": seed, **metrics}
            seed_results.append(seed_result)
            print(f"Experiment {trial.number}, seed {seed} metrics: {metrics}")

        run = build_run(args.task, trial.number, seed_results, trial.params)
        runs.append(run)
        write_output(args.task, args.model, runs)

        monitor_output_key = cast(
            Mapping[str, str],
            TASKS[args.task]["metric_output_keys"],
        )[TASKS[args.task]["monitor"]]
        return float(run[monitor_output_key]["avg"])

    study = optuna.create_study(
        direction="maximize",
        storage=storage_url,
        study_name=study_name,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.exp)

    print(f"Optuna study database: {storage_path}")
    print(f"Optuna study name: {study_name}")
    print(f"Best experiment: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")


if __name__ == "__main__":
    main()
