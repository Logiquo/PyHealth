import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, cast

import numpy as np
import torch

from pyhealth.datasets import MIMIC4Dataset, get_dataloader, split_by_patient
from pyhealth.models import AdaCare, GAMENet, RETAIN, RNN
from pyhealth.tasks import (
    DrugRecommendationMIMIC4,
    LengthOfStayPredictionMIMIC4,
    MortalityPredictionMIMIC4,
)
from pyhealth.trainer import Trainer


MIMIC4_ROOT = "/srv/local/data/physionet.org/files/mimiciv/2.2/"
EHR_TABLES = ["diagnoses_icd", "procedures_icd", "prescriptions"]
SPLIT_RATIOS = [0.8, 0.1, 0.1]
SEEDS = [0, 1, 2, 3, 4]
NUM_WORKERS = 16
DEFAULT_BATCH_SIZE = 128
EARLY_STOPPING_PATIENCE = 5


TASKS = {
    "mp": {
        "task_cls": MortalityPredictionMIMIC4,
        "allowed_models": {"rnn", "retain", "adacare"},
        "metrics": ["roc_auc", "pr_auc"],
        "monitor": "roc_auc",
        "metric_output_keys": {
            "roc_auc": "auroc",
            "pr_auc": "prauc",
        },
    },
    "los": {
        "task_cls": LengthOfStayPredictionMIMIC4,
        "allowed_models": {"rnn", "retain", "adacare"},
        "metrics": ["f1_macro", "f1_micro"],
        "monitor": "f1_macro",
        "metric_output_keys": {
            "f1_macro": "f1_macro",
            "f1_micro": "f1_micro",
        },
    },
    "dr": {
        "task_cls": DrugRecommendationMIMIC4,
        "allowed_models": {"rnn", "gamenet", "adacare"},
        "metrics": ["f1_micro", "pr_auc_samples", "jaccard_samples"],
        "monitor": "f1_micro",
        "metric_output_keys": {
            "pr_auc_samples": "pr_auc_samples",
            "jaccard_samples": "jaccard_samples",
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run default MIMIC-IV PyHealth baselines."
    )
    parser.add_argument("--task", required=True, choices=sorted(TASKS.keys()))
    parser.add_argument(
        "--model",
        required=True,
        choices=["rnn", "retain", "adacare", "gamenet"],
    )
    parser.add_argument(
        "--cuda",
        required=True,
        type=int,
        choices=range(8),
        metavar="{0,1,2,3,4,5,6,7}",
        help="CUDA device index used for training.",
    )
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(cuda_index: int) -> str:
    if torch.cuda.is_available():
        torch.cuda.set_device(cuda_index)
        return f"cuda:{cuda_index}"
    print("CUDA is not available; falling back to CPU.")
    return "cpu"


def load_sample_dataset(task_code: str):
    task_cfg = TASKS[task_code]
    print("Loading MIMIC-IV dataset...")
    base_dataset = MIMIC4Dataset(
        ehr_root=MIMIC4_ROOT,
        ehr_tables=EHR_TABLES,
        num_workers=NUM_WORKERS,
    )

    print(f"Applying {task_cfg['task_cls'].__name__}...")
    sample_dataset = base_dataset.set_task(
        task_cfg["task_cls"](),
        num_workers=NUM_WORKERS,
    )
    print(f"Prepared {len(sample_dataset)} samples.")
    return sample_dataset


def model_and_train_config(model_code: str) -> Dict[str, Any]:
    common = {
        "batch_size": DEFAULT_BATCH_SIZE,
        "epochs": 30,
        "lr": 1e-5,
        "weight_decay": 0.0,
        "model_kwargs": {},
    }

    if model_code == "rnn":
        return {
            **common,
            "model_cls": RNN,
            "model_kwargs": {
                "embedding_dim": 128,
                "hidden_dim": 128,
            },
        }
    if model_code == "retain":
        return {
            **common,
            "model_cls": RETAIN,
            "weight_decay": 1e-4,
            "model_kwargs": {
                "embedding_dim": 128,
                "dropout": 0.6,
            },
        }
    if model_code == "adacare":
        return {
            **common,
            "model_cls": AdaCare,
            "lr": 1e-3,
            "model_kwargs": {
                "embedding_dim": 128,
                "hidden_dim": 128,
                "kernel_size": 2,
                "r_v": 4,
                "r_c": 4,
                "dropout": 0.5,
            },
        }
    if model_code == "gamenet":
        return {
            **common,
            "model_cls": GAMENet,
            "epochs": 40,
            "lr": 2e-4,
            "model_kwargs": {
                "embedding_dim": 64,
                "hidden_dim": 64,
                "dropout": 0.4,
                "num_layers": 1,
            },
        }
    raise ValueError(f"Unsupported model: {model_code}")


def validate_pair(task_code: str, model_code: str) -> None:
    allowed_models = TASKS[task_code]["allowed_models"]
    if model_code not in allowed_models:
        allowed = ", ".join(sorted(allowed_models))
        raise ValueError(
            f"Model '{model_code}' is not valid for task '{task_code}'. "
            f"Allowed models: {allowed}."
        )


def run_seed(
    sample_dataset,
    task_code: str,
    model_code: str,
    seed: int,
    device: str,
) -> Dict[str, float]:
    task_cfg = TASKS[task_code]
    train_cfg = model_and_train_config(model_code)

    set_global_seed(seed)
    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset,
        SPLIT_RATIOS,
        seed=seed,
    )
    print(
        f"Seed {seed}: train={len(train_dataset)}, "
        f"val={len(val_dataset)}, test={len(test_dataset)}"
    )

    train_loader = get_dataloader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
    )
    val_loader = get_dataloader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
    )
    test_loader = get_dataloader(
        test_dataset,
        batch_size=train_cfg["batch_size"],
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
        output_path=str(Path.cwd() / "output" / "default" / task_code / model_code),
        exp_name=f"experiment-0-seed-{seed}",
    )
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=train_cfg["epochs"],
        monitor=task_cfg["monitor"],
        patience=EARLY_STOPPING_PATIENCE,
        optimizer_params={"lr": train_cfg["lr"]},
        weight_decay=train_cfg["weight_decay"],
    )

    results = trainer.evaluate(test_loader)
    mapped_results = {}
    for metric_name, output_key in task_cfg["metric_output_keys"].items():
        mapped_results[output_key] = float(results[metric_name])
    return mapped_results


def summarize_results(
    task_code: str,
    model_code: str,
    seed_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    run: Dict[str, Any] = {"experiment": 0}
    metric_output_keys = cast(
        Mapping[str, str],
        TASKS[task_code]["metric_output_keys"],
    )

    for metric_key in metric_output_keys.values():
        values = [float(seed_result[metric_key]) for seed_result in seed_results]
        run[metric_key] = {
            "avg": float(np.mean(values)),
            "std": float(np.std(values)),
        }
    run["seeds"] = seed_results

    return {
        "task": task_code,
        "model": model_code,
        "runs": [run],
    }


def main() -> None:
    args = parse_args()
    validate_pair(args.task, args.model)

    device = get_device(args.cuda)
    sample_dataset = load_sample_dataset(args.task)

    seed_results = []
    for seed in SEEDS:
        print("=" * 80)
        print(f"Running task={args.task}, model={args.model}, seed={seed}")
        metrics = run_seed(
            sample_dataset=sample_dataset,
            task_code=args.task,
            model_code=args.model,
            seed=seed,
            device=device,
        )
        seed_result = {"seed": seed, **metrics}
        seed_results.append(seed_result)
        print(f"Seed {seed} metrics: {metrics}")

    output = summarize_results(args.task, args.model, seed_results)
    output_path = Path.cwd() / f"default-{args.task}-{args.model}.json"
    with output_path.open("w") as f:
        json.dump(output, f, indent=4)
        f.write("\n")
    print(f"Wrote results to {output_path}")


if __name__ == "__main__":
    main()
