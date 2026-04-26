# Setup

The MIMIC-4 data is located at /srv/local/data/physionet.org/files/mimiciv/2.2/

The cache path should be default.

The `num_worker` in dataset should be 16.

There are 3 PyHealth tasks in focus
- mp: MortalityPredictionMIMIC4 in pyhealth/tasks/mortality_prediction.py
- los: LengthOfStayPredictionMIMIC4 in pyhealth/tasks/length_of_stay_prediction.py
- dr: DrugRecommendationMIMIC4

For each PyHealth task, there are 3 models in focus. For mp,
- rnn: RNN in pyhealth/models/rnn.py
- retain: RETAIN in pyhealth/models/retain.py
- adacare: AdaCare in pyhealth/models/adacare.py
and should report AUROC and PRAUC

For los:
- rnn: RNN in pyhealth/models/rnn.py
- retain: RETAIN in pyhealth/models/retain.py
- adacare: AdaCare in pyhealth/models/adacare.py
and should report F1 Macro and F1 Micro

For dr:
- rnn: RNN in pyhealth/models/rnn.py
- gamenet: GAMENet in pyhealth/models/gamenet.py
- adacare: AdaCare in pyhealth/models/adacare.py
and should report PRAUC samples and Jaccard Samples

The five random seeds for the train/test split will be [0, 1, 2, 3, 4]

Use an 80/10/10 patient-level train/validation/test split with
`split_by_patient(sample_dataset, [0.8, 0.1, 0.1], seed=<seed>)`.

Load MIMIC-IV through `MIMIC4Dataset` with the EHR tables needed by all three
tasks: `diagnoses_icd`, `procedures_icd`, and `prescriptions`. The dataset
loader adds the basic patient/admission tables automatically. Apply the task
with `num_workers=16`.

Use early stopping patience 5 when early stopping is enabled. Monitor
`roc_auc` for mortality prediction, `f1_macro` for length-of-stay prediction,
and `f1_micro` for drug recommendation.

# Default Baseline

The input args are
- `--task`: mp/los/dr
- `--model`: rnn/retain/adacare/gamenet

The script should autodetect the CUDA device with the most free memory. If CUDA
is unavailable or all visible CUDA devices are occupied, the script should print
a message and exit without training.

Write a script that train the models with 5 different seeds, save the result in `default-<task>-<model>.json` file in the workspace root directory. In the following format:
```
{
    "task": "<the code of the task, e.g. mp>",
    "model": "<the code of the model, e.g. rnn>",
    "runs": [
        {
            "experiment": 0, // default only have experiment 0
            "auroc": { "avg": <the mean of all the seed runs>, "std": <the standard deviation of all the seed runs> },
            "prauc": { ... same as auroc },
            "seeds": [
                {
                    "seed": <the seed of the run, e.g. 0>,
                    "auroc": <the auroc of this run>,
                    "prauc": <the prauc of this run>
                }
            ]
        }
    ]
}
```

Use these output metric keys:
- mp: `auroc`, `prauc`
- los: `f1_macro`, `f1_micro`
- dr: `pr_auc_samples`, `jaccard_samples`

The hyperparamter specification of task-model pairs are
- mp-rnn: Lr 1e-5, 30 epochs, early stopping true
- mp-retain: Dropout vi = 0.6, Dropout ci = 0.6, L2 regularization loss = 0.0001, hidden size =128, embedding size = 128, layers=1, Batch_size = 128, Epochs = 30
- mp-adacare: Hidden size = 128, Kernel size = 2, Dilation rates [1,3,5], Compression ratio = 4, Learning rate = 1e-3, Adam optimizer, Batch-size 128, Dropout rate = 0.5, Epochs = 30

- los-rnn: Lr 1e-5, 30 epochs, early stopping true
- los-retain: Dropout vi = 0.6, Dropout ci = 0.6, L2 regularization loss = 0.0001, hidden size =128, embedding size = 128, layers=1, Batch_size = 128, Epochs = 30
- los-adacare: Hidden size = 128, Kernel size = 2, Dilation rates [1,3,5], Compression ratio = 4, Learning rate = 1e-3, Adam optimizer, Batch-size 128, Dropout rate = 0.5, Epochs = 30

- dr-gamenet: Epochs = 40, learning rate = 0.0002, embedding size = 64, hidden size = 64, dropout probability = 0.4, GCN layers 2, 1 layer of GRU
- dr-rnn: Lr 1e-5, 30 epochs, early stopping true
- dr-adacare: Hidden size = 128, Kernel size = 2, Dilation rates [1,3,5], Compression ratio = 4, Learning rate = 1e-3, Adam optimizer, Batch-size 128, Dropout rate = 0.5, Epochs = 30

# Optuna Baseline
TODO
