from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from finetuning_scripts.constant_utils import SupportedDevice, TaskType
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from finetuning_scripts.metric_utils.ag_metrics import Scorer
    from tabpfn.model.transformer import PerFeatureTransformer


def create_val_data(
    *,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    rng: np.random.RandomState,
    n_samples: int,
    is_classification: bool,
) -> tuple[
    pd.DataFrame | np.ndarray,
    pd.DataFrame | np.ndarray,
    pd.Series | np.ndarray,
    pd.Series | np.ndarray,
]:
    # Split data ourselves
    if n_samples < 10000:
        test_size = 0.33
    elif n_samples < 500000:
        test_size = 0.2
    elif n_samples < 1000000:
        test_size = 0.1
    else:
        test_size = 0.05

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=test_size,
        random_state=rng,
        stratify=y_train if is_classification else None,
    )
    return X_train, X_val, y_train, y_val


def validate_tabpfn(
    *,
    X_train: torch.Tensor,  # (n_samples, batch_size, n_features)
    y_train: torch.Tensor,  # (n_samples, batch_size, 1)
    X_val: torch.Tensor,  # (n_samples, batch_size, 1)
    y_val: torch.Tensor,  # (n_samples, batch_size, 1)
    validation_metric: Scorer,
    model: PerFeatureTransformer,
    model_forward_fn: Callable,
    task_type: TaskType,
    device: SupportedDevice,
    return_y_pred_proba: bool = False,
    use_native_validation: bool = False,
    checkpoint_config,
    is_data_parallel,
) -> float:
    """Validate the TabPFN model and return a loss (lower is better).

    This code assumes that batch_size for validation is 1. Otherwise,
    need to write a loop, I guess?
    """
    if use_native_validation:
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        X_val = X_val.to(device)
        y_val = y_val.to(device)

        pred_logits = model_forward_fn(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_val,
            forward_for_validation=True,
        )

        match task_type:
            case TaskType.REGRESSION:
                y_pred = pred_logits.float().flatten().cpu().detach().numpy()
                y_true = y_val.float().flatten().cpu().detach().numpy()
            case TaskType.BINARY_CLASSIFICATION:
                # TODO: check that this works / is exhaustive.
                if validation_metric.needs_threshold or validation_metric.needs_proba:
                    y_pred = (
                        torch.nn.functional.sigmoid(pred_logits[:, 0, 1])
                        .cpu()
                        .detach()
                        .numpy()
                    )
                else:
                    # Required to get the correct classes for the metrics
                    y_pred = (
                        torch.nn.functional.softmax(pred_logits[:, 0, :], dim=-1)
                        .cpu()
                        .detach()
                        .numpy()
                    )
                y_true = y_val.long().flatten().cpu().detach().numpy()
            case TaskType.MULTICLASS_CLASSIFICATION:
                y_pred = (
                    torch.nn.functional.softmax(pred_logits[:, 0, :], dim=-1)
                    .cpu()
                    .detach()
                    .numpy()
                )
                y_true = y_val.long().flatten().cpu().detach().numpy()
            case _:
                raise ValueError(f"Task type {task_type} not supported.")

        X_train.cpu()
        y_train.cpu()
        X_val.cpu()
        y_val.cpu()

    else:
        from tabpfn import TabPFNClassifier
        import os
        import tempfile

        X_train = X_train.cpu().detach().numpy()[:, 0, :]
        y_train = y_train.long().flatten().cpu().detach().numpy()
        X_val = X_val.cpu().detach().numpy()[:, 0, :]
        y_true = y_val.long().flatten().cpu().detach().numpy()

        categorical_features_indices = model_forward_fn.keywords["categorical_features_index"]
        with tempfile.TemporaryDirectory() as tmpdirname:
            save_path = tmpdirname + os.sep + "/fine_tuned_model_val.ckpt"
            torch.save(
                dict(
                    state_dict=model.module.state_dict() if is_data_parallel else model.state_dict(),
                    config=checkpoint_config),
                str(save_path),
            )
            clf = TabPFNClassifier(model_path=save_path,n_estimators=1, categorical_features_indices=categorical_features_indices, device="cuda")
            y_pred = clf.fit(X_train, y_train).predict_proba(X_val)[:, 1]

    score = validation_metric(y_true=y_true, y_pred=y_pred)

    if return_y_pred_proba:
        return validation_metric.convert_score_to_error(score=score), y_pred

    return validation_metric.convert_score_to_error(score=score)