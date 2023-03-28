import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score, recall_score, precision_score


def train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_functions: list,
    epochs: int,
    best_model_path: str
) -> tuple[list[float], list[float]]:
    """
    Implements training loop for the model
    """
    metrics = defaultdict(list)
    model.train()
    best_score = 999999
    for epoch in tqdm(range(epochs)):
        sum_loss = 0
        for idx, (x_batch, y_batch) in enumerate(train_dataloader):
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = 0
            for loss_func in loss_functions:
                loss += loss_func(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()

        evaluation_results = evaluate_on_val(
            model=model,
            validation_loader=validation_dataloader,
            loss_functions=loss_functions
        )
        train_results = evaluate_on_train(
            model=model,
            train_loader=train_dataloader,
            loss_functions=loss_functions
        )
        evaluation_results.update(train_results)
        metrics['epochs'].append(epoch)
        metrics['train_loss'].append(sum_loss/(idx+1))
        for metric, value in evaluation_results.items():
            metrics[metric].append(value)

        print(
            f'Train loss: {metrics["train_loss"][-1]}',
            f'-- Validation loss: {metrics["val_loss"][-1]}'
        )
        if metrics["val_loss"][-1] < best_score:
            best_score = metrics["val_loss"][-1]
            torch.save(model.state_dict(), best_model_path)
    return metrics


@torch.inference_mode()
def evaluate_on_val(
    model: nn.Module,
    validation_loader: DataLoader,
    loss_functions
) -> dict[str, float]:
    """
    Evaluates model on evaluation dataloader and returns dict of metrics
    """
    total_loss = 0
    model.eval()
    predictions = []
    labels = []
    for idx, batch in enumerate(validation_loader):
        x_batch, y_batch = batch
        y_pred = model(x_batch)
        loss = 0
        for loss_func in loss_functions:
            loss += loss_func(y_pred, y_batch)
        total_loss += loss.item()
        predictions += torch.argmax(y_pred, dim=-1).tolist()
        labels += y_batch[:, 1].tolist()
    model.train()
    total_loss = total_loss / (idx + 1)
    return {
        'val_loss': total_loss,
        'val_accuracy': accuracy_score(y_true=predictions, y_pred=labels),
        'val_recall': recall_score(y_true=predictions, y_pred=labels, zero_division=0),
        'val_precision': precision_score(y_true=predictions, y_pred=labels, zero_division=0)
    }



@torch.inference_mode()
def evaluate_on_train(
    model: nn.Module,
    train_loader: DataLoader,
    loss_functions
) -> dict[str, float]:
    """
    Evaluates model on evaluation dataloader and returns dict of metrics
    """
    total_loss = 0
    model.eval()
    predictions = []
    labels = []
    for idx, batch in enumerate(train_loader):
        x_batch, y_batch = batch
        y_pred = model(x_batch)
        loss = 0
        for loss_func in loss_functions:
            loss += loss_func(y_pred, y_batch)
        total_loss += loss.item()
        predictions += torch.argmax(y_pred, dim=-1).tolist()
        labels += y_batch[:, 1].tolist()
    model.train()
    total_loss = total_loss / (idx + 1)
    return {
        'train_accuracy': accuracy_score(y_true=predictions, y_pred=labels),
        'train_recall': recall_score(y_true=predictions, y_pred=labels, zero_division=0),
        'train_precision': precision_score(y_true=predictions, y_pred=labels, zero_division=0)
    }


