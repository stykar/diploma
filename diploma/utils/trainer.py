import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_functions : list,
    epochs: int,
    evaluation_steps: int,
    best_model_path: str
) -> tuple[list[float], list[float]]:
    """
    Implements training loop for the model
    """
    train_losses = []
    validation_losses = []
    validation_accs = []
    model.train()
    global_step = 0
    best_score = 999999
    for _ in tqdm(range(epochs)):
        training_step = 0
        sum_loss = 0
        for x_batch, y_batch in train_dataloader:    
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = 0
            for loss_func in loss_functions:
                loss += loss_func(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            training_step += 1
            global_step += 1
            if evaluation_steps > 0 and training_step % evaluation_steps == 0:
                validation_loss, validation_acc = evaluate_model(
                    model=model,
                    validation_loader=validation_dataloader,
                    loss_functions=loss_functions
                )
                train_losses.append(sum_loss/training_step)
                validation_losses.append(validation_loss)
                validation_accs.append(validation_acc)
                print(f'Train loss: {train_losses[-1]} -- Validation loss: {validation_losses[-1]} -- Validation accuracy: {validation_accs[-1]}')
                if validation_loss < best_score:
                    best_score = validation_loss
                    torch.save(model.state_dict(), best_model_path)
    
    return train_losses, validation_losses, validation_accs

@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    validation_loader: DataLoader,
    loss_functions
) -> tuple[float, float]:
    """
    Evaluates model on evaluation dataloader and returns (loss, eval accuracy)
    """
    total_loss = 0 
    model.eval()
    preds = []
    for idx, batch in enumerate(validation_loader):
        x_batch, y_batch = batch
        y_pred = model(x_batch)
        loss = 0
        for loss_func in loss_functions:
            loss += loss_func(y_pred, y_batch)
            preds += torch.abs(torch.argmax(y_pred, dim=1) - y_batch[:,1]).tolist()

        total_loss += loss.item()
    model.train()
    return total_loss / (idx+1), (len(preds) - np.asarray(preds).sum()) / len(preds)