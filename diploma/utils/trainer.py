import torch
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
    model.train()
    global_step = 0
    best_score = 999999
    for _ in tqdm(range(epochs)):
        training_step = 0
        sum_loss = 0
        for batch in train_dataloader:    
            optimizer.zero_grad()
            y_pred = model(batch)
            loss = 0
            for loss_func in loss_functions:
                loss += loss_func(y_pred, batch)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            training_step += 1
            global_step += 1
            if evaluation_steps > 0 and training_step % evaluation_steps == 0:
                validation_loss = evaluate_model(
                    model=model,
                    validation_loader=validation_dataloader,
                    loss_functions=loss_functions
                )
                train_losses.append(sum_loss/training_step)
                validation_losses.append(validation_loss)

                if validation_loss < best_score:
                    best_score = validation_loss
                    torch.save(model.state_dict(), best_model_path)

@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    validation_loader: DataLoader,
    loss_functions
) -> float:
    """
    Evaluates model on evaluation dataloader and returns loss
    """
    total_loss = 0 
    model.eval()
    for idx, batch in enumerate(validation_loader):
        y_pred = model(batch)
        loss = 0
        for loss_func in loss_functions:
            loss += loss_func(y_pred, batch)
        total_loss += loss.item()
    model.train()
    return total_loss / (idx+1)