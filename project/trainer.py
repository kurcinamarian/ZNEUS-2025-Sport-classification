
import time
from pathlib import Path

import pandas as pd

import wandb
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import numpy as np

from dataloader import create_dataloaders
from model import CNN

# Initial parameters

CSV_PATH = r".\.scratch\dataset\sports.csv"
ROOT_DIR = r".\.scratch\dataset"

EPOCHS = 50
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
NUM_WORKERS = 2
EARLY_STOP_PATIENCE = 10

OUT_DIR = Path("runs") / "sports_cnn_v1"

USE_WANDB = True
WANDB_PROJECT = "zneus-project-2"
WANDB_RUN_NAME = "sports_cnn_1"
WANDB_LOG_FREQUENCY = 500

@torch.no_grad()
def evaluate(model, loader, device, criterion, return_predictions=False):
    
    model.eval()
    
    total_loss, total, correct = 0.0, 0, 0
    all_preds = []      # Predicted class_ids
    all_labels = []     # True labels
    all_probs = []      # Predicted probability for each class
    
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        
        probs = torch.softmax(outputs, dim=1)
        _, preds = outputs.max(1)
        
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        
        if return_predictions:
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    avg_loss = total_loss / max(1, total)
    accuracy = correct / max(1, total)
    
    if return_predictions:
        return avg_loss, accuracy, np.array(all_preds), np.array(all_labels), np.array(all_probs)
    
    return avg_loss, accuracy

def get_metrics(y_true, y_pred, y_probs):
    
    # Calculte metrics for evaluation
    
    num_classes = 100
    
    # Weighted average metrics (f1, precision, recall)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Average metrics (f1, precision, recall)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate per-class specificity
    specificity_per_class = []
    for i in range(num_classes):
        true_negative = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        false_positive = cm[:, i].sum() - cm[i, i]
        specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive > 0) else 0
        specificity_per_class.append(specificity)
        
    metrics = {
        'f1_weighted': f1_weighted,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_macro': f1_macro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'confusion_matrix': cm,
        'classification_report': report,
        'specificity_per_class': specificity_per_class
    }
    
    return metrics

def train_one_epoch(model, loader, device, optimizer, criterion, scaler):
    
    # Training for one epoch, return average loss and accuracy
    
    model.train()
    total_loss, total, correct = 0.0, 0, 0
    for images, labels in loader:
        # Get images and labels
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True) # Clear gradients

        if device.type == 'cuda': # GPU training with scaler
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else: # CPU training
            with torch.amp.autocast(device_type="cpu"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    
    return (total_loss / max(1, total)), (correct / max(1, total))

def save_checkpoint(path, model, optimizer, scheduler, epoch, best_acc):
    # Save the current state
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_acc": best_acc,
    }, path)
    
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Make dataloaders
    train_loader, valid_loader, test_loader = create_dataloaders(
        csv_path=CSV_PATH,
        root_dir=ROOT_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    if USE_WANDB:
        wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_RUN_NAME,
            config={
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "label_smoothing": LABEL_SMOOTHING,
                "early_stopping_patience": EARLY_STOP_PATIENCE,
                "num_classes": 100
            }
        )

    model = CNN().to(device)
    
    if USE_WANDB:
        wandb.watch(model, log="all", log_freq=WANDB_LOG_FREQUENCY)
    
    # Loss function: CrossEntropyLoss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    
    # Optimizer: Adam with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler: CosineAnnealingLR (gradual LR decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Scaler: Mixed precision training for faster GPU training -> usage suggested by ChatGPT
    scaler = torch.amp.GradScaler(device.type if device.type == "cuda" else "cpu")
    
    best_path = OUT_DIR / "best.pt"
    last_path = OUT_DIR / "last.pt"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0 # current best validation accuracy
    epochs_no_improvement = 0 # Count epoch without improvement for early stopping
    
    # Training loop
    for epoch in range(EPOCHS):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, device, optimizer, criterion, scaler)
        
        # Log detailed metrics every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_loss, val_acc, val_preds, val_labels, val_probs = evaluate(
                model, valid_loader, device, criterion, return_predictions=True)
            val_metrics = get_metrics(val_preds, val_labels, val_probs)
        else:
            val_loss, val_acc = evaluate(model, valid_loader, device, criterion, return_predictions=False)
            val_metrics = None
        
        scheduler.step()
        dt = time.time() - t0
        
        print(
            f"Epoch {epoch+1:03d}/{EPOCHS} | "
            f"train {train_loss:.4f}/{train_acc:.4f} | "
            f"val {val_loss:.4f}/{val_acc:.4f} | "
            f"{dt:.1f}s")
        
        if USE_WANDB:
            # Log metrics
            log_dict = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "lr": optimizer.param_groups[0]["lr"]
            }
            # Log detailed metrics
            if val_metrics is not None:
                log_dict["val_f1_weighted"] = val_metrics["f1_weighted"]
                log_dict["val_precision_weighted"] = val_metrics["precision_weighted"]
                log_dict["val_recall_weighted"] = val_metrics["recall_weighted"]
                log_dict["val_f1_macro"] = val_metrics["f1_macro"]
                log_dict["val_precision_macro"] = val_metrics["precision_macro"]
                log_dict["val_recall_macro"] = val_metrics["recall_macro"]
            wandb.log(log_dict)
        
        save_checkpoint(last_path, model, optimizer, scheduler, epoch, best_val_acc) # Save current state
        
        # Update best state (Save new best state if it is better than the current best state)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improvement = 0
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_val_acc)
            print(f"New best acc: {best_val_acc:.4f} -> {best_path}")
        else:
            epochs_no_improvement += 1
        # If too many epochs without improvement, stop early
        if epochs_no_improvement >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
    if best_path.exists():
        # Load best checkpoint for test evaluation
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])

    # Test evaluation
    
    test_loss, test_acc, test_preds, test_labels, test_probs = evaluate(
        model, test_loader, device, criterion, return_predictions=True)
    test_metrics = get_metrics(test_preds, test_labels, test_probs)
    
    print(f"TEST | loss={test_loss:.4f} acc={test_acc:.4f}")
    print(f"TEST | F1-weighted={test_metrics['f1_weighted']:.4f} "
          f"Precision={test_metrics['precision_weighted']:.4f} "
          f"Recall={test_metrics['recall_weighted']:.4f}")
    
    if USE_WANDB:
        # Log metrics
        wandb.log({
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "test_f1_weighted": test_metrics["f1_weighted"],
            "test_precision_weighted": test_metrics["precision_weighted"],
            "test_recall_weighted": test_metrics["recall_weighted"],
        })
        
        # Log detailed 100x100 confusion matrix
        wandb.log({"confusion_matrix_detailed": wandb.plot.confusion_matrix(
            probs=None,
            y_true=test_labels,
            preds=test_preds,
            class_names=[str(i) for i in range(100)]
        )})
        
        # Create simplified confusion matrix (correct vs incorrect)
        binary_labels = [1] * len(test_labels)
        binary_preds = [1 if test_labels[i] == test_preds[i] else 0 for i in range(len(test_labels))]
        
        wandb.log({"confusion_matrix_simplified": wandb.plot.confusion_matrix(
            probs=None,
            y_true=binary_labels,
            preds=binary_preds,
            class_names=["Incorrect", "Correct"]
        )})
        
        df = pd.read_csv(CSV_PATH)
        id_to_label = df.groupby("class id")["labels"].first().to_dict()
        
        # Log per-class metrics
        per_class_data = []
        report = test_metrics['classification_report']
        for class_id in range(100):
            class_str = str(class_id)
            if class_str in report:
                per_class_data.append([
                    class_id,
                    id_to_label.get(class_id, class_str),
                    report[class_str]['precision'],
                    report[class_str]['recall'],
                    report[class_str]['f1-score'],
                    test_metrics['specificity_per_class'][class_id],
                    report[class_str]['support']
                ])
                
        per_class_table = wandb.Table(
            columns=["class_id", "class_label", "precision", "recall", "f1", "specificity", "support"],
            data=per_class_data
        )
        wandb.log({"per_class_metrics": per_class_table})
        
        wandb.log({
            "per_class_f1": wandb.plot.bar(
                per_class_table, "class_label", "f1",
                title="F1 score per class"
            ),
            "per_class_precision": wandb.plot.bar(
                per_class_table, "class_label", "precision",
                title="Precision per class"
            ),
            "per_class_recall": wandb.plot.bar(
                per_class_table, "class_label", "recall",
                title="Recall per class"
            )
        })
        
        wandb.finish()


if __name__ == "__main__":
    main()

