
import time
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

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
WANDB_RUN_NAME = "sports_cnn_model3_1"
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

def create_per_class_charts(preds, labels, probs, csv_path, prefix=""):
    
    # Create per-class metric bar charts as Images
    
    metrics = get_metrics(preds, labels, probs)
    
    df = pd.read_csv(csv_path)
    id_to_label = df.groupby("class id")["labels"].first().to_dict()
    
    # Build data
    per_class_data = []
    report = metrics['classification_report']
    for class_id in range(100):
        class_str = str(class_id)
        if class_str in report:
            per_class_data.append([
                class_id,
                id_to_label.get(class_id, class_str),
                report[class_str]['precision'],
                report[class_str]['recall'],
                report[class_str]['f1-score'],
                metrics['specificity_per_class'][class_id],
                report[class_str]['support']
            ])
    
    class_labels = [row[1] for row in per_class_data]
    class_ids = [row[0] for row in per_class_data]
    
    # Compute per-class accuracy from confusion matrix
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix
    cm = sk_confusion_matrix(labels, preds, labels=list(range(100)))
    supports = [row[6] for row in per_class_data]
    accuracy_values = [float(cm[i, i]) / supports[i] if supports[i] else 0.0 for i in range(len(supports))]

    metrics_to_plot = {
        'accuracy': (accuracy_values, 'Accuracy per Class'),
        'f1': ([row[4] for row in per_class_data], 'F1 Score per Class'),
        'precision': ([row[2] for row in per_class_data], 'Precision per Class'),
        'recall': ([row[3] for row in per_class_data], 'Recall per Class')
    }
    
    log_dict = {}
    for metric_name, (values, title) in metrics_to_plot.items():
        fig, ax = plt.subplots(figsize=(20, 6))
        ax.bar(class_ids, values)
        ax.set_xlabel('Class ID')
        ax.set_ylabel(metric_name.capitalize())
        ax.set_title(f"{prefix}{title}" if prefix else title)
        ax.set_xticks(class_ids)
        ax.set_xticklabels(class_labels, rotation=90, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        log_dict[f"{prefix}per_class_{metric_name}"] = wandb.Image(fig)
        plt.close(fig)
    
    return log_dict

def create_confusion_matrix_images(y_true, y_pred, class_names=None, title_prefix=""):
    
    # Create confusion matrix images

    log_dict = {}

    unique_classes = sorted(set(y_true) | set(y_pred))
    n_classes = len(unique_classes)
    labels = list(range(100)) if class_names is None else list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # Row-normalize
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype(np.float32)
        row_sums = cm_norm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm_norm, row_sums, out=np.zeros_like(cm_norm), where=row_sums!=0)

    figsize = (min(14, max(8, n_classes/8)), min(12, max(6, n_classes/10)))
    fig1, ax1 = plt.subplots(figsize=figsize)
    im = ax1.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    ax1.set_title(f"{title_prefix}Confusion Matrix (row-normalized)")
    ax1.set_xlabel('Predicted label')
    ax1.set_ylabel('True label')
    cbar = fig1.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Proportion', rotation=90)

    if n_classes <= 20 and class_names is not None:
        ax1.set_xticks(range(n_classes))
        ax1.set_yticks(range(n_classes))
        ax1.set_xticklabels(class_names, rotation=90, fontsize=8)
        ax1.set_yticklabels(class_names, fontsize=8)
    else:
        ax1.set_xticks([])
        ax1.set_yticks([])
    plt.tight_layout()
    log_dict[f"{title_prefix}confusion_matrix_img"] = wandb.Image(fig1)
    plt.close(fig1)

    return log_dict

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

    # Final evaluation on all sets
    train_loss, train_acc, train_preds, train_labels, train_probs = evaluate(
        model, train_loader, device, criterion, return_predictions=True)
    
    val_loss, val_acc, val_preds, val_labels, val_probs = evaluate(
        model, valid_loader, device, criterion, return_predictions=True)
    
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
        
        # Log confusion matrix
        cm_images = create_confusion_matrix_images(
            test_labels, test_preds,
            class_names=[str(i) for i in range(100)],
            title_prefix="final_test_"
        )
        wandb.log(cm_images)
        
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
        
        # Create per-class charts for all sets
        train_chart_dict = create_per_class_charts(
            train_preds, train_labels, train_probs, CSV_PATH, prefix="final_train_"
        )
        val_chart_dict = create_per_class_charts(
            val_preds, val_labels, val_probs, CSV_PATH, prefix="final_val_"
        )
        test_chart_dict = create_per_class_charts(
            test_preds, test_labels, test_probs, CSV_PATH, prefix="final_test_"
        )
        wandb.log({**train_chart_dict, **val_chart_dict, **test_chart_dict})
        
        wandb.finish()


if __name__ == "__main__":
    main()

