# The model training and evaluating function are made to complement kaggle notebooks

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import numpy as np
from sklearn.metrics import f1_score, classification_report, precision_recall_curve, average_precision_score, confusion_matrix, matthews_corrcoef
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay 
import matplotlib.pyplot as plt
import copy
import os
import pickle
import mlflow
from torch.optim.lr_scheduler import CosineAnnealingLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model_train(model, criterion, optimizer, train_loader, val_loader,scheduler=None, 
                save_name=None, epochs=20, patience=3, early_stopping=False, dnn=False, model2=False):
    '''
        Deep Network training function which excpects model, criterion, scheuler, train data loader and val data loader
        Early stopping can be done for regularization by setting early_stopping (default: False)
        and providing early stopping patience (default: 3)
        save_name if provided saves the plot in working directory with "save_name"_loss.png 
        --
        returns : model with least val loss, list of train and val loss per epoch
        
    '''

    if not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer.")
    
    if not isinstance(patience, int) or patience <= 0:
        raise ValueError("patience must be a positive integer.")

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_f1s, val_f1s = [], []
    
    best_val_f1 = 0.0
    best_epoch = 0
    best_model_weights = None
    patience_counter = 0  # for early stopping

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        train_preds_list, train_targets_list = [], []

        for batch in train_loader:
            if model2:
                X_batch, X_batch_mean, mask, y_batch = batch
                X_batch, X_batch_mean, mask, y_batch = X_batch.to(device), X_batch_mean.to(device), mask.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs, _ = model(X_batch, X_batch_mean, mask)
            else:
                X_batch, mask, y_batch = batch
                X_batch, mask, y_batch = X_batch.to(device), mask.to(device), y_batch.to(device)
                optimizer.zero_grad()
                if dnn:
                    outputs = model(X_batch)
                else:
                    outputs, _ = model(X_batch, mask)
                
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler:
                scheduler.step()

            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            train_correct += (preds == y_batch).sum().item()
            train_total += y_batch.size(0)
            train_preds_list.extend(preds.cpu().numpy())
            train_targets_list.extend(y_batch.cpu().numpy())

        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_correct / train_total)
        train_f1s.append(f1_score(train_targets_list, train_preds_list, average='macro', zero_division=0))

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        val_preds_list, val_targets_list = [], []

        with torch.no_grad():
            for batch in val_loader:
                if model2:
                    X_batch, X_batch_mean, mask, y_batch = batch
                    X_batch, X_batch_mean, mask, y_batch = X_batch.to(device), X_batch_mean.to(device), mask.to(device), y_batch.to(device)
                    outputs, _ = model(X_batch, X_batch_mean, mask)
                else:
                    X_batch, mask, y_batch = batch
                    X_batch, mask, y_batch = X_batch.to(device), mask.to(device), y_batch.to(device)
                    if dnn:
                        outputs = model(X_batch)
                    else:
                        outputs, _ = model(X_batch, mask)
                    
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)
                val_preds_list.extend(preds.cpu().numpy())
                val_targets_list.extend(y_batch.cpu().numpy())
                
        current_val_loss = val_loss / len(val_loader)
        current_val_acc = val_correct / val_total
        current_val_f1 = f1_score(val_targets_list, val_preds_list, average='macro', zero_division=0)
        
        val_losses.append(current_val_loss)
        val_accuracies.append(current_val_acc)
        val_f1s.append(current_val_f1)

        # Early stopping logic
        if current_val_f1 > best_val_f1:
            best_val_f1 = current_val_f1
            best_epoch = epoch + 1 
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            if early_stopping:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1} (no improvement in {patience} epochs)")
                    break

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f} | "
              f"Train F1: {train_f1s[-1]:.4f} | Val F1: {val_f1s[-1]:.4f}")

    temp_plot_dir = "/kaggle/working/plots"
    os.makedirs(temp_plot_dir, exist_ok=True)
    plot_filename = f"{save_name}_curves.png"
    plot_path = os.path.join(temp_plot_dir, plot_filename)

    # Plotting with vertical dotted line for best epoch
    plt.figure(figsize=(18, 6))

    if save_name: # Use save_name as the main title if provided
        plt.suptitle(f"Training Metrics for: {save_name}", fontsize=16)
    else: # Fallback title if save_name is not provided
        plt.suptitle("Training Metrics", fontsize=16)
    
    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss')
    best_loss_at_f1_epoch = val_losses[best_epoch-1] if best_epoch > 0 else 0
    plt.axvline(x=best_epoch, color='red', linestyle=':', 
                label=f'Best F1 Epoch: {best_epoch}\nBest F1 val loss: {best_loss_at_f1_epoch:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)

    # Accuracy plot
    plt.subplot(1, 3, 2)
    plt.plot(range(1, len(train_accuracies)+1), train_accuracies, label='Train Acc')
    plt.plot(range(1, len(val_accuracies)+1), val_accuracies, label='Val Acc')
    best_acc_at_f1_epoch = val_accuracies[best_epoch-1] if best_epoch > 0 else 0
    plt.axvline(x=best_epoch, color='red', linestyle=':', 
                label=f'Best F1 Epoch: {best_epoch}\nVal Acc: {best_acc_at_f1_epoch*100:.2f}%')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid(True)

    # F1 plot 
    plt.subplot(1, 3, 3) 
    plt.plot(range(1, len(train_f1s)+1), train_f1s, label='Train Macro F1')
    plt.plot(range(1, len(val_f1s)+1), val_f1s, label='Val Macro F1')
    plt.axvline(x=best_epoch, color='red', linestyle=':',
                label=f'Best Epoch: {best_epoch}\nVal F1: {best_val_f1:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.title('Macro F1 Curve')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    if save_name:
        print(f"Training curves(Loss, accuracy, macro f1) will be saved as {save_name}_curves.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.show()

    # Load best model weights
    model.load_state_dict(best_model_weights)

    # Saving the weights
    temp_model_dir = "/kaggle/working/models"
    os.makedirs(temp_model_dir, exist_ok=True)
    model_filename = f"{save_name}_state_dict.pt"
    model_path = os.path.join(temp_model_dir, model_filename)
    torch.save(model.state_dict(), model_path)
    
    return model, train_losses, val_losses, train_f1s, val_f1s, plot_path, model_path


def evaluate_model(model, test_loader, label_map=None, save_name = None, dnn=False, model2=False):
    '''
        Use save_name to version control
        when save_name is given returns classification report dict
    '''
    model.eval()  # Set model to evaluation mode
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            if model2:
                X_batch, X_batch_mean, mask, y_batch = batch
                X_batch = X_batch.to(device)
                X_batch_mean = X_batch_mean.to(device)
                mask = mask.to(device)
                y_batch = y_batch.to(device)
                outputs, _ = model(X_batch, X_batch_mean, mask)
            else:
                X_batch, mask, y_batch = batch
                X_batch = X_batch.to(device)
                mask = mask.to(device)
                y_batch = y_batch.to(device)
                if dnn:
                    outputs = model(X_batch)
                else:
                    outputs, _ = model(X_batch, mask)
            probs = torch.softmax(outputs, dim=1)  # Shape: (batch_size, num_classes)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(y_batch.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    num_classes = all_probs.shape[1]

    # Compute the Matthews correlation coefficient (MCC)
    mcc = matthews_corrcoef(all_labels, all_preds)
    print(f"Matthews correlation coefficient: {mcc}\n-------------------------------------\n")
    
    # Classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, labels=list(label_map.keys()), target_names=list(label_map.values())))

    temp_plot_dir = "/kaggle/working/plots"
    os.makedirs(temp_plot_dir, exist_ok=True)
    pr_filename = f"{save_name}_PR.png"
    pr_path = os.path.join(temp_plot_dir, pr_filename)
    cm_filename = f"{save_name}_cm.png"
    cm_path = os.path.join(temp_plot_dir, cm_filename)

    # PR Curve for each class
    plt.figure(figsize=(10, 6))
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(all_labels == i, all_probs[:, i])
        avg_precision = average_precision_score(all_labels == i, all_probs[:, i])
        if not label_map:
            plt.plot(recall, precision, label=f'Class {i} (AP = {avg_precision:.2f})')
        else:
            plt.plot(recall, precision, label=f'{label_map[i]} (AP = {avg_precision:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Per-Class)')
    plt.legend(loc='upper right')
    plt.grid()
    if save_name:
        print(f"PR curve will be saved as {save_name}_PR.png")
        plt.savefig(pr_path, bbox_inches='tight', dpi=300)
    plt.show()

    # Confusion Matrix 
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 8))
    
    if not label_map:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(all_labels))
    else:
        labels = [label_map[i] for i in sorted(label_map.keys())]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    
    disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
    plt.title('Confusion Matrix', pad=20, fontsize=14)
    
    # Rotate x-axis labels and adjust layout
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    plt.yticks(rotation=0)
    plt.tight_layout() 
    plt.grid(False)
    plt.gca().set_facecolor('#f8f8f8')  
    if save_name:
        print(f"Confusion Matrix will be saved as {save_name}_cm.png")
        plt.savefig(cm_path, bbox_inches='tight', dpi=300)
    
    plt.show()

    if save_name:
        classification_dict = classification_report(all_labels, all_preds, output_dict=True)
        classification_dict["mcc"] = mcc
        return classification_dict, pr_path, cm_path

class DNNModifiedDataset(Dataset):
    '''
        Flattens the sequence embeddings by mean pooling, useful for simle DNN
    '''
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
    
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        item = self.original_dataset[idx]
        
        features, mask, label = item
        
        if mask.dtype != torch.bool:
            mask = mask.bool() 

        filtered_features = features[mask]
        mean_features = torch.mean(filtered_features, dim=0)
            
        return (mean_features, mask, label)

# Simple feedforward neural network
class DNN_Classifier(nn.Module):
    def __init__(self, input_dim, output_dim,input_neurons=64, hidden_neurons=32, input_dropout=0.3, hidden_dropout=0.3):
        super(DNN_Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_neurons)
        self.bn1 = nn.BatchNorm1d(input_neurons)
        self.dropout1 = nn.Dropout(input_dropout)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_neurons, hidden_neurons)
        self.bn2 = nn.BatchNorm1d(hidden_neurons)
        self.dropout2 = nn.Dropout(hidden_dropout)
        self.fc3 = nn.Linear(hidden_neurons, output_dim) 
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Supports both residual and plain lstm
class BiLSTMAttn(nn.Module):
    '''
        Requires ResLSTM imported from lstm.py in Residual Learning directory to run the residual lstm wit residual=True
    '''

    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1, dropout=0.5, bidirectional=True, lstm_dropout=0, residual=False):
        super().__init__()
        self.lstm = ResLSTM(input_dim, hidden_dim,num_layers=num_layers, batch_first=True, bidirectional=bidirectional) if residual else nn.LSTM(input_dim, hidden_dim,num_layers=num_layers, dropout=lstm_dropout, batch_first=True, bidirectional=bidirectional) 
        self.direction = 2 if bidirectional else 1
        self.ln = nn.LayerNorm(hidden_dim * self.direction)
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.Linear(hidden_dim * self.direction, 1) # since BiLSTM
        self.fc = nn.Linear(hidden_dim * self.direction, num_classes)

    def forward(self, x, mask):
        lstm_out, _ = self.lstm(x)  # (Batch, SequenceLength, 2 * Hidden Dim)
        lstm_out = self.ln(lstm_out)  # Normalize
        lstm_out = self.dropout(lstm_out)  # Dropout

        # Attention scores
        scores = self.attn(lstm_out).squeeze(-1)  # (B, L)

        # Mask padding positions
        scores = scores.masked_fill(~mask, float('-inf'))  # invalid positions → -inf

        attn_weights = torch.softmax(scores, dim=1)  # (B, L)
        attn_weights = attn_weights.unsqueeze(-1)   # (B, L, 1)

        # Weighted sum
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (B, 2H)
        context = self.ln(context) # normalize before FC
        logits = self.fc(context)  # (B, num_classes)

        return logits, attn_weights

class ModifiedDataset(Dataset):
    '''
        Proposed Architecture requires the mean pooled input
    '''
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
    
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        item = self.original_dataset[idx]
        
        features, mask, label = item
        
        if mask.dtype != torch.bool:
            mask = mask.bool() 

        filtered_features = features[mask]
        mean_features = torch.mean(filtered_features, dim=0)
            
        return (features, mean_features, mask, label)

class BiLSTMAttn2(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1, dropout=0.5, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim,num_layers=num_layers, batch_first=True, bidirectional=bidirectional) 
        self.direction = 2 if bidirectional else 1
        self.ln = nn.LayerNorm(hidden_dim * self.direction)
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.Linear(hidden_dim * self.direction, 1) # since BiLSTM
        self.proj = nn.Linear(input_dim, hidden_dim * self.direction)
        self.fc1 = nn.Linear(hidden_dim * self.direction, hidden_dim * self.direction)
        self.fc2 = nn.Linear(hidden_dim * self.direction, num_classes)

    def forward(self, x, x_mean, mask):
        lstm_out, _ = self.lstm(x)  # (Batch, SequenceLength, 2 * Hidden Dim)
        lstm_out = self.ln(lstm_out)  # Normalize
        lstm_out = self.dropout(lstm_out)  # Dropout

        # Attention scores
        scores = self.attn(lstm_out).squeeze(-1)  # (B, L)

        # Mask padding positions
        scores = scores.masked_fill(~mask, float('-inf'))  # invalid positions → -inf

        attn_weights = torch.softmax(scores, dim=1)  # (B, L)
        attn_weights = attn_weights.unsqueeze(-1)   # (B, L, 1)

        # Weighted sum
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (B, 2H)
        context = self.ln(context) # normalize before adding

        # Handling x_mean to shape (B, 2H)
        x_mean_proj = self.proj(x_mean)

        # Adding two tensors
        fused = context + x_mean_proj 
        fused = self.fc1(fused)
        fused = self.ln(fused)
        fused = self.dropout(fused)
        
        logits = self.fc2(fused)  # (B, num_classes)

        return logits, attn_weights

class ResBiLSTMAttn_prev(nn.Module):
    '''
        Requires ResLSTM imported as ResLSTM_previous from lstm_prev in Residual Learning
    '''
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1, dropout=0.5, bidirectional=True):
        super().__init__()
        self.lstm = ResLSTM_previous(input_dim, hidden_dim,num_layers, batch_first=True, bidirectional = bidirectional)
        self.ln = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.Linear(hidden_dim*2 , 1)
        self.fc = nn.Linear(hidden_dim *2, num_classes)

    def forward(self, x, mask):
        lstm_out, _ = self.lstm(x)  # (Batch, SequenceLength, 2 * Hidden Dim)
        lstm_out = self.ln(lstm_out)  # Normalize
        lstm_out = self.dropout(lstm_out)  # Dropout

        # Attention scores
        scores = self.attn(lstm_out).squeeze(-1)  # (B, L)

        # Mask padding positions
        scores = scores.masked_fill(~mask, float('-inf'))  # invalid positions → -inf

        attn_weights = torch.softmax(scores, dim=1)  # (B, L)
        attn_weights = attn_weights.unsqueeze(-1)   # (B, L, 1)

        # Weighted sum
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (B, 2H)
        context = self.ln(context) # normalize before FC
        logits = self.fc(context)  # (B, num_classes)

        return logits, attn_weights

def get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        # Linear lr increase
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # Linearly decay lr
        return max(
            0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
        )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# --- Helper to save data to temporary files for MLflow logging ---
def save_temp_file(data, filename_prefix, artifact_subdir):
    temp_dir = os.path.join("/kaggle/working", artifact_subdir)
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, filename_prefix)
    with open(temp_path, "wb") as f:
        pickle.dump(data, f)
    return temp_path

def run_cross_validation(arch_config, full_dataset, all_labels_for_skf, device, label_map, n_splits=5):
    '''
        arch_config: dict = {
            "name" : Name to save and name run in Mlflow,
            "model_class": one of the model classes above to use,
            "dataset_wrapper": Provide Dataset class instance from above if dataset requires modification like DNN and BiLSTM2 ,
            "params" : dict of hyperparameters for model training, DNN has different set of parameters compared to LSTM implemetations,
            "is_dnn" : Bool to denote if the model is simple DNN,
            "is_model2" : Bool to denote if proposed BiLSTM is being used,
            "early_stopping"  : training function supports early stopping for f1, for our experiment we won't use it,
            "patience": patience if early stopping is used
            "residual" : Bool to denote if the model is using residual learning for BiLSTMAttn class,
            "scheduler" : linear-warmup or cosine else None
        },
        full_dataset: full dataset that is not split,
        all_labels_for_skf: all the labels retreived and made numpy array from full dataset,
        label_map: label map for int class values,
        n_splits: defaults 5 which splits 80:20 for in each train and val split in each fold
    '''
    if arch_config["name"] is None or not isinstance(arch_config["name"],str):
        raise ValueError("Provide name for the arch_config")

    if arch_config['scheduler'] is not None and arch_config['scheduler'] not in ["linear_warmup", "cosine"]:
        raise ValueError("arch_config's scheduler must be None or 'linear_warmup' or 'cosine'")

    if arch_config["params"] is None:
        raise ValueError("provide params for training")

    architecture_name = arch_config['name']
    
    # === PARENT RUN for each Architecture ===
    with mlflow.start_run(run_name=f"{architecture_name}") as parent_run:
        mlflow.log_params(arch_config["params"]) # Log architecture-specific parameters
        for key, value in arch_config.items():
            if key not in ["params"]:
                mlflow.log_param(key, value)

        print(f"\nStarting Cross-Validation for: {architecture_name}")

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        fold_final_reports = [] # To store final evaluation reports for aggregation
        metrics_for_plotting = [] # To store epoch-wise metrics from each fold for avg/std plotting

        # Apply dataset wrapper if specified
        current_full_dataset = full_dataset

        if arch_config["dataset_wrapper"]:
            print(f"  Applying dataset wrapper: {arch_config['dataset_wrapper'].__name__}")
            current_full_dataset = arch_config["dataset_wrapper"](full_dataset)

        for fold_idx, (train_ids, val_ids) in enumerate(skf.split(np.arange(len(current_full_dataset)), all_labels_for_skf)):
            # Create PyTorch Subsets for the current fold
            train_subset = Subset(current_full_dataset, train_ids)
            val_subset = Subset(current_full_dataset, val_ids)

            train_loader = DataLoader(train_subset, batch_size=arch_config["params"]["train_batch"], shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=arch_config["params"]["val_batch"], shuffle=False)

            # --- Initialize Model, Optimizer, Criterion for this Fold ---
            # Each fold gets a fresh model instance to avoid data leakage
            model = arch_config["model_class"](**arch_config["params"])
            criterion = nn.CrossEntropyLoss(label_smoothing = arch_config["params"]["label_smoothing"])
            optimizer = torch.optim.Adam(model.parameters(), lr=arch_config["params"]["lr"]) if arch_config["is_dnn"] else torch.optim.AdamW(model.parameters(), lr=arch_config["params"]["lr"], weight_decay=arch_config["params"]["weight_decay"])
            
            scheduler = None 
            if arch_config['scheduler'] == "linear_warmup":
                total_steps = len(train_loader) * arch_config["params"]["epochs"]
                warmup_steps = int(arch_config["params"]["warmup_ratio"] * total_steps)
                scheduler = get_linear_warmup_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)
            if arch_config['scheduler'] == "cosine":
                scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * arch_config["params"]["epochs"])
                

            # --- Train Model for this Fold ---
            model, train_losses, val_losses, train_f1s, val_f1s, plot_path, model_path = model_train(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                scheduler=scheduler,
                save_name=f"{architecture_name}_fold_{fold_idx+1}", 
                epochs=arch_config["params"]["epochs"],
                patience=arch_config["params"]["patience"],
                early_stopping=arch_config["params"]["early_stopping"],
                dnn=arch_config["is_dnn"],
                model2=arch_config["is_model2"]
            )

            # === CHILD RUN for each Fold ===
            with mlflow.start_run(run_name=f"Fold {fold_idx + 1}", nested=True) as child_run:
                print(f"  Logging Fold {fold_idx + 1} results under run_id: {child_run.info.run_id}")

                mlflow.log_param("fold_index", fold_idx + 1)

                # Log epoch-wise metrics from train_output to child run artifacts
                mlflow.log_artifact(
                    local_path=save_temp_file({"train_losses": train_losses, "val_losses": val_losses},
                                              f"{architecture_name}_fold_{fold_idx+1}_losses.pkl", "metrics_raw"),
                    artifact_path="metrics_raw"
                )
                mlflow.log_artifact(
                    local_path=save_temp_file({"train_f1s": train_f1s, "val_f1s": val_f1s},
                                              f"{architecture_name}_fold_{fold_idx+1}_f1s.pkl", "metrics_raw"),
                    artifact_path="metrics_raw"
                )

                mlflow.log_artifact(plot_path, artifact_path="curves")

                # Log the best model's state dict for this fold
                mlflow.log_artifact(model_path, artifact_path="model_weights")
                print(f"  Logged best model for Fold {fold_idx+1} to artifacts/model_weights/{model_path}")

                # --- Evaluate Model for this Fold ---
                eval_report_dict, pr_plot_path, cm_plot_path = evaluate_model(
                    model=model, # Model with best weights loaded
                    test_loader=val_loader, # Using val_loader as test for fold evaluation
                    label_map=label_map,
                    save_name=f"{architecture_name}_fold_{fold_idx+1}",
                    dnn=arch_config["is_dnn"],
                    model2=arch_config["is_model2"]
                )

                # Log evaluation metrics to child run
                mlflow.log_metrics({
                    "fold_accuracy": eval_report_dict["accuracy"],
                    "fold_macro_precision": eval_report_dict["macro avg"]["precision"],
                    "fold_macro_recall": eval_report_dict["macro avg"]["recall"],
                    "fold_macro_f1": eval_report_dict["macro avg"]["f1-score"],
                    "fold_weighted_f1": eval_report_dict["weighted avg"]["f1-score"],
                    "fold_mcc": eval_report_dict["mcc"]
                })

                # Log evaluation plots to child run
                mlflow.log_artifact(pr_plot_path, artifact_path="evaluation_plots_per_fold")
                mlflow.log_artifact(cm_plot_path, artifact_path="evaluation_plots_per_fold")
                os.remove(pr_plot_path) # Clean up
                os.remove(cm_plot_path) # Clean up
                print(f"  Logged evaluation plots for Fold {fold_idx+1}")

                # Store this fold's aggregated evaluation report for parent aggregation
                fold_final_reports.append(eval_report_dict)
                
                # Store epoch-wise metrics for the parent plot
                metrics_for_plotting.append({
                    'train_loss': train_output['train_losses'],
                    'val_loss': train_output['val_losses'],
                    'train_f1': train_output['train_f1s'],
                    'val_f1': train_output['val_f1s']
                })

        # --- Aggregate and Log Metrics at Parent Run Level ---
        print(f"\nAggregating results for {architecture_name} across {n_splits} folds...")
        # Calculate mean across relevant metrics from fold_final_reports
        avg_accuracy = np.mean([r["accuracy"] for r in fold_final_reports])
        avg_macro_precision = np.mean([r["macro avg"]["precision"] for r in fold_final_reports])
        avg_macro_recall = np.mean([r["macro avg"]["recall"] for r in fold_final_reports])
        avg_macro_f1 = np.mean([r["macro avg"]["f1-score"] for r in fold_final_reports])
        avg_weighted_f1 = np.mean([r["weighted avg"]["f1-score"] for r in fold_final_reports])
        avg_mcc = np.mean([r["mcc"] for r in fold_final_reports])

        # Log aggregated metrics to the parent run
        mlflow.log_metrics({
            "avg_accuracy": avg_accuracy,
            "avg_macro_precision": avg_macro_precision,
            "avg_macro_recall": avg_macro_recall,
            "avg_macro_f1": avg_macro_f1,
            "avg_weighted_f1": avg_weighted_f1,
            "avg_mcc": avg_mcc
        })
        
        print(f"Overall Results for {architecture_name}:")
        print(f"  Avg Accuracy: {avg_accuracy:.4f}")
        print(f"  Avg Macro F1: {avg_macro_f1:.4f}")
        print(f"  Avg Weighted F1: {avg_weighted_f1:.4f}")
        print(f"  Avg MCC: {avg_mcc:.4f}")
    
    return {
        'arch_name': architecture_name,
        'fold_metrics': metrics_for_plotting
    }