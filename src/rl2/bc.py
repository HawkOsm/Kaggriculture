import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from collections import Counter
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy import BCPolicy

def run():
    out_dir = Path("/home/osm/Projects/Kaggriculture/.claude/scratch/rl2")
    X_path = out_dir / "demo_X.npy"
    y_path = out_dir / "demo_y.npy"
    spec_path = out_dir / "feature_spec.json"
    
    print("Loading data...")
    X = np.load(X_path)
    y = np.load(y_path)
    with open(spec_path, "r") as f:
        spec = json.load(f)
    vocab = spec["vocab"]
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Split
    split_idx = int(0.9 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Standardize
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0 # avoid div by zero
    
    np.savez(out_dir / "scaler.npz", mean=mean, std=std)
    
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    
    # Majority baseline
    counts = Counter(y_val)
    majority_class, majority_count = counts.most_common(1)[0]
    maj_acc = majority_count / len(y_val)
    
    print(f"Majority-class baseline accuracy: {maj_acc:.4f} (Class {vocab[majority_class]})")
    
    # DataLoaders
    batch_size = 2048
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BCPolicy(in_dim=X.shape[1], out_dim=len(vocab)).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    patience = 5
    stagnant = 0
    
    print("Training...")
    for epoch in range(50):
        model.train()
        train_loss = 0.0
        for bx, by in train_dl:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for bx, by in val_dl:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                preds = logits.argmax(dim=1)
                val_correct += (preds == by).sum().item()
                
        val_acc = val_correct / len(val_ds)
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss/len(train_dl):.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            stagnant = 0
            torch.save(model.state_dict(), out_dir / "bc_model.pt")
        else:
            stagnant += 1
            if stagnant >= patience:
                print("Early stopping triggered.")
                break
                
    print("Evaluating best model...")
    model.load_state_dict(torch.load(out_dir / "bc_model.pt"))
    model.eval()
    
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for bx, by in val_dl:
            bx = bx.to(device)
            logits = model(bx)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(by.numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    acc = (all_preds == all_targets).mean()
    print(f"\nFinal Val Top-1 Accuracy: {acc:.4f}")
    print("\nPer-class Val Accuracy:")
    for i in range(len(vocab)):
        mask = (all_targets == i)
        if mask.sum() > 0:
            c_acc = (all_preds[mask] == all_targets[mask]).mean()
            print(f"{vocab[i]:>20}: {c_acc:.4f} ({mask.sum()} samples)")
        else:
            print(f"{vocab[i]:>20}: N/A (0 samples)")

if __name__ == "__main__":
    run()
