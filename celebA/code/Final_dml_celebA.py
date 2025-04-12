import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import os
from PIL import Image
import time
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.multiprocessing as mp

# ========== Config ==========
class Config:
    batch_size = 32
    lr = 1e-3
    num_classes = 40
    grad_clip = 1.0
    grad_accum_steps = 1
    max_epochs = 10
    early_stop_patience = 5
    T = 3.0
    alpha_conf = 0.1
    alpha_cos = 0.3
    alpha_ensemble = 0.2
    threshold = 0.5
    root_dir = 'C:/celeba/img_align_celeba/img_align_celeba'
    csv_path = 'C:/celeba/list_attr_celeba.csv'
    partition_csv_path = 'C:/celeba/list_eval_partition.csv'
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

# ========== Dataset ==========
class CelebADataset(Dataset):
    def __init__(self, root_dir, csv_path, partition_csv_path, split, transform=None):
        self.df = pd.read_csv(csv_path).replace(-1, 0)
        partition_df = pd.read_csv(partition_csv_path)
        partition_df['partition'] = partition_df['partition'].astype(int)
        self.df = self.df.merge(partition_df, on='image_id', how='inner')

        split_codes = {'train': 0, 'valid': 1, 'validation': 1, 'test': 2}
        split_code = split_codes.get(split.lower())
        if split_code is None:
            raise ValueError(f"Invalid split: {split}")

        self.df = self.df[self.df['partition'] == split_code].copy()
        self.root_dir = root_dir
        self.transform = transform
        self.labels = self.df.drop(['image_id', 'partition'], axis=1).values.astype('float32')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.df.iloc[idx]['image_id'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

# ========== Model ==========
def get_resnet18(device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, Config.num_classes)
    return model.to(device)

# ========== Loss Functions ==========
def confidence_regularization(logits):
    probs = torch.sigmoid(logits)
    probs = torch.clamp(probs, min=1e-6, max=1-1e-6)
    return -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))

def pseudo_teacher_kl(logits, pseudo_probs, T):
    pseudo_probs = torch.clamp(pseudo_probs, min=1e-6)
    log_probs = F.log_softmax(logits / T, dim=1)
    return F.kl_div(log_probs, pseudo_probs.detach(), reduction='batchmean') * (T**2)

def enhanced_mutual_loss(out1, out2, T):
    kl = F.kl_div(F.log_softmax(out1 / T, dim=1), F.softmax(out2.detach() / T, dim=1), reduction='batchmean') * (T**2)
    cos = -F.cosine_similarity(out1, out2.detach()).mean()
    conf = confidence_regularization(out1) + confidence_regularization(out2)
    ensemble_probs = (F.softmax(out1 / T, dim=1) + F.softmax(out2 / T, dim=1)) / 2
    pseudo_kl_1 = pseudo_teacher_kl(out1, ensemble_probs, T)
    pseudo_kl_2 = pseudo_teacher_kl(out2, ensemble_probs, T)
    return kl + Config.alpha_cos * cos + Config.alpha_conf * conf + Config.alpha_ensemble * (pseudo_kl_1 + pseudo_kl_2)

# ========== Evaluation ==========
def evaluate_model(model, test_loader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = torch.sigmoid(model(images))
            preds = (outputs > Config.threshold).float().cpu().numpy()
            y_pred.append(preds)
            y_true.append(targets.numpy())

    y_pred = np.vstack(y_pred)
    y_true = np.vstack(y_true)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print(f"Test Accuracy:  {acc:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall:    {recall:.4f}")
    print(f"Test F1 Score:  {f1:.4f}")
    return acc, precision, recall, f1

def evaluate_ensemble(model1, model2, test_loader, device):
    model1.eval()
    model2.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Evaluating Ensemble"):
            images = images.to(device)
            out1 = torch.sigmoid(model1(images))
            out2 = torch.sigmoid(model2(images))
            avg_output = (out1 + out2) / 2
            preds = (avg_output > Config.threshold).float().cpu().numpy()
            y_pred.append(preds)
            y_true.append(targets.numpy())

    y_pred = np.vstack(y_pred)
    y_true = np.vstack(y_true)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print(f"\n--- Ensemble Evaluation ---")
    print(f"Test Accuracy:  {acc:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall:    {recall:.4f}")
    print(f"Test F1 Score:  {f1:.4f}")
    return acc, precision, recall, f1

# ========== Training ==========
def train_dml(train_loader, val_loader, model_s1, model_s2, device):
    optimizer_s1 = optim.AdamW(model_s1.parameters(), lr=Config.lr, weight_decay=0.05)
    optimizer_s2 = optim.AdamW(model_s2.parameters(), lr=Config.lr, weight_decay=0.05)
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=torch.ones(Config.num_classes).to(device) * 1.2)
    scaler = torch.cuda.amp.GradScaler()

    best_val_loss = float('inf')
    patience = 0

    for epoch in range(Config.max_epochs):
        model_s1.train()
        model_s2.train()
        running_loss = 0.0

        progress = tqdm(train_loader, desc=f"[Epoch {epoch+1}] Training")
        for i, (images, targets) in enumerate(progress):
            images, targets = images.to(device), targets.to(device)

            with torch.cuda.amp.autocast():
                out1 = model_s1(images)
                out2 = model_s2(images)
                task_loss = 0.7 * (criterion_bce(out1, targets) + criterion_bce(out2, targets))
                mutual = enhanced_mutual_loss(out1, out2, Config.T)
                total_loss = task_loss + 0.3 * mutual

            if torch.isnan(total_loss):
                print("NaN detected in loss")
                continue

            scaler.scale(total_loss).backward()

            if (i + 1) % Config.grad_accum_steps == 0:
                scaler.unscale_(optimizer_s1)
                scaler.unscale_(optimizer_s2)
                torch.nn.utils.clip_grad_norm_(model_s1.parameters(), Config.grad_clip)
                torch.nn.utils.clip_grad_norm_(model_s2.parameters(), Config.grad_clip)
                scaler.step(optimizer_s1)
                scaler.step(optimizer_s2)
                scaler.update()
                optimizer_s1.zero_grad(set_to_none=True)
                optimizer_s2.zero_grad(set_to_none=True)

            running_loss += total_loss.item()
            progress.set_postfix(loss=total_loss.item())

        model_s1.eval()
        model_s2.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"[Epoch {epoch+1}] Validating"):
                images, targets = images.to(device), targets.to(device)
                out1 = model_s1(images)
                out2 = model_s2(images)
                val_loss += criterion_bce(out1, targets).item() + criterion_bce(out2, targets).item()

        avg_val_loss = val_loss / (2 * len(val_loader))
        print(f"Epoch {epoch+1} | Train Loss: {running_loss / len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience = 0
            torch.save(model_s1.state_dict(), 'best_student1.pth')
            torch.save(model_s2.state_dict(), 'best_student2.pth')
        else:
            patience += 1
            if patience >= Config.early_stop_patience:
                print("Early stopping triggered.")
                break

    print("Training complete.")

# ========== Main ==========
def main():
    mp.set_start_method("spawn", force=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(Config.mean, Config.std)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(Config.mean, Config.std)
    ])

    train_set = CelebADataset(Config.root_dir, Config.csv_path, Config.partition_csv_path, 'train', train_transform)
    val_set = CelebADataset(Config.root_dir, Config.csv_path, Config.partition_csv_path, 'valid', eval_transform)
    test_set = CelebADataset(Config.root_dir, Config.csv_path, Config.partition_csv_path, 'test', eval_transform)

    train_loader = DataLoader(train_set, batch_size=Config.batch_size, shuffle=True, num_workers=6,pin_memory=True,persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=Config.batch_size, shuffle=False, num_workers=6,pin_memory=True,persistent_workers=True)
    test_loader = DataLoader(test_set, batch_size=Config.batch_size, shuffle=False, num_workers=6,pin_memory=True,persistent_workers=True)

    model_s1 = get_resnet18(device)
    model_s1.load_state_dict(torch.load("C:/Users/akash/Downloads/best_model_attention.pth", map_location=device), strict=False)
    model_s2 = get_resnet18(device)
    model_s2.load_state_dict(torch.load("C:/Users/akash/Downloads/best_model_soft.pth", map_location=device), strict=False)

    train_dml(train_loader, val_loader, model_s1, model_s2, device)

    model_s1.load_state_dict(torch.load("best_student1.pth"))
    model_s2.load_state_dict(torch.load("best_student2.pth"))

    print("\n--- Evaluating Student 1 ---")
    evaluate_model(model_s1, test_loader, device)

    print("\n--- Evaluating Student 2 ---")
    evaluate_model(model_s2, test_loader, device)

    evaluate_ensemble(model_s1, model_s2, test_loader, device)

if __name__ == "__main__":
    main()