# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 10:53:05 2026
@author: user
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import random
import copy
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import joblib
# --- 0. REPRODUCIBILITY ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# --- 1. SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_dir = r'C:\Users\user\Desktop\Bioinformatics_Data\8749 samples'
file_path = os.path.join(base_dir, '8744_real_patients.csv')
# --- 2. DATA LOADING ---
print("Loading real 8,744-patient dataset...")
df = pd.read_csv(file_path).reset_index(drop=True)

# Explicit, known metadata/label columns to exclude from the gene matrix
known_meta_cols = ['sampleID', 'sample_type', 'Final_Label', 'Final_Label1', 'Final_Label2',
                    'PAM50Call_RNAseq', 'PAM50_mRNA_nature2012', '_cohort']
cols_to_drop = [c for c in df.columns if c in known_meta_cols]
genes_df = df.drop(columns=cols_to_drop)

# Safety net: catch any remaining non-numeric column we didn't anticipate by name
non_numeric_check = genes_df.apply(lambda col: pd.to_numeric(col, errors='coerce').isna().all())
extra_non_numeric = non_numeric_check[non_numeric_check].index.tolist()
if extra_non_numeric:
    print(f"⚠ Dropping additional non-numeric columns not in the known list: {extra_non_numeric}")
    genes_df = genes_df.drop(columns=extra_non_numeric)

X_numeric = genes_df.apply(pd.to_numeric, errors='coerce').fillna(0)
X = X_numeric.values
print(f"Final gene feature count: {X.shape[1]}")
assert X.shape[1] == 5000, f"Expected 5000 genes, got {X.shape[1]} - check column filtering above"
X_tensor = torch.tensor(X, dtype=torch.float32)
print(f"Loaded {len(df)} rows, {X.shape[1]} genes.")
print(df['Final_Label'].value_counts())
# --- 2b. STRATIFIED 80/20 SPLIT (no oversampling, so no leakage risk) ---
idx_all = np.arange(len(df))
idx_train, idx_val = train_test_split(
    idx_all, test_size=0.2, random_state=SEED, stratify=df['Final_Label']
)
train_tensor = X_tensor[idx_train]
val_tensor = X_tensor[idx_val]
train_size, val_size = len(train_tensor), len(val_tensor)
print(f"Train: {train_size} | Val: {val_size}")
print("Val set label breakdown:")
print(df.loc[idx_val, 'Final_Label'].value_counts())
BATCH_SIZE = 64  # within the 32-128 range from your spec; adjust if you like
train_loader = DataLoader(train_tensor, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_tensor, batch_size=BATCH_SIZE)
# --- 3. VAE ARCHITECTURE ---
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder_net = nn.Sequential(
            nn.Linear(input_dim, 500), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(500, 100), nn.ReLU()
        )
        self.fc_mu = nn.Linear(100, latent_dim)
        self.fc_logvar = nn.Linear(100, latent_dim)
        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim, 100), nn.ReLU(),
            nn.Linear(100, 500), nn.ReLU(),
            nn.Linear(500, input_dim)
        )
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, x):
        h = self.encoder_net(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, min=-10, max=10)   # NEW - prevents exp() overflow / runaway KLD
        z = self.reparameterize(mu, logvar)
        return self.decoder_net(z), mu, logvar
def loss_function(recon_x, x, mu, logvar):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE, KLD
# --- 4. TRAINING LOOP WITH EARLY STOPPING ---
latent_dims = [10, 20, 40]
MAX_EPOCHS = 300
PATIENCE = 15
for l_dim in latent_dims:
    print(f"\n🚀 Training VAE | Latent Dim: {l_dim}")
    model = VAE(input_dim=X.shape[1], latent_dim=l_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0
    stopped_epoch = MAX_EPOCHS
    for epoch in range(MAX_EPOCHS):
        model.train()
        t_loss, t_mse, t_kld = 0, 0, 0
        for batch in train_loader:
            data = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(data)
            mse, kld = loss_function(recon, data, mu, logvar)
            loss = mse + kld
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)   # NEW - safety net against exploding gradients
            optimizer.step()
            t_loss += loss.item(); t_mse += mse.item(); t_kld += kld.item()
        model.eval()
        v_loss, v_mse, v_kld = 0, 0, 0
        with torch.no_grad():
            for v_batch in val_loader:
                v_data = v_batch.to(device)
                v_recon, v_mu, v_logvar = model(v_data)
                vmse, vkld = loss_function(v_recon, v_data, v_mu, v_logvar)
                v_loss += (vmse + vkld).item(); v_mse += vmse.item(); v_kld += vkld.item()
        avg_train, avg_train_mse, avg_train_kld = t_loss / train_size, t_mse / train_size, t_kld / train_size
        avg_val, avg_val_mse, avg_val_kld = v_loss / val_size, v_mse / val_size, v_kld / val_size
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if (epoch + 1) % 10 == 0 or epochs_no_improve == 0:
            print(f"Epoch {epoch+1:03d} | Train: {avg_train:.2f} (MSE {avg_train_mse:.2f} / KLD {avg_train_kld:.2f}) "
                  f"| Val: {avg_val:.2f} (MSE {avg_val_mse:.2f} / KLD {avg_val_kld:.2f}) | Best Val: {best_val_loss:.2f} | NoImprove: {epochs_no_improve}")
        if epochs_no_improve >= PATIENCE:
            print(f"⏹ Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
            stopped_epoch = epoch + 1
            break
    # restore best weights before saving/encoding
    model.load_state_dict(best_state)
    print(f"✅ Best epoch: {best_epoch} | Best Val Loss: {best_val_loss:.2f}")
    plt.figure(figsize=(8, 4))
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.axvline(best_epoch - 1, color='gray', linestyle='--', label=f'Best epoch ({best_epoch})')
    plt.title(f'VAE Z={l_dim} Training Profile (8,744 real patients)')
    plt.xlabel('Epoch'); plt.ylabel('Total Loss')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(base_dir, f'VAE_Z{l_dim}_Loss_Plot_8744.png'))
    plt.close()
    model_path = os.path.join(base_dir, f'VAE_Z{l_dim}_model_8744.pt')
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model weights saved to: {model_path}")
    # --- Latent Z for ALL 8,744 samples, using encoder mean (mu) ---
    model.eval()
    with torch.no_grad():
        _, z_mu, _ = model(X_tensor.to(device))
        z_df = pd.DataFrame(z_mu.cpu().numpy(), columns=[f'Z{i+1}' for i in range(l_dim)])
        out_df = pd.DataFrame({'sampleID': df['sampleID'].values})
        out_df = pd.concat([out_df, z_df], axis=1)
        out_df['Final_Label'] = df['Final_Label'].values  # sampleID, features, Final_Label
        z_out = os.path.join(base_dir, f'VAE_Latent_Z{l_dim}_8744.csv')
        out_df.to_csv(z_out, index=False)
        print(f"✅ Latent results saved to: {z_out} ({out_df['sampleID'].nunique()} unique IDs)")
# --- 5. PCA BASELINE (also on the real 8,744, for consistency) ---
print("\n📉 Running PCA baseline...")
pca = PCA(n_components=40, random_state=SEED)
pca_results = pca.fit_transform(X)
pca_df = pd.DataFrame(pca_results, columns=[f'PC{i+1}' for i in range(40)])
pca_out = pd.DataFrame({'sampleID': df['sampleID'].values})
pca_out = pd.concat([pca_out, pca_df], axis=1)
pca_out['Final_Label'] = df['Final_Label'].values
pca_out.to_csv(os.path.join(base_dir, 'PCA_Results_8744.csv'), index=False)
joblib.dump(pca, os.path.join(base_dir, 'PCA_40_model_8744.joblib'))
print("✅ PCA_Results_8744.csv and PCA_40_model_8744.joblib saved.")
print("\n🎉 Process Complete!")
