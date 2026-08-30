# # AccentEdge — Colab Training Pipeline
# ## FACodec Extraction + Indian Pretraining + Accent Conversion
# 
# **Repo**: https://github.com/yagami009/accentedge/tree/colab-training
# 
# This notebook runs the full AccentEdge training pipeline on Colab GPU.


!git clone -b colab-training https://github.com/yagami009/accentedge.git
%cd accentedge
!pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install -q transformers datasets huggingface_hub librosa soundfile praat-parselmouth scipy
!pip install -q -e .
print("Environment ready")

print("---")

from google.colab import drive
drive.mount('/content/drive')

# Set HF cache to Drive to persist across sessions
import os
os.environ['HF_HOME'] = '/content/drive/MyDrive/hf_cache'
os.environ['HF_DATASETS_CACHE'] = '/content/drive/MyDrive/hf_cache/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/content/drive/MyDrive/hf_cache/hub'
print("Drive mounted, HF cache set to Drive")

print("---")

from huggingface_hub import snapshot_download

model_dir = snapshot_download(
    repo_id='Plachta/FAcodec',
    local_dir='/content/facodec_model',
    allow_patterns=['*.pth', '*.yaml', '*.json', '*.py']
)
print(f"FACodec downloaded to: {model_dir}")

print("---")

from datasets import load_dataset
import os

datasets = {
    # Phase 0 — TGFP v2
    'cmu_arctic': ('MikhailT/cmu-arctic', None),
    'l2_arctic': ('KoelLabs/L2Arctic', None),
    # Phase 1 — Benchmark + Training
    'libritts_100': ('openslr/librispeech_asr', 'clean'),  # subset
    'fleurs_en_us': ('google/fleurs', 'en_us'),
    # Phase 2 — Indian pretraining
    'indicvoices_hindi': ('ai4bharat/IndicVoices', 'hindi'),
    'indicvoices_tamil': ('ai4bharat/IndicVoices', 'tamil'),
    'indicvoices_telugu': ('ai4bharat/IndicVoices', 'telugu'),
}

for name, (repo, config) in datasets.items():
    print(f"Downloading {name}...")
    ds = load_dataset(repo, config, trust_remote_code=False)
    ds.save_to_disk(f'/content/drive/MyDrive/accentedge_data/{name}')
    print(f"  {name}: {len(ds['train'])} samples saved")

print("\nAll datasets downloaded")

print("---")

import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm

# Load FACodec model
sys.path.insert(0, '/content/facodec_model')
from facodec import FAcodec

model = FAcodec(
    n_codebooks=8,
    nyquist=24000,
    hidden_dim=512,
    n_heads=8,
    n_layers=8,
       codebook_size=1024,
).cuda()
model.eval()

def extract_latents(audio_path, model):
    wav, sr = torchaudio.load(audio_path)
    if sr != 24000:
        wav = torchaudio.functional.resample(wav, sr, 24000)
    wav = wav.unsqueeze(0).cuda()
    with torch.no_grad():
        z_q = model.encode(wav)  # [B, 8, T] codebook indices
    return z_q.cpu()

# Extract latents for all datasets
data_root = Path('/content/drive/MyDrive/accentedge_data')
cache_root = Path('/content/drive/MyDrive/accentedge_latents')
cache_root.mkdir(exist_ok=True)

for ds_name in ['cmu_arctic', 'l2_arctic', 'indicvoices_hindi']:
    ds_path = data_root / ds_name
    if not ds_path.exists():
        continue
    
    # Find all wav files
    wav_files = list(ds_path.rglob('*.wav'))
    print(f"\nExtracting latents for {ds_name} ({len(wav_files)} files)...")
    
    for wav_path in tqdm(wav_files):
        rel = wav_path.relative_to(ds_path)
        cache_path = cache_root / ds_name / rel.with_suffix('.pt')
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        if cache_path.exists():
            continue
        
        try:
            latents = extract_latents(str(wav_path), model)
            torch.save(latents, cache_path)
        except Exception as e:
            print(f"  Error: {wav_path}: {e}")

print("\nLatent extraction complete")

print("---")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Load extracted latents
latent_root = Path('/content/drive/MyDrive/accentedge_latents')

class LatentDataset(Dataset):
    def __init__(self, ds_names):
        self.files = []
        for name in ds_names:
            self.files.extend(list((latent_root / name).rglob('*.pt')))
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        latents = torch.load(self.files[idx])
        return latents.squeeze(0).T  # [T, 8]

# Simple pretraining: predict next codebook token
class PretrainModel(nn.Module):
    def __init__(self, vocab_size=1024, d_model=256, n_heads=4, n_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, 500, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=512)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.head = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # x: [B, T, 8] codebook indices
        B, T, C = x.shape
        x = self.embed(x.long())
        x = x.sum(dim=2) + self.pos[:, :T]
        x = self.transformer(x)
        return self.head(x)

# Train
model = PretrainModel().cuda()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

ds = LatentDataset(['indicvoices_hindi', 'indicvoices_tamil', 'indicvoices_telugu'])
loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2)

for epoch in range(5):
    total_loss = 0
    for batch in tqdm(loader):
        batch = batch.cuda()  # [B, T, 8]
        logits = model(batch[:, :-1])  # predict next
        loss = criterion(logits.reshape(-1, 1024), batch[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}: loss={total_loss/len(loader):.4f}")
    torch.save(model.state_dict(), f'/content/drive/MyDrive/accentedge_checkpoints/pretrain_epoch{epoch}.pt')

print("Pretraining complete")

print("---")

# Uses CMU ARCTIC (US) as source + L2-ARCTIC (Indian) as target

# TODO: Implement accent conversion fine-tuning
# - Load paired source/target latents
# - Train diffusion model to convert z_c1 from US→Indian accent
# - Save checkpoints to Drive

print("Accent conversion fine-tuning — implementation pending")

print("---")
