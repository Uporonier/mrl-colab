# ✅ MRL-style embedding fusion training for LLaMA-1B and LLaMA-3B

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os
import torch.nn.functional as F


# 配置
model_A_path = '/data/zhaoxinpeng-slurm/workspace/EmbeddingMerage/EmbeddingMerage/models/llama-1b'
model_B_path = '/data/zhaoxinpeng-slurm/workspace/EmbeddingMerage/EmbeddingMerage/models/llama-3b'
max_length = 512
batch_size = 4
device_ids = [1, 3]
save_dir = "./checkpoints_mrl"
os.makedirs(save_dir, exist_ok=True)

# 设备设置
device_2 = f'cuda:{device_ids[0]}'
device_3 = f'cuda:{device_ids[1]}'

# 加载模型
print("Loading models...")
tokenizer = AutoTokenizer.from_pretrained(model_B_path)
tokenizer.pad_token = tokenizer.eos_token
model_A = AutoModelForCausalLM.from_pretrained(model_A_path, torch_dtype=torch.float16).to(device_2)
model_B = AutoModelForCausalLM.from_pretrained(model_B_path, torch_dtype=torch.float16).to(device_3)
for p in model_A.parameters(): p.requires_grad = False
for p in model_B.parameters(): p.requires_grad = False
vocab_size = model_A.lm_head.out_features

# 嵌套维度
nested_dims = [512, 1024, 2048]

# MRL 投影器
class MRLProjector(nn.Module):
    def __init__(self, in_dim=2048 + 3072, nested_dims=[512, 1024, 2048]):
        super().__init__()
        self.nested_dims = nested_dims
        self.projectors = nn.ModuleDict({
            str(dim): nn.Sequential(
                nn.Linear(in_dim, 4096),
                nn.GELU(),
                nn.LayerNorm(4096),
                nn.Linear(4096, dim)
            ) for dim in nested_dims
        })
        self.lm_head = nn.Linear(2048, vocab_size)  # 共享分类头

    def forward(self, z_cat):
        losses = {}
        logits_all = {}
        for dim in self.nested_dims:
            z_proj = self.projectors[str(dim)](z_cat)  # [bsz, seq, dim]
            z_pad = (F.pad(z_proj, (0, 2048 - dim)) if dim < 2048 else z_proj)
            logits = self.lm_head(z_pad)
            logits_all[dim] = logits
        return logits_all

projector = MRLProjector(nested_dims=nested_dims).to(device_2)
projector = nn.DataParallel(projector, device_ids=device_ids)

optimizer = torch.optim.AdamW(projector.parameters(), lr=5e-6, weight_decay=0.01)
scaler = torch.cuda.amp.GradScaler()

# 数据集
dataset = load_dataset("yahma/alpaca-cleaned", split="train[:5000]")

def collate_fn(examples):
    texts = [
        f"{ex['instruction']}\n{ex['input']}\n{ex['output']}" if ex['input'].strip() != ""
        else f"{ex['instruction']}\n{ex['output']}"
        for ex in examples
    ]
    return tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# 训练循环
for epoch in range(50):
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in progress_bar:
        try:
            input_ids = batch["input_ids"].to(device_2)
            attention_mask = batch["attention_mask"].to(device_2)
            labels = input_ids.detach().clone()

            with torch.no_grad():
                emb_A = model_A.model.embed_tokens(input_ids)  # [bsz, seq, 2048]
                emb_B = model_B.model.embed_tokens(input_ids.to(device_3)).to(device_2)  # [bsz, seq, 3072]

            with torch.cuda.amp.autocast():
                z_cat = torch.cat([emb_A, emb_B], dim=-1)  # [bsz, seq, 5120]
                logits_dict = projector(z_cat)  # 多个维度输出

                loss = 0
                for dim in nested_dims:
                    logits = logits_dict[dim]
                    loss += F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
                loss /= len(nested_dims)  # 平均多个嵌套维度

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(projector.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            progress_bar.set_postfix(loss=loss.item())

        except Exception as e:
            print(f"Error: {str(e)}")
            continue

    if epoch % 5 == 0:
        torch.save(projector.module.state_dict(), os.path.join(save_dir, f"mrl_projector_epoch{epoch}.pt"))
