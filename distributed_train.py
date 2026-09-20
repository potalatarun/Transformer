# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.distributed as dist
# import torch.multiprocessing as mp
# from torch.utils.data import Dataset, DataLoader
# from torch.utils.data.distributed import DistributedSampler
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.cuda.amp import autocast, GradScaler

# # 1. SETUP & CLEANUP UTILITIES
# def setup_ddp(rank, world_size):
#     """Initializes the distributed environment."""
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12355' # Any free port
    
#     # Kaggle T4 GPUs use the "nccl" backend for maximum speed
#     dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
#     torch.cuda.set_device(rank)

# def cleanup_ddp():
#     """Cleans up the distributed environment after training finishes."""
#     dist.destroy_process_group()

# # 2. DUMMY DATASET (Replace with your text/tokenized dataset)
# class DummyDataset(Dataset):
#     def __init__(self, size=1000, seq_len=128):
#         self.x = torch.randint(0, 1000, (size, seq_len))
#         self.y = torch.randint(0, 2, (size,))
#     def __len__(self):
#         return len(self.x)
#     def __getitem__(self, idx):
#         return self.x[idx], self.y[idx]

# # 3. CORE TRAINING WORKER
# def train_worker(rank, world_size, epochs, batch_size):
#     """This function is executed independently on GPU 0 and GPU 1."""
#     setup_ddp(rank, world_size)
    
#     # Instantiate dataset
#     dataset = DummyDataset()
    
#     # CRITICAL: DistributedSampler ensures GPU 0 and GPU 1 get different data chunks
#     sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
#     # DataLoader: Note that batch_size is PER GPU (Global batch = batch_size * 2)
#     # pin_memory=True speeds up data transfer from CPU to GPU
#     dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, pin_memory=True)
    
#     # Initialize Model & Move to the assigned GPU rank
#     # Replace nn.Linear with your actual Transformer architecture
#     model = nn.Sequential(nn.Linear(128, 512), nn.ReLU(), nn.Linear(512, 2)).to(rank)
    
#     # Wrap model in DDP
#     model = DDP(model, device_ids=[rank])
    
#     # Setup Loss & Optimizer
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
#     # GradScaler is REQUIRED for mixed precision (FP16) on T4 Tensor Cores
#     scaler = GradScaler()
    
#     for epoch in range(epochs):
#         # CRITICAL: Tell the sampler which epoch it is to ensure proper shuffling
#         sampler.set_epoch(epoch)
        
#         model.train()
#         epoch_loss = 0.0
        
#         for x, y in dataloader:
#             x, y = x.to(rank, non_blocking=True), y.to(rank, non_blocking=True)
#             optimizer.zero_grad()
            
#             # Forward pass with mixed precision (AMP)
#             with autocast():
#                 outputs = model(x.float()) # casting dummy data to float
#                 loss = criterion(outputs, y)
            
#             # Backward pass & Optimization using the scaler
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
            
#             epoch_loss += loss.item()
            
#         # Logging: Only print from Rank 0 to avoid duplicate, messy print statements
#         if rank == 0:
#             avg_loss = epoch_loss / len(dataloader)
#             print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")
            
#     cleanup_ddp()

# # 4. MASTER LAUNCHER
# def main():
#     WORLD_SIZE = 2      # Kaggle's dual T4 setup = 2 GPUs
#     BATCH_SIZE = 16     # Per-GPU batch size (Effective total batch size = 32)
#     EPOCHS = 5
    
#     print(f"Starting distributed training across {WORLD_SIZE} GPUs...")
    
#     # Spawns 2 distinct processes running train_worker() concurrently
#     mp.spawn(
#         train_worker,
#         args=(WORLD_SIZE, EPOCHS, BATCH_SIZE),
#         nprocs=WORLD_SIZE,
#         join=True
#     )
#     print("Training Complete!")

# if __name__ == "__main__":
#     main()


# %%writefile "/kaggle/working/Transformer/train.py"


import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from Dataset import BilingualDataset, casual_mask
from model import build_transformer
from config import get_weights_file_path, get_config

from pathlib import Path
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

import warnings

#for distributed training
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
import torch.multiprocessing as mp

def setup_ddp(rank, world_size):
    """Initializes the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' # Any free port
    
    # Kaggle T4 GPUs use the "nccl" backend for maximum speed
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """Cleans up the distributed environment after training finishes."""
    dist.destroy_process_group()

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[EOS]", "[PAD]", "[SOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(config, rank, world_size):
    ds_raw = load_dataset("opus_books", f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    # ds_raw = load_dataset("HPLT/translate-en-te-v2.0-hplt_opus", f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    # Build tokenizers

    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # keep 90% for training and 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size   = len(ds_raw) - train_ds_size

    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids_len = len(tokenizer_src.encode(item['translation'][config['lang_src']]).ids)
        tgt_ids_len = len(tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids)

        max_len_src = max(max_len_src, src_ids_len)
        max_len_tgt = max(max_len_tgt, tgt_ids_len)

    print("Max length of source sentence:", max_len_src) 
    print("Max length of targets sentence:", max_len_tgt) 

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True)

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], sampler=sampler, pin_memory=True)
    return train_loader, val_loader, tokenizer_src, tokenizer_tgt
    

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model


def train_model(rank, world_size, config):
    setup_ddp(rank, world_size)
    # define device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"using the device: {device}")

    Path(config['model_folder']).mkdir(parents=True, exist_ok = True)
    train_dataloader, valid_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config, rank, world_size)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())
    model = model.to(device)
    model = DDP(model, device_ids=[rank])

    # Tensor Board

    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-4)
        
    initial_epoch = 0
    global_step = 0

    scaler = GradScaler()
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Pre-loading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)


    for epoch in range(initial_epoch, config['epochs']):
        # sampler.set_epoch(epoch)
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) #(B, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) #(B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)
        
            # run the tensor throught the transformer
            with autocast():
                encoder_output = model.module.encode(encoder_input, encoder_mask) #(B, seq_len, d_model)
                decoder_output = model.module.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) #(B, seq_len, d_model)
                proj_output = model.module.project(decoder_output) #(B, seq_len, tgt_vocab_size)
    
                label = batch['label'].to(device) #(B, seq_len)
                
                # (B, seq_len, tgt_vocab_size) -> (B * seq_len, tgt_vocab_size)
                loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))

            batch_iterator.set_postfix({"loss":f"{loss.item():6.3f}"})

            # log the loss to the tensor board
            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            # loss.backward()

            # Update the weights
            # optimizer.step()
            # optimizer.zero_grad()

            global_step += 1

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        # save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch':epoch,
            'model_state_dict':model.state_dict(),
            # 'optimizer_state_dict':optimizer.state_dict(),
            'global_step': global_step,
            }, model_filename)
    cleanup_ddp()

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    config = get_config()
    # train_model(config)
    WORLD_SIZE = 2      # Kaggle's dual T4 setup = 2 GPUs
    BATCH_SIZE = 16     # Per-GPU batch size (Effective total batch size = 32)
    EPOCHS = 5
    
    print(f"Starting distributed training across {WORLD_SIZE} GPUs...")
    
    # Spawns 2 distinct processes running train_worker() concurrently
    mp.spawn(
        train_model,
        args=(WORLD_SIZE, config),
        nprocs=WORLD_SIZE,
        join=True
    )
    print("Training Complete!")
