import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm # Import tqdm
import os
from models.poem2layout import Poem2LayoutGenerator
# from trainers.loss import layout_loss # Not used directly in this trainer anymore, as model handles it

class LayoutTrainer:
    def __init__(self, model: Poem2LayoutGenerator, train_loader: DataLoader, val_loader: DataLoader, config: dict):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=0.01
        )
        # Optional: Add scheduler
        # from transformers import get_linear_schedule_with_warmup
        # self.scheduler = get_linear_schedule_with_warmup(
        #     self.optimizer, # Fixed typo: optAimizer -> optimizer
        #     num_training_steps=len(train_loader) * config['training']['epochs'],
        #     num_warmup_steps=config['training']['warmup_steps']
        # )

        self.output_dir = config['training']['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_cls_loss = 0
        total_coord_loss = 0
        num_batches = 0

        # Use tqdm with default bar format, update postfix for loss
        pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch}")
        for batch in pbar:
            self.optimizer.zero_grad()

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            layout_seq = batch['layout_seq'].to(self.device)
            layout_mask = batch['layout_mask'].to(self.device)

            # --- CRITICAL SLICING ---
            model_input_seq = layout_seq[:, :-5] # [B, S-5]
            target_layout_seq = layout_seq[:, 5:] # [B, S-5]
            target_layout_mask = layout_mask[:, 5:] # [B, S-5]

            pred_cls, pred_coord = self.model(input_ids, attention_mask, model_input_seq) # Teacher forcing: input is [start, ..., end-5]

            # Calculate loss
            loss, cls_loss, coord_loss = self.model.get_loss(pred_cls, pred_coord, target_layout_seq, target_layout_mask)

            loss.backward()
            self.optimizer.step()
            # if self.scheduler:
            #     self.scheduler.step()

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_coord_loss += coord_loss.item()
            num_batches += 1

            # Update postfix (this shows Loss=xxx at the end of the progress bar)
            pbar.set_postfix({'Loss': loss.item()})

        avg_loss = total_loss / num_batches
        avg_cls_loss = total_cls_loss / num_batches
        avg_coord_loss = total_coord_loss / num_batches
        print(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}, Cls: {avg_cls_loss:.4f}, Coord: {avg_coord_loss:.4f}")

    @torch.no_grad()
    def validate_epoch(self, epoch):
        self.model.eval()
        total_loss = 0
        total_cls_loss = 0
        total_coord_loss = 0
        num_batches = 0

        # Use tqdm for validation as well, with postfix
        for batch in tqdm(self.val_loader, desc=f"Validating Epoch {epoch}"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            layout_seq = batch['layout_seq'].to(self.device)
            layout_mask = batch['layout_mask'].to(self.device)

            # --- CRITICAL SLICING ---
            model_input_seq = layout_seq[:, :-5] # [B, S-5]
            target_layout_seq = layout_seq[:, 5:] # [B, S-5]
            target_layout_mask = layout_mask[:, 5:] # [B, S-5]

            pred_cls, pred_coord = self.model(input_ids, attention_mask, model_input_seq)

            # Calculate loss
            loss, cls_loss, coord_loss = self.model.get_loss(pred_cls, pred_coord, target_layout_seq, target_layout_mask)

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_coord_loss += coord_loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_cls_loss = total_cls_loss / num_batches
        avg_coord_loss = total_coord_loss / num_batches
        print(f"Epoch {epoch} - Val Loss: {avg_loss:.4f}, Cls: {avg_cls_loss:.4f}, Coord: {avg_coord_loss:.4f}")
        return avg_loss

    def train(self):
        # We no longer track best_val_loss or save best model every epoch
        # Remove: best_val_loss = float('inf')

        for epoch in range(self.config['training']['epochs']):
            self.train_epoch(epoch)
            val_loss = self.validate_epoch(epoch)

            # Only save checkpoints every 10 epochs (or at the end)
            if (epoch + 1) % 10 == 0 or (epoch + 1) == self.config['training']['epochs']:
                checkpoint_path = os.path.join(self.output_dir, f"model_epoch_{epoch}_val_loss_{val_loss:.4f}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
            # else:
            #     # Remove the other save logic that was here previously
            #     pass
