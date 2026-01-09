import os
import torch
import numpy as np
from src.datasets.processed_dataset import PreprocessedGraphDataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from tqdm import tqdm
from transformers import AutoTokenizer

from src.model.stage_models import Stage1Wrapper
from src.loss.molcaloss import MolCALoss
from src.datasets.collate import TrainCollater
from src.model.utils import get_scheduler

from IPython.display import clear_output

def main():
    plot_dir = "./plots"
    os.makedirs(plot_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

    train_dataset = PreprocessedGraphDataset(graph_path="./data/train_graphs.pkl") #train_graphs.pkl
    val_dataset = PreprocessedGraphDataset(graph_path="./data/validation_graphs.pkl") 

    num_workers, pin_memory = 4, True

    train_loader = DataLoader(train_dataset, 
                            batch_size=64, shuffle=True, 
                            num_workers=num_workers, pin_memory=pin_memory, 
                            collate_fn=TrainCollater(tokenizer, 256)
                            )
    val_loader = DataLoader(val_dataset, 
                            batch_size=64, shuffle=False, 
                            num_workers=num_workers, pin_memory=pin_memory, 
                            collate_fn=TrainCollater(tokenizer, 256)
                            )

    warmup_steps = 1000
    init_lr = 1e-4
    min_lr = 1e-5
    weight_decay = 0.05
    warmup_lr = 1e-6
    retrieval_eval_epoch = 10
    num_epochs = 50
    max_steps = len(train_loader)*num_epochs

    stage1model = Stage1Wrapper(gnn_pretrained="./checkpoints/graphcl_80.pth").to(device)
    optimizer = torch.optim.AdamW(stage1model.parameters(), lr=init_lr, weight_decay=weight_decay)
    scheduler = get_scheduler(optimizer, max_steps, warmup_steps, min_lr, start_factor=warmup_lr/init_lr)
    criterion = MolCALoss(learnable_temp=False)

    total_loss = []
    val_loss = []

    for epoch in tqdm(range(1, num_epochs+1)):
        clear_output(wait=True)
        epoch_loss = []
        stage1model.train()
        for graphs, text_ids, attention_mask in train_loader:
            graphs, text_ids, attention_mask  = graphs.to(device), text_ids.to(device), attention_mask.to(device)
            graph_feats, t_feats = stage1model(graphs, text_ids, attention_mask)
            loss = criterion(graph_feats, t_feats, stage1model.itm_head)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss.append(loss.detach().cpu().numpy())

        total_loss.append(np.mean(epoch_loss))

        plt.figure(figsize=(12, 5))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title("Train Set")
        plt.plot(np.arange(len(total_loss)), total_loss)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"stage1_train_loss_epoch_{epoch}.png"))
        plt.show()
        
        if epoch%retrieval_eval_epoch==0:
            stage1model.eval()
            val_epoch_loss = []
            for graphs, text_ids, attention_mask in val_loader:

                graphs, text_ids, attention_mask  = graphs.to(device), text_ids.to(device), attention_mask.to(device)
                graph_feats, t_feats = stage1model(graphs, text_ids, attention_mask)
                loss = criterion(graph_feats, t_feats, stage1model.itm_head)

                val_epoch_loss.append(loss.detach().cpu().numpy())
            val_loss.append(np.mean(val_epoch_loss))

        
            val_epochs = [i for i in range(1, epoch + 1) if i % retrieval_eval_epoch == 0]
            plt.figure(figsize=(12, 5))
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title("Val Set")
            plt.plot(val_epochs, val_loss)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"stage1_val_loss_epoch_{epoch}.png"))
            plt.show()


    torch.save(stage1model.adapter.state_dict(), "./checkpoints/mlp_adapter_stage1.pth")

if __name__=='__main__':
    main()