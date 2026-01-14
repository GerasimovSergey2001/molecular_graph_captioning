import os
import torch
import numpy as np
from src.datasets.processed_dataset import PreprocessedGraphDataset
from torch.utils.data import DataLoader
from torch.amp import GradScaler

import matplotlib.pyplot as plt

from tqdm import tqdm
from transformers import AutoTokenizer,  AutoModelForCausalLM

from src.model.stage_models import Stage2Wrapper
from src.datasets.collate import TrainCollater2
from src.model.utils import get_scheduler, freeze_model, count_trainable_params

from sacrebleu import corpus_bleu
from bert_score import score as bertscore

from IPython.display import clear_output

def main():
    plot_dir = "./plots"
    os.makedirs(plot_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    accumulation_steps = 8

    model_name = "facebook/galactica-1.3b"

    galactica_tokenizer = AutoTokenizer.from_pretrained(model_name)


    train_dataset = PreprocessedGraphDataset(graph_path="./data/train_graphs.pkl")
    val_dataset = PreprocessedGraphDataset(graph_path="./data/validation_graphs.pkl") 

    num_workers, pin_memory = 4, True

    train_loader = DataLoader(train_dataset, 
                                batch_size=16, shuffle=True, 
                                num_workers=num_workers, pin_memory=pin_memory, 
                                collate_fn=TrainCollater2(galactica_tokenizer, 2048)
                                )
    val_loader = DataLoader(val_dataset, 
                                batch_size=16, shuffle=False, 
                                num_workers=num_workers, pin_memory=pin_memory, 
                                collate_fn=TrainCollater2(galactica_tokenizer, 2048)
                                )



    # 2. Добавляем специальный токен для молекулы, если его нет в словаре
    special_tokens = {"additional_special_tokens": ["<mol>"]}
    galactica_tokenizer.add_special_tokens(special_tokens)
    galactica_tokenizer.pad_token = "<pad>"
    galactica_tokenizer.padding_side = "left"

    if galactica_tokenizer.eos_token is None:
        galactica_tokenizer.eos_token = "</s>"
        galactica_tokenizer.eos_token_id = 2

    stage2model = Stage2Wrapper(
                        gnn_pretrained='./checkpoints/graphcl_80.pth', 
                        adapter_pretrained="./checkpoints/mlp_adapter_stage1.pth"
                        ).to(device)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map=device,
        torch_dtype=torch.float16, 
        trust_remote_code=True
    )
    model.gradient_checkpointing_enable()
    model.resize_token_embeddings(len(galactica_tokenizer))
    model.eval()
    freeze_model(model)

    count_trainable_params(model, "Galactica")

    warmup_steps = 1000
    init_lr = 1e-4
    min_lr = 1e-5
    weight_decay = 0.05
    warmup_lr = 1e-6
    retrieval_eval_epoch = 4
    num_epochs = 10
    max_steps = (len(train_loader) // accumulation_steps) * num_epochs

    optimizer = torch.optim.AdamW(stage2model.parameters(), lr=init_lr, weight_decay=weight_decay)
    scheduler = get_scheduler(optimizer, max_steps, warmup_steps, min_lr, start_factor=warmup_lr/init_lr)
    scaler = GradScaler('cuda', enabled=(device == 'cuda'))
    total_loss = []
    val_loss = []
    for epoch in tqdm(range(1, num_epochs+1)):
        clear_output(wait=True)
        stage2model.train()
        epoch_loss = []
        for i, batch in enumerate(train_loader):
            for k, v in batch.items():
                batch[k] = v.to(device)

            with torch.autocast(device_type=device, dtype=torch.float16):
            
                graph_embs= stage2model(batch['batch_graph'])
                
                embs = model.get_input_embeddings()(batch['input_ids']).clone()

                embs[batch['mol_mask']] = graph_embs.reshape(-1, graph_embs.shape[-1]).to(embs.dtype)

                attention_mask = batch['attention_mask']

                labels = batch['labels']

                assert embs.shape[1] == attention_mask.shape[1] == labels.shape[1]

                loss = model(
                    inputs_embeds=embs,       
                    attention_mask=attention_mask,
                    labels=labels,              
                    return_dict=True
                ).loss / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            epoch_loss.append(loss.detach().cpu().numpy()*accumulation_steps)
        

        total_loss.append(np.mean(epoch_loss))

        plt.figure(figsize=(12, 5))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title("Train Set")
        plt.plot(np.arange(len(total_loss)), total_loss)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"stage2_train_loss_epoch_{epoch}.png"))
        plt.show()
        plt.close()

        if (epoch-1)%retrieval_eval_epoch==0:
            
            torch.save(stage2model.gnn.state_dict(), f"./checkpoints/gnn_stage2_epoch{epoch}.pth")
            torch.save(stage2model.adapter.state_dict(), f"./checkpoints/mlp_adapter_stage2_epoch{epoch}.pth")

            val_epoch_loss = []
            refs, preds = [], []
            stage2model.eval()
            for batch in val_loader:

                for k, v in batch.items():
                    batch[k] = v.to(device)

                with torch.no_grad():
                    with torch.autocast(device_type=device, dtype=torch.float16):
                    
                        graph_embs= stage2model(batch['batch_graph'])
                    
                        embs = model.get_input_embeddings()(batch['input_ids'])

                        embs[batch['mol_mask']] = graph_embs.reshape(-1, graph_embs.shape[-1]).to(embs.dtype)

                        attention_mask = batch['attention_mask']

                        labels = batch['labels']

                        assert embs.shape[1] == attention_mask.shape[1] == labels.shape[1]

                        loss = model(
                            inputs_embeds=embs,       
                            attention_mask=attention_mask,
                            labels=labels,           
                            return_dict=True).loss / accumulation_steps
                        
                        # prompt_embs = model.get_input_embeddings()(batch['prompt_ids'])
                        # prompt_embs[batch['prompt_mol_mask']] = graph_embs.reshape(-1, graph_embs.shape[-1]).to(prompt_embs.dtype)
                        
                        # generated_ids = model.generate(
                        #     inputs_embeds = prompt_embs,
                        #     attention_mask=batch["prompt_attention_mask"],
                        #     max_new_tokens=128,
                        #     pad_token_id=galactica_tokenizer.pad_token_id,
                        #     eos_token_id=galactica_tokenizer.eos_token_id,
                        #     do_sample=False,
                        #     repetition_penalty=1.2
                        # )
                    # preds.extend(
                    #     galactica_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                    # )
                    # refs.extend(batch['batch_graph'].description)
                    
                    val_epoch_loss.append(loss.detach().cpu().numpy()*accumulation_steps)  

            val_loss.append(np.mean(val_epoch_loss))
            val_epochs = [i for i in range(1, epoch + 1) if (i-1) % retrieval_eval_epoch == 0]
            plt.figure(figsize=(12, 5))
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title("Val Set")
            plt.plot(val_epochs, val_loss)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"stage2_val_loss_epoch_{epoch}.png"))
            plt.show()
            plt.close()

    torch.save(stage2model.gnn.state_dict(), "./checkpoints/gnn_stage2.pth")
    torch.save(stage2model.adapter.state_dict(), "./checkpoints/mlp_adapter_stage2.pth")

if __name__=="__main__":
    main()