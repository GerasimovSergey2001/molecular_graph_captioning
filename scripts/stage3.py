import os
import torch
import torch.optim as optim
import numpy as np
from src.datasets.processed_dataset import PreprocessedGraphDataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.model.stage_models import Stage2Wrapper
from src.datasets.collate import TrainCollater2
from src.model.utils import get_scheduler, count_trainable_params

from sacrebleu import corpus_bleu
from bert_score import score as bertscore

from IPython.display import clear_output

def main():
    plot_dir = "./plots"
    os.makedirs(plot_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    accumulation_steps = 4

    model_name = "facebook/galactica-1.3b"

    galactica_tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens = {"additional_special_tokens": ["<mol>"]}
    galactica_tokenizer.add_special_tokens(special_tokens)
    galactica_tokenizer.pad_token = "<pad>"
    galactica_tokenizer.padding_side = "left"
    if galactica_tokenizer.eos_token is None:
        galactica_tokenizer.eos_token = "</s>"
        galactica_tokenizer.eos_token_id = 2


    train_dataset = PreprocessedGraphDataset(graph_path="./data/train_graphs.pkl")
    val_dataset = PreprocessedGraphDataset(graph_path="./data/validation_graphs.pkl") 

    num_workers, pin_memory = 4, True

    train_loader = DataLoader(train_dataset, 
                                batch_size=32, shuffle=True, 
                                num_workers=num_workers, pin_memory=pin_memory, 
                                collate_fn=TrainCollater2(galactica_tokenizer, 2048)
                                )
    val_loader = DataLoader(val_dataset, 
                                batch_size=32, shuffle=False, 
                                num_workers=num_workers, pin_memory=pin_memory, 
                                collate_fn=TrainCollater2(galactica_tokenizer, 2048)
                                )

    

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map=device,
        torch_dtype=torch.float16, 
        trust_remote_code=True
    )
    
    
    model.resize_token_embeddings(len(galactica_tokenizer))

    lora_config = {
    "base_model_name_or_path": None,
    "bias": "none",
    "fan_in_fan_out": False,
    "inference_mode": False,
    "init_lora_weights": True,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "target_modules": ["q_proj", "v_proj", "out_proj", "fc1", "fc2"],
    "peft_type": "LORA",
    "r": 16,
    "modules_to_save": None,
    "task_type": "CAUSAL_LM"
    }
    lora_config = LoraConfig(
    **lora_config
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    count_trainable_params(model, "Galactica with LoRA")

    
    stage3model = Stage2Wrapper(
        gnn_pretrained='./checkpoints/gnn_stage2.pth', 
        adapter_pretrained='./checkpoints/mlp_adapter_stage2.pth'
    ).to(device)

    warmup_steps = 1000
    init_lr = 1e-4
    min_lr = 1e-5
    weight_decay = 0.05
    warmup_lr = 1e-6
    retrieval_eval_epoch = 10
    num_epochs = 33
    max_steps = (len(train_loader) // accumulation_steps) * num_epochs

    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))

    optimizer = optim.AdamW([
        {
            "params": [p for p in model.parameters() if p.requires_grad],
            "lr": init_lr, 
            "weight_decay": weight_decay
        },
        {
            "params": [p for p in stage3model.parameters() if p.requires_grad],
            "lr": init_lr, 
            "weight_decay": weight_decay
        }
    ])
    scheduler = get_scheduler(optimizer, max_steps, warmup_steps, min_lr, start_factor=warmup_lr/init_lr)

    total_loss = []
    val_loss = []
    for epoch in tqdm(range(1, num_epochs+1), desc="Epoch"):
        clear_output(wait=True)
        stage3model.train()
        model.train()
        epoch_loss = []
        for i, batch in enumerate(train_loader):
            for k, v in batch.items():
                batch[k] = v.to(device)

            with torch.autocast(device_type=device, dtype=torch.float16):
            
                graph_embs= stage3model(batch['batch_graph'])
                
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
        plt.savefig(os.path.join(plot_dir, f"stage3_train_loss_epoch_{epoch}.png"))
        plt.show()
        plt.close()

        if epoch == 1:
            with torch.no_grad():
                gnn_grad_norm = torch.nn.utils.clip_grad_norm_(stage3model.parameters(), float('inf'))
                lora_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
                print(f"GNN/Adapter Grad Norm: {gnn_grad_norm:.4f}")
                print(f"LoRA Grad Norm: {lora_grad_norm:.4f}")

        if (epoch-1)%retrieval_eval_epoch==0:
            val_epoch_loss = []
            refs, preds = [], []
            stage3model.eval()
            model.eval()
            for i, batch in enumerate(val_loader):

                for k, v in batch.items():
                    batch[k] = v.to(device)

                with torch.no_grad():
                    with torch.autocast(device_type=device, dtype=torch.float16):
                        graph_embs= stage3model(batch['batch_graph'])
                    
                        embs = model.get_input_embeddings()(batch['input_ids'])

                        embs[batch['mol_mask']] = graph_embs.reshape(-1, graph_embs.shape[-1]).to(embs.dtype)

                        attention_mask = batch['attention_mask']

                        labels = batch['labels']

                        assert embs.shape[1] == attention_mask.shape[1] == labels.shape[1]

                        loss = model(
                            inputs_embeds=embs,       
                            attention_mask=attention_mask,
                            labels=labels,           
                            return_dict=True).loss
                        
                        if i < 5:
                    
                            prompt_embs = model.get_input_embeddings()(batch['prompt_ids'])
                            prompt_embs[batch['prompt_mol_mask']] = graph_embs.reshape(-1, graph_embs.shape[-1]).to(prompt_embs.dtype)
                            
                            generated_ids = model.generate(
                                inputs_embeds = prompt_embs,
                                attention_mask=batch["prompt_attention_mask"],
                                max_new_tokens=256,
                                do_sample=True,         
                                temperature=0.7,       
                                top_p=0.7,               
                                repetition_penalty=1.2,
                                pad_token_id=galactica_tokenizer.pad_token_id,
                                eos_token_id=galactica_tokenizer.eos_token_id,
                            )
                            preds.extend(
                                galactica_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                            )
                            refs.extend(batch['batch_graph'].description)
                    
                    val_epoch_loss.append(loss.detach().cpu().numpy())  

            val_loss.append(np.mean(val_epoch_loss))
            val_epochs = [i for i in range(1, epoch + 1) if (i-1) % retrieval_eval_epoch == 0]
            plt.figure(figsize=(12, 5))
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title("Val Set")
            plt.plot(val_epochs, val_loss)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"stage3_val_loss_epoch_{epoch}.png"))
            plt.show()
            plt.close()

            _, _, f1 = bertscore(
                        preds, 
                        refs, 
                        lang="en", 
                        device=device,
                        verbose=False
                    )
            f1 = f1.mean().item()

            bleu = corpus_bleu(preds, [[ref] for ref in refs]).score
            print("Bleu: ", bleu)
            print("F1: ", f1)
            print("Reference Example: ", refs[0])
            print("Generation Example: ", preds[0])

            model.save_pretrained(f"./checkpoints/galactica_lora_epoch{epoch}")
            torch.save(stage3model.gnn.state_dict(), f"./checkpoints/gnn_stage3_epoch{epoch}.pth")
            torch.save(stage3model.adapter.state_dict(), f"./checkpoints/mlp_adapter_stage3{epoch}.pth")


    merged_model = model.merge_and_unload()
    merged_model.save_pretrained("./checkpoints/galactica_full_final")
    galactica_tokenizer.save_pretrained("./checkpoints/galactica_full_final")

    torch.save(stage3model.gnn.state_dict(), "./checkpoints/gnn_final.pth")
    torch.save(stage3model.adapter.state_dict(), "./checkpoints/mlp_adapter_final.pth")

if __name__=='__main__':
    main()

