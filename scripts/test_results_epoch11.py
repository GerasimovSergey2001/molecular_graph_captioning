import os
import torch
import numpy as np
import pandas as pd
from src.datasets.processed_dataset import PreprocessedGraphDataset
from torch.utils.data import DataLoader

from tqdm import tqdm
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.model.stage_models import Stage2Wrapper
from src.datasets.collate import TestCollater

from pathlib import Path

def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dir = Path("./checkpoints")
    
    repo_id = "SergeiGerasimov/galactica_full_final"

    lora_weights_path = "galactica_lora_epoch11"
    model_name = "facebook/galactica-1.3b"

    tokenizer = AutoTokenizer.from_pretrained(repo_id)

    base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16, 
    device_map=device
    )
    base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, str(dir/lora_weights_path))
    
    stage3model = Stage2Wrapper(
        gnn_pretrained="./checkpoints/gnn_stage3_epoch11.pth", 
        adapter_pretrained="./checkpoints/mlp_adapter_stage311.pth"
        ).to(device)

    test_dataset = PreprocessedGraphDataset(graph_path="./data/test_graphs.pkl")

    num_workers, pin_memory = 4, True

    test_loader = DataLoader(test_dataset, 
                                batch_size=32, shuffle=False, 
                                num_workers=num_workers, pin_memory=pin_memory, 
                                collate_fn=TestCollater(tokenizer, 2048)
                                )
    
    model.eval()
    stage3model.eval()

    ids = []
    description = []

    for batch in tqdm(test_loader):

        for k, v in batch.items():
            batch[k] = v.to(device)

        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.float16):

                graph_embs= stage3model(batch['batch_graph'])
                                
                prompt_embs = model.get_input_embeddings()(batch['prompt_ids'])
                prompt_embs[batch['prompt_mol_mask']] = graph_embs.reshape(-1, graph_embs.shape[-1]).to(prompt_embs.dtype)
                                    

                prompt_embs = model.get_input_embeddings()(batch['prompt_ids'])
                prompt_embs[batch['prompt_mol_mask']] = graph_embs.reshape(-1, graph_embs.shape[-1]).to(prompt_embs.dtype)                    
                generated_ids = model.generate(
                                inputs_embeds = prompt_embs,
                                attention_mask=batch["prompt_attention_mask"],
                                max_new_tokens=256,
                                do_sample=True,         
                                temperature=0.7,       
                                top_p=0.5,               
                                repetition_penalty=1.2,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                                )
        description.extend(
            tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True
                )
            )
        ids.extend(batch['batch_graph'].to('cpu').id)

    submission = pd.DataFrame(
        {
            'ID': ids,
            'description':description
        }
    )
    submission.to_csv(f"./submissions/galactica_epoch11.csv", index=False)

if __name__=='__main__':
    main()