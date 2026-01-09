import torch
from torch_geometric.data import Batch
from rdkit import Chem



class TrainCollater(object):
    def __init__(self, tokenizer, text_max_len):
        self.tokenizer = tokenizer
        self.text_max_len = text_max_len

    def process_edge_attr(self, edge_attr):
        bond_type = edge_attr[:, 0]
        new_bond_attr = torch.zeros_like(bond_type)

        new_bond_attr[bond_type == 1] = 0  
        new_bond_attr[bond_type == 2] = 1  
        new_bond_attr[bond_type == 3] = 2 
        new_bond_attr[bond_type > 4] = 3   

        stereo = edge_attr[:, 1]
        new_stereo_attr = torch.zeros_like(stereo)
        new_stereo_attr[stereo == 2] = 1  
        new_stereo_attr[stereo == 3] = 2 

        return torch.stack([new_bond_attr, new_stereo_attr], dim=1)

    def __call__(self, batch):
        
        batch_graph = Batch.from_data_list(batch)
        
        batch_graph.x = batch_graph.x[:, :2].clone()
        
        batch_graph.edge_attr = self.process_edge_attr(batch_graph.edge_attr)
               
        text_list = [data.description for data in batch]
        text_batch = self.tokenizer(
            text_list, 
            padding='max_length', 
            truncation=True, 
            max_length=self.text_max_len, 
            return_tensors='pt'
        )
        
        return batch_graph, text_batch.input_ids, text_batch.attention_mask


class TrainCollater2(object):
    def __init__(self, tokenizer, text_max_len):
        self.tokenizer = tokenizer
        self.text_max_len = text_max_len

        # Карты маппинга OGB (входящие данные)
        self.x_map = {
            'atomic_num': list(range(0, 119)),
            'chirality': [
                'CHI_UNSPECIFIED','CHI_TETRAHEDRAL_CW','CHI_TETRAHEDRAL_CCW','CHI_OTHER',
                'CHI_TETRAHEDRAL','CHI_ALLENE','CHI_SQUAREPLANAR','CHI_TRIGONALBIPYRAMIDAL',
                'CHI_OCTAHEDRAL',
            ],
            'formal_charge': list(range(-5, 7)),
            'num_hs': list(range(0, 9)),
            'is_aromatic': [False, True],
        }

        self.e_map = {
            'bond_type': [
                'UNSPECIFIED','SINGLE','DOUBLE','TRIPLE','QUADRUPLE','QUINTUPLE','HEXTUPLE',
                'ONEANDAHALF','TWOANDAHALF','THREEANDAHALF','FOURANDAHALF','FIVEANDAHALF',
                'AROMATIC','IONIC','HYDROGEN','THREECENTER','DATIVEONE','DATIVE','DATIVEL',
                'DATIVER','OTHER','ZERO',
            ],
            'stereo': [
                'STEREONONE','STEREOANY','STEREOZ','STEREOE','STEREOCIS','STEREOTRANS',
            ],
        }
        self.BOND_LOOKUP = {s: getattr(Chem.rdchem.BondType, s.upper()) for s in self.e_map['bond_type'] if s != 'UNSPECIFIED'}

    def process_edge_attr(self, edge_attr):
        bond_type = edge_attr[:, 0]
        new_bond_attr = torch.zeros_like(bond_type)

        new_bond_attr[bond_type == 1] = 0  
        new_bond_attr[bond_type == 2] = 1  
        new_bond_attr[bond_type == 3] = 2 
        new_bond_attr[bond_type > 4] = 3   

        stereo = edge_attr[:, 1]
        new_stereo_attr = torch.zeros_like(stereo)
        new_stereo_attr[stereo == 2] = 1  
        new_stereo_attr[stereo == 3] = 2 

        return torch.stack([new_bond_attr, new_stereo_attr], dim=1)


    def get_mol_and_smiles(self, graph):
        """
        Только восстановление SMILES из графа OGB
        """
        mol = Chem.RWMol()
        node_features = graph.x.cpu().numpy()
        
        # 1. Добавляем атомы
        for feat in node_features:
            atom = Chem.Atom(int(self.x_map['atomic_num'][feat[0]]))
            atom.SetChiralTag(getattr(Chem.rdchem.ChiralType, self.x_map['chirality'][feat[1]]))
            atom.SetFormalCharge(int(self.x_map['formal_charge'][feat[3]]))
            atom.SetNumExplicitHs(int(self.x_map['num_hs'][feat[4]]))
            atom.SetIsAromatic(bool(self.x_map['is_aromatic'][feat[7]]))
            mol.AddAtom(atom)

        adj = graph.edge_index.cpu().numpy()
        edge_attr = graph.edge_attr.cpu().numpy()
        
        # Сет для предотвращения дублирования связей
        added_bonds = set()
        
        # 2. Добавляем связи
        for i in range(adj.shape[1]):
            u, v = int(adj[0, i]), int(adj[1, i])
            
            if u >= v: continue # Добавляем связь только один раз (u < v)
                
            bond_key = (u, v)
            bt_str = self.e_map['bond_type'][edge_attr[i, 0]]
            
            if bt_str != 'UNSPECIFIED':
                mol.AddBond(u, v, self.BOND_LOOKUP.get(bt_str, Chem.rdchem.BondType.SINGLE))

        final_mol = mol.GetMol()
        
        # 3. Финализация и генерация SMILES
        try:
            # SanitizeMol обязателен, чтобы MolToSmiles выдал корректную строку
            Chem.SanitizeMol(final_mol) 
            smiles = Chem.MolToSmiles(final_mol, isomericSmiles=True, canonical=True)
        except:
            # Если молекула химически невозможна (ошибки валентности и т.д.)
            smiles = "" 
                
        return smiles

    def genereate_prompt(self, smiles_text):
        mol_placeholders = "<mol> "*32
        prompt = f"Given the SMILES of the molecule [START_I_SMILES] {smiles_text} [END_I_SMILES] " \
        + f"{mol_placeholders}. "  + "Description: "
        # prompt = f"[START_I_SMILES] {smiles_text} [END_I_SMILES] \nDescription: "
        return prompt

    def __call__(self, batch):
        all_smiles = [self.get_mol_and_smiles(g) for g in batch]
        batch_graph = Batch.from_data_list(batch)
        
        prompts_list = [self.genereate_prompt(s) for s in all_smiles]
        desc_list = [data.description + self.tokenizer.eos_token for data in batch]

        # 1. Токенизируем промпты отдельно для генерации
        # Используем padding=True и выравнивание слева
        self.tokenizer.padding_side = 'left'
        prompt_features = self.tokenizer(
            prompts_list, 
            padding=True, 
            truncation=True,
            max_length=self.text_max_len if self.text_max_len else 1024,
            return_tensors='pt', 
            add_special_tokens=False
            )
        prompt_ids = prompt_features.input_ids
        prompt_attention_mask = prompt_features.attention_mask
        
        # Возвращаем настройку обратно (обычно для обучения удобнее right)
        self.tokenizer.padding_side = 'right'

        # 2. Логика для обучения (склеивание)
        prompt_tokens = self.tokenizer(prompts_list, add_special_tokens=False)
        desc_tokens = self.tokenizer(desc_list, add_special_tokens=False)

        all_input_ids = []
        all_labels = []
        
        for p_ids, d_ids in zip(prompt_tokens.input_ids, desc_tokens.input_ids):
            combined_ids = p_ids + d_ids
            combined_labels = [-100] * len(p_ids) + d_ids
            
            all_input_ids.append(torch.tensor(combined_ids[:self.text_max_len]))
            all_labels.append(torch.tensor(combined_labels[:self.text_max_len]))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            all_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            all_labels, batch_first=True, padding_value=-100
        )
        
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        # 3. Маски для молекулярных токенов
        mol_token_id = self.tokenizer.convert_tokens_to_ids("<mol>")
        train_mol_mask = (input_ids == mol_token_id)
        prompt_mol_mask = (prompt_ids == mol_token_id) # Маска специально для генерации
        
        # Обработка графа
        batch_graph.x = batch_graph.x[:, :2].clone()
        batch_graph.edge_attr = self.process_edge_attr(batch_graph.edge_attr)

        return {
            "batch_graph": batch_graph,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "mol_mask": train_mol_mask,
            "prompt_ids": prompt_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "prompt_mol_mask": prompt_mol_mask
        }