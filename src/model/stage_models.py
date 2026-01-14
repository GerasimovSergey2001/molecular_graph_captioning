import torch 
import torch.nn as nn
from src.model.gine import GNN
from src.model.utils import freeze_model
from transformers import AutoModel



class MLPAdapter(nn.Module):
    """
    Adapter for transfering 2D-Graph data to LLM
    """
    def __init__(self, gin_hidden_dim, galactica_hidden_dim, num_query_tokens=32):
        super().__init__()

        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, gin_hidden_dim))
        
        # Cross-Attention Block
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=gin_hidden_dim, 
            num_heads=6, 
            batch_first=True
        )
        mlp = []
        for _ in range(3):
            mlp.append(nn.Linear(gin_hidden_dim*2, gin_hidden_dim * 2))
            mlp.append(nn.GELU())
        # MLP bridge
        self.mlp = nn.Sequential(
            nn.Linear(gin_hidden_dim, gin_hidden_dim * 2),
            nn.GELU(), 
            *mlp, 
            nn.Linear(gin_hidden_dim * 2, galactica_hidden_dim)
        )

    def forward(self, graph_node_embeddings, node_mask):
        """
        graph_node_embeddings: [batch_size, max_nodes, gin_dim]
        node_mask: [batch_size, max_nodes]
        """
        batch_size = graph_node_embeddings.size(0)
        
        queries = self.query_tokens.expand(batch_size, -1, -1)
        
        attn_out, _ = self.cross_attn(
            query=queries, 
            key=graph_node_embeddings, 
            value=graph_node_embeddings,
            key_padding_mask=~node_mask 
        )
        
        output = self.mlp(attn_out)
        return output

class MLPLogit(nn.Module):
    """
    For the Image-Text Matching (ITM) loss.
    """
    def __init__(self, emb_dim, hidden_dim=256):
        super().__init__()
        self.inp = nn.Linear(2 * emb_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.inp(x)
        x = self.act(x)
        return self.out(x)
    
class Stage1Wrapper(nn.Module):
    def __init__(self, gnn_hidden=300, bert_hidden=768, llm_hidden=2048, hidden_num=512, gnn_pretrained="checkpoints/graphcl_80.pth"):
        super().__init__()

        self.gnn = GNN(5, 300)
        state_dict = torch.load(gnn_pretrained, map_location='cpu')
        self.gnn.load_state_dict(state_dict)
        freeze_model(self.gnn)

        self.bert_projector = nn.Linear(bert_hidden, hidden_num)
        

        self.adapter_projector = nn.Linear(llm_hidden, hidden_num)

        self.adapter = MLPAdapter(gnn_hidden, llm_hidden)

        self.text_encoder = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
        freeze_model(self.text_encoder)

        self.itm_head = MLPLogit(hidden_num)

    def forward(self, graph, text_ids, attention_mask):
        g_embs, g_mask = self.gnn(graph)

        t_embs = self.text_encoder(
                    input_ids=text_ids,
            attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]

        g_features = self.adapter_projector(self.adapter(g_embs, g_mask))
        t_features = self.bert_projector(t_embs)
        return g_features, t_features
    
class Stage2Wrapper(nn.Module):
    def __init__(self, gnn_hidden=300, llm_hidden=2048, gnn_pretrained="checkpoints/graphcl_80.pth", adapter_pretrained= "mlp_adapter_stage1.pth", map_location='cpu'):
        super().__init__()

        self.gnn = GNN(5, 300)
        try:
            state_dict = torch.load(gnn_pretrained, map_location=map_location)
            self.gnn.load_state_dict(state_dict)
        except:
            print("No such file")
        
        self.adapter = MLPAdapter(gnn_hidden, llm_hidden)
        try:
            state_dict = torch.load(adapter_pretrained, map_location=map_location)
            self.adapter.load_state_dict(state_dict)
        except:
            print("No such file")

    def forward(self, graph):
        g_embs, g_mask = self.gnn(graph)
        return self.adapter(g_embs, g_mask)