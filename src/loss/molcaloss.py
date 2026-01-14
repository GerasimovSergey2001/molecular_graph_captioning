import torch
import torch.nn as nn
import torch.nn.functional as F

class MolCALoss(nn.Module):
    """
    Class which realizes loss for the first stage from MolCA ()
    """
    def __init__(self, init_temp=0.07, learnable_temp=True):
        super().__init__()
        if learnable_temp:
            self.temp = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / init_temp)))
        else:
            self.register_buffer("temp", torch.log(torch.tensor(1 / init_temp)))

    def get_contrastive_loss(self, g_embeds, t_embeds):
        """
        g_embeds: [batch_size, emb_dim]
        t_embeds: [batch_size, emb_dim] (CLS tokens)
        """
        g_embeds = F.normalize(g_embeds, p=2, dim=-1)
        t_embeds = F.normalize(t_embeds, p=2, dim=-1)

        t_scale = torch.exp(self.temp)
        logits_per_graph = t_scale * (g_embeds @ t_embeds.t())
        logits_per_text = logits_per_graph.t()

        labels = torch.arange(g_embeds.size(0), device=g_embeds.device)

        loss_g = F.cross_entropy(logits_per_graph, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)

        return (loss_g + loss_t) / 2, logits_per_graph

    def get_matching_loss(self, g_embeds, t_embeds, sim_matrix, itm_head):
        batch_size = g_embeds.size(0)
        
        labels = torch.cat([
            torch.ones(batch_size, device=g_embeds.device), 
            torch.zeros(2 * batch_size, device=g_embeds.device)
        ])

        with torch.no_grad():
            sim_mask = torch.eye(batch_size, device=g_embeds.device).bool()
            sim_matrix_masked = sim_matrix.clone().masked_fill(sim_mask, -1e9)

            idx_t_neg = sim_matrix_masked.argmax(dim=1)
            idx_g_neg = sim_matrix_masked.argmax(dim=0)

        g_all = torch.cat([g_embeds, g_embeds, g_embeds[idx_g_neg]], dim=0)
        t_all = torch.cat([t_embeds, t_embeds[idx_t_neg], t_embeds], dim=0)

        logits = itm_head(torch.cat([g_all, t_all], dim=-1)).squeeze(-1)
        
        return F.binary_cross_entropy_with_logits(logits, labels)

    def forward(self, graph_feats, text_feats, itm_head):
        """
        graph_feats: [batch_size, num_nodes, emb_dim]
        text_feats:  [batch_size, emb_dim] (CLS токен)
        itm_head: nn.Module - head for clasification

        Returns: sum of losses
        """
        g_embeds_global = graph_feats.mean(dim=1) 
        
        itc_loss, sim_matrix = self.get_contrastive_loss(g_embeds_global, text_feats)
        
        itm_loss = self.get_matching_loss(g_embeds_global, text_feats, sim_matrix, itm_head)
        
        return itc_loss + itm_loss
    