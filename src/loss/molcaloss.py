import torch
import torch.nn as nn
import torch.nn.functional as F

class MolCALoss(nn.Module):
    def __init__(self, init_temp=0.07, learnable_temp=True):
        super().__init__()
        # Обучаемая температура позволяет модели самой настроить "резкость" контраста
        if learnable_temp:
            self.temp = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / init_temp)))
        else:
            self.register_buffer("temp", torch.log(torch.tensor(1 / init_temp)))

    def get_contrastive_loss(self, g_embeds, t_embeds):
        """
        ITC Loss: Выравнивание векторов графа и текста.
        """
        # 1. Нормализация (L2)
        g_embeds = F.normalize(g_embeds, p=2, dim=-1)
        t_embeds = F.normalize(t_embeds, p=2, dim=-1)

        # 2. Матрица сходства (Косинусное расстояние)
        # exp(temp) здесь работает как коэффициент масштабирования
        t_scale = torch.exp(self.temp)
        logits_per_graph = t_scale * g_embeds @ t_embeds.T
        logits_per_text = logits_per_graph.T

        # 3. Таргеты (диагональ - правильные пары)
        labels = torch.arange(g_embeds.size(0), device=g_embeds.device)

        # Двусторонняя кросс-энтропия
        loss_g = F.cross_entropy(logits_per_graph, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)

        return (loss_g + loss_t) / 2, logits_per_graph

    def get_matching_loss(self, g_embeds, t_embeds, sim_matrix, itm_head):
        """
        ITM Loss: Бинарная классификация пар с учетом сложных негативов.
        """
        batch_size = g_embeds.size(0)
        labels = torch.cat([torch.ones(batch_size), torch.zeros(2 * batch_size)]).to(g_embeds.device)

        with torch.no_grad():
            # Маскируем диагональ, чтобы не выбрать правильную пару как негатив
            sim_mask = torch.eye(batch_size, device=g_embeds.device).bool()
            sim_matrix_masked = sim_matrix.clone().masked_fill(sim_mask, -1e9)

            # Майним сложные негативы: 
            # 1. Самый похожий текст для каждого графа
            idx_t_neg = sim_matrix_masked.argmax(dim=1)
            # 2. Самый похожий граф для каждого текста
            idx_g_neg = sim_matrix_masked.argmax(dim=0)

        # Формируем пары для классификатора
        # Позитивы: (G_i, T_i)
        # Негативы: (G_i, T_hard_neg) и (G_hard_neg, T_i)
        g_all = torch.cat([g_embeds, g_embeds, g_embeds[idx_g_neg]], dim=0)
        t_all = torch.cat([t_embeds, t_embeds[idx_t_neg], t_embeds], dim=0)

        # matching_head должен принимать конкатенацию векторов
        logits = itm_head(torch.cat([g_all, t_all], dim=-1)).squeeze(-1)
        
        return F.binary_cross_entropy_with_logits(logits, labels.long().float())

    def forward(self, graph_feats, text_feats, itm_head):
        # ITC (Contrastive)
        itc_loss, sim_matrix = self.get_contrastive_loss(graph_feats, text_feats)
        
        # ITM (Matching)
        itm_loss = self.get_matching_loss(graph_feats, text_feats, sim_matrix, itm_head)
        
        return itc_loss + itm_loss