import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleRelaxedContrastiveLoss(nn.Module):
    def __init__(self, alpha=0.7, lambda_neg=0.7, gamma_semi=0.7):
        super(DoubleRelaxedContrastiveLoss, self).__init__()
        self.alpha = alpha
        self.lambda_neg = lambda_neg
        self.gamma_semi = gamma_semi

    def forward(self, text_embeddings, image_embeddings, gt_img_ids, pseudo_gt_img_ids):
        device = text_embeddings.device

        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        image_embeddings = F.normalize(image_embeddings, p=2, dim=1)

        flattened_gt_img_ids = [item for sublist in gt_img_ids for item in sublist]
        id_to_index = {flattened_gt_img_ids[i]: i for i in range(len(flattened_gt_img_ids))}

        batch_size = text_embeddings.size(0)
        semi_positive_indices = torch.full((batch_size, len(pseudo_gt_img_ids)), -1, dtype=torch.long, device=device)

        semi_positive_count = 0

        for i in range(batch_size):
            valid_indices = [id_to_index.get(pseudo_gt_img_ids[j][i], -1) for j in range(len(pseudo_gt_img_ids))]
            semi_positive_indices[i, : len(valid_indices)] = torch.tensor(valid_indices, device=device)
            semi_positive_count += len([idx for idx in valid_indices if idx != -1])

        # print(f"semi_positive_count: {semi_positive_count}")
        all_sim = text_embeddings @ image_embeddings.T

        positive_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        semi_positive_mask = torch.zeros_like(all_sim, dtype=torch.bool, device=device)
        for i in range(batch_size):
            semi_indices = semi_positive_indices[i][semi_positive_indices[i] != -1]
            for idx in semi_indices:
                if idx != i:  # Exclude the true positive pair from semi-positive mask
                    semi_positive_mask[i, idx] = True

        l_pos = (all_sim[positive_mask] - 1).pow(2).sum()
        l_semi_pos = torch.clamp(self.alpha - all_sim[semi_positive_mask], min=0).pow(2).sum()
        negative_mask = ~(positive_mask | semi_positive_mask)
        l_neg = torch.relu(all_sim[negative_mask]).pow(2).sum()
        # l_neg = (all_sim[negative_mask]).pow(2).sum()

        loss = l_pos + self.lambda_neg * l_neg + self.gamma_semi * l_semi_pos
        return loss
