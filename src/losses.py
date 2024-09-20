import torch
import torch.nn.functional as F

def distance_based_ce(log_prob_preds, targets, protos, epsilon=1e-8):
    
    prob_preds = F.softmax(log_prob_preds, dim=1)    
    cdist = torch.cdist(targets, protos, p=2)
    cdist = cdist / cdist.sum(dim=1, keepdim=True)
    loss = torch.sum(cdist * prob_preds, dim=1)

    return loss.mean()

def distance_based_labeling(cdist,logits):
    cdist_probs = F.softmax(-cdist, dim=1)
    probs = F.softmax(logits, dim=1)
    
    return F.kl_div(probs, cdist_probs, reduction='batchmean')

def entropy_loss(logits):
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)  # Use log_softmax for numerical stability
    return - torch.sum(probs * log_probs, dim=1).mean()

def repulsion_loss(prototypes, margin=1.0):
    # Calculate the pairwise distances between prototypes
    dists = torch.cdist(prototypes, prototypes, p=2)
    
    # Mask the diagonal (distance between a prototype and itself)
    mask = torch.eye(dists.size(0), device=dists.device).bool()
    dists = dists.masked_fill(mask, float('inf'))
    
    # Apply a margin to ensure prototypes are at least 'margin' apart
    repulsive_term = torch.clamp(margin - dists, min=0)
    return repulsive_term.sum()


# Temp Controls the softness, lower is closer to hard Voronoi
def softmin_grads(prototypes, temperature = 0.1):
    proto_dist_list = torch.cdist(prototypes, prototypes)
    proto_negexp_distances = torch.nn.functional.softmin(proto_dist_list / temperature, dim=-1)
    proto_log_areas = (proto_negexp_distances * torch.eye(len(proto_negexp_distances)).to(prototypes.device)).sum(dim=0)
    proto_log_soft_areas = torch.log(proto_log_areas)
    return proto_log_soft_areas.sum()


def mindist_loss(train_y,protos):
    cdist_list = torch.cdist(train_y, protos, p=2)
    mindist, pos = torch.min(cdist_list, dim=1)
    return mindist.mean()

    