import torch
import torch.nn.functional as F

def distance_based_ce(logits, cdist, areas, temp_logits=1.0, temp_cdist=1.0, epsilon=1e-8):
    adjusted_logits = logits + torch.log(areas).unsqueeze(0).detach()  # Unsqueeze to match batch dimension
    
    adjusted_logits = adjusted_logits - adjusted_logits.max(dim=1, keepdim=True).values  # Stabilize logits before softmax
    log_probs = F.softmax(adjusted_logits / temp_logits, dim=1)

    cdist = cdist / temp_cdist
    cdist = cdist / cdist.sum(dim=1, keepdim=True)

    loss = -torch.sum(cdist * log_probs, dim=1)

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
