import numpy
import numpy as np
import torch
import math
from module.utils import *
from cross_modal_recall import *
def gen_label(labels):
    num = len(labels)
    gt = numpy.zeros(shape=(num,num))
    for i, label in enumerate(labels):
        for k in range(num):
            if labels[k] == label:
                gt[i,k] = 1
    return gt

def gen_label_from_text_sim(x):
    x = x / x.norm(dim=-1, keepdim=True)
    return x @ x.t()

def get_m_theta(cos_theta, m=4):
    cos_m_theta = mlambda[m](cos_theta)
    temp = cos_theta.clone().detach()
    theta = torch.acos(temp.clamp(-1.+1e-6, 1.-1e-6))
    k = (theta*m / math.pi).floor()
    sign = -2 * torch.remainder(k, 2) + 1  # (-1)**k
    phi_theta = sign * cos_m_theta - 2. * k
    return phi_theta
    # d_theta = phi_theta - cos_theta
    # return d_theta + x


def create_logits(x1, x2, logit_scale, exp=True):
    x1 = x1 / x1.norm(dim=-1, p=2, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, p=2, keepdim=True)
    if exp:
        scale = logit_scale.exp()
    else:
        scale = logit_scale
    # cosine similarity as logits
    logits_per_x1 = scale * x1 @ x2.float().t()
    logits_per_x2 = logits_per_x1.t()
    # if orth:
    #     ortho_constraint = OrthoRegularizer(lambda_ortho=0.5, mode='full')
    #     loss_orth = ortho_constraint(x1,x2)
    #     logits_per_x1 += loss_orth
    #     logits_per_x2 += loss_orth

    # shape = [global_batch_size, global_batch_size]
    return logits_per_x1, logits_per_x2

def create_logits_causal(x1, x2, logit_scale, zest, head, tracker, global_step,exp=True, causal=False):
    x1 = x1 / x1.norm(dim=-1, p=2, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, p=2, keepdim=True)
    # if exp:
    #     scale = logit_scale.exp()
    # else:
    #     scale = logit_scale
    # cosine similarity as logits
    # logits_per_x1 = scale * x1 @ x2.float().t()
    # logits_per_x2 = logits_per_x1.t()
    # zest = ConfounderEstimator(d_in=768, num_z=4, use_text=True).cuda()
    # head = CausalAlignHead(d=768, num_z=4, temp=0.07).cuda()
    q, logits = zest(x1, x2)
    # print(f'logits std() {logits.std()}')
    p_batch = q.mean(dim=0)
    #----------entropy
    pz_given_k = q.detach()
    # pz_given_k = logits.detach()
    # p = pz_given_k.clamp_min(1e-8)
    # loss_cond_ent = -(p * p.log()).sum(dim=1).mean()
    stats_entropy = tracker.update(global_step, pz_given_k, logits)
    #----------end
    # head.update_prior(q.detach())
    head.update_prior(p_batch)
    # head.update_prior(q) # SCR ab study

    loss, stats = head.backdoor_clip_loss(x1, x2)
    # loss = loss + 0.05 * loss_cond_ent

    # return loss, {"qz": q.detach().cpu(), "pz": stats["pz"], "losses_z": stats["losses_z"]}, [logits_per_x1, logits_per_x2]
    # return loss, {"qz": q.detach().cpu(), "pz": stats["pz"], "losses_z": stats["losses_z"]}
    return loss, {"losses_z": stats["losses_z"]}
    # if orth:
    #     ortho_constraint = OrthoRegularizer(lambda_ortho=0.5, mode='full')
    #     loss_orth = ortho_constraint(x1,x2)
    #     logits_per_x1 += loss_orth
    #     logits_per_x2 += loss_orth

    # shape = [global_batch_size, global_batch_size]
    # return logits_per_x1, logits_per_x2

def create_sim_matrix(x1, x2, alpha=1):
    x1 = x1 / x1.norm(dim=-1, keepdim=True).float()
    x2 = x2 / x2.norm(dim=-1, keepdim=True).float()
    x1x1 = alpha * x1 @ x1.t()
    x1x2 = alpha * x1 @ x2.t()
    x2x2 = alpha * x2 @ x2.t()
    return x1x1, x1x2, x2x2
    

def get_acc(x1, x2, unseen_label, label, cross_model=False):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)
    logits = x1 @ x2.t() # 128, 5
    pred = torch.argmax(logits, dim=1)
    unseen_label = torch.tensor(unseen_label).cuda()
    pred = torch.index_select(unseen_label,0,pred)
    acc = pred.eq(label.view_as(pred)).float().mean()
    if cross_model:
        compute_cross_modal_recall_at_k(x1,x2,unseen_label,label,[1,5])
    return acc, pred

def get_acc_gzsl(x1, x2, label, unseen_label, seen_label):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)
    logits = x1 @ x2.t() # 128, 5
    pred = torch.argmax(logits, dim=1)
    unseen_label = torch.tensor(unseen_label).cuda()
    pred = torch.index_select(unseen_label,0,pred)
    acc = pred.eq(label.view_as(pred)).float().mean()
    unseen_label = unseen_label.cpu().numpy()
    gzsl_acc = compute_gzsl_accuracy(label, pred, seen_label, unseen_label)
    return acc, pred, gzsl_acc

def compute_gzsl_accuracy(y_true, y_pred, seen_classes, unseen_classes):
    """
    Compute GZSL metrics: seen accuracy, unseen accuracy, harmonic mean.

    Args:
        y_true (array-like): ground-truth labels
        y_pred (array-like): predicted labels
        seen_classes (list or set): labels of seen classes
        unseen_classes (list or set): labels of unseen classes

    Returns:
        dict: {'Acc_s': ..., 'Acc_u': ..., 'H': ...}
    """
    # y_true = np.array(y_true)
    y_true = y_true.cpu().numpy().squeeze()
    # y_pred = np.array(y_pred)
    y_pred = y_pred.cpu().numpy()
    # Mask for seen and unseen samples
    seen_mask = np.isin(y_true, seen_classes).squeeze()
    unseen_mask = np.isin(y_true, unseen_classes).squeeze()

    # Avoid division by zero
    t_s_t = y_true[seen_mask]
    t_s_p = y_pred[seen_mask]
    t_c_s = t_s_t == t_s_p
    av_s = np.mean(t_c_s)
    acc_s = np.mean(y_pred[seen_mask] == y_true[seen_mask]) if np.any(seen_mask) else 0.0
    acc_u = np.mean(y_pred[unseen_mask] == y_true[unseen_mask]) if np.any(unseen_mask) else 0.0

    # Harmonic mean
    if acc_s + acc_u > 0:
        H = 2 * acc_s * acc_u / (acc_s + acc_u)
    else:
        H = 0.0

    return {'Acc_s': acc_s, 'Acc_u': acc_u, 'H': H}

def get_acc_v2(x1, x2, unseen_label, label, paired_language,cross_model):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)
    logits = x1.float() @ x2.float().t() # 128, 5
    pred = torch.argmax(logits, dim=1)
    # unseen_len = len(unseen_label)
    unseen_label = torch.tensor(unseen_label)
    if cross_model:
        results = compute_cross_modal_recall_at_k(x2, x1, unseen_label,label,[1,5])
        print(results)
    
    old_pred = pred
    ent = softmax_entropy(logits)
    
    # unseen_len = len(unseen_label)
    # for i in range(unseen_len):
    #     class_support_set = x1[pred == i]
    #     class_logit = logits[pred == i]
    #     class_ent = softmax_entropy(class_logit)
    #     _, indices = torch.topk(class_ent, 5)
    #     z = torch.mean(class_support_set[indices], dim=-1)
    #     z_list.append(z)
    unseen_label = torch.tensor(unseen_label).cuda()
    pred = torch.index_select(unseen_label, 0, pred)
    acc = pred.eq(label.view_as(pred)).float().mean()
    return acc, pred, old_pred, ent, x1
# @torch.no_grad()
def calibrated_stacking(logits: torch.Tensor, seen_mask: torch.Tensor, gamma: float) -> torch.Tensor:
    # out = logits.clone()
    t = logits[:,seen_mask]
    t_u = logits[:,~seen_mask]
    logits[:,seen_mask] -= gamma
    t_1 = logits[:,seen_mask]
    t_u_1 = logits[:,~seen_mask]
    return logits

def get_acc_v2_gzsl(x1, x2, test_label, label, unseen_label, seen_label):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)
    logits = x1.float() @ x2.float().t()  # 128, 5
    # logits = logits/temp
    y_true = label.cpu().numpy()
    seen_mask = np.isin(range(60), seen_label).squeeze()
    # print(f'max-min:{logits.max(),logits.min()}')
    logits = calibrated_stacking(logits,seen_mask, 0.055)
    # logits = logits/2
    pred = torch.argmax(logits, dim=1)
    # test_len = len(test_label)

    old_pred = pred
    ent = softmax_entropy(logits)

    # unseen_len = len(unseen_label)
    # for i in range(unseen_len):
    #     class_support_set = x1[pred == i]
    #     class_logit = logits[pred == i]
    #     class_ent = softmax_entropy(class_logit)
    #     _, indices = torch.topk(class_ent, 5)
    #     z = torch.mean(class_support_set[indices], dim=-1)
    #     z_list.append(z)
    test_label = torch.tensor(test_label).cuda()
    pred = torch.index_select(test_label, 0, pred)
    acc = pred.eq(label.view_as(pred)).float().mean()
    gzsl_acc = compute_gzsl_accuracy(label, pred, seen_label, unseen_label)
    return acc, pred, old_pred, ent, x1, gzsl_acc

def get_acc_v3(x1, x2, unseen_label, label):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)
    logits = x1 @ x2.t() # 128, 5
    pred = torch.argmax(logits, dim=1)
    ent = softmax_entropy(logits)
    unseen_label = torch.tensor(unseen_label).cuda()
    pred = torch.index_select(unseen_label,0,pred)
    acc = pred.eq(label.view_as(pred)).float().mean()
    return acc, pred, ent

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

# def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
#     """Entropy of softmax distribution from logits."""
#     return -(x.softmax(1) * math.log2(math.e) * x.log_softmax(1)).sum(1)
