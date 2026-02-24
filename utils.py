import numpy as np
from scipy.stats import sem
import scipy.stats as stats
def compute_performance(end_task_acc_arr):
    """
    Given test accuracy results from multiple runs saved in end_task_acc_arr,
    compute the average accuracy, forgetting, and task accuracies as well as their confidence intervals.

    :param end_task_acc_arr:       (list) List of lists
    :param task_ids:                (list or tuple) Task ids to keep track of
    :return:                        (avg_end_acc, forgetting, avg_acc_task)
    """
    n_run, n_tasks = end_task_acc_arr.shape[:2]
    t_coef = stats.t.ppf((1+0.95) / 2, n_run-1)     # t coefficient used to compute 95% CIs: mean +- t *

    # compute average test accuracy and CI
    end_acc = end_task_acc_arr[:, -1, :]                         # shape: (num_run, num_task)
    avg_acc_per_run = np.mean(end_acc, axis=1)      # mean of end task accuracies per run
    avg_end_acc = (np.mean(avg_acc_per_run), t_coef * sem(avg_acc_per_run))

    # compute forgetting
    best_acc = np.max(end_task_acc_arr, axis=1)
    final_forgets = best_acc - end_acc
    avg_fgt = np.mean(final_forgets, axis=1)
    avg_end_fgt = (np.mean(avg_fgt), t_coef * sem(avg_fgt))

    # compute ACC
    acc_per_run = np.mean((np.sum(np.tril(end_task_acc_arr), axis=2) /
                           (np.arange(n_tasks) + 1)), axis=1)
    avg_acc = (np.mean(acc_per_run), t_coef * sem(acc_per_run))


    # compute BWT+
    bwt_per_run = (np.sum(np.tril(end_task_acc_arr, -1), axis=(1,2)) -
                  np.sum(np.diagonal(end_task_acc_arr, axis1=1, axis2=2) *
                         (np.arange(n_tasks, 0, -1) - 1), axis=1)) / (n_tasks * (n_tasks - 1) / 2)
    bwtp_per_run = np.maximum(bwt_per_run, 0)
    avg_bwtp = (np.mean(bwtp_per_run), t_coef * sem(bwtp_per_run))

    # compute FWT
    fwt_per_run = np.sum(np.triu(end_task_acc_arr, 1), axis=(1,2)) / (n_tasks * (n_tasks - 1) / 2)
    avg_fwt = (np.mean(fwt_per_run), t_coef * sem(fwt_per_run))
    return avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt

def sample_similar_for_proto(images, labels, buffer, per_sample_for_p,class_x_list, class_y_list, class_id_map):
    images_for_proto = []
    labels_for_proto = []
    unique_label = []
    for c in labels:
        if c.item() not in unique_label:
            unique_label.append(c.item())

    for choose_cls in unique_label:
        from_current_idx = (choose_cls ==  labels)
        select_images = images[from_current_idx]
        len_select_images = len(select_images)

        if choose_cls not in class_id_map:
            images_for_proto.append(select_images)
            labels_for_proto.extend([choose_cls for _ in range(select_images.size(0))])
            continue
        class_idx = class_id_map[choose_cls]
        len_choose_class_buf = len(class_x_list[class_idx])
        sample_num = min(per_sample_for_p, len_choose_class_buf)
        diff_num =per_sample_for_p - sample_num 

        if diff_num == 0:
            class_idx = class_id_map[choose_cls]
            class_sample_idx = torch.from_numpy(np.random.choice(class_x_list[class_idx].tolist(), sample_num)).long()
            class_x = buffer.x[class_sample_idx]
            images_for_proto.append(class_x)
            labels_for_proto.extend([choose_cls for _ in range(sample_num)])
        else:
            if select_images.size(0) < diff_num:
                diff_num = select_images.size(0)
            class_idx = class_id_map[choose_cls]
            class_sample_idx = torch.from_numpy(np.random.choice(class_x_list[class_idx].tolist(), sample_num)).long()
            class_x = buffer.x[class_sample_idx]
            images_for_proto.append(torch.cat((select_images[:diff_num], class_x), dim=0))
            labels_for_proto.extend([choose_cls for _ in range(diff_num+sample_num)])
        
    images_for_proto = torch.cat(images_for_proto)
    labels_for_proto = torch.tensor(labels_for_proto).long()
    return images_for_proto, labels_for_proto
import os,sys
import numpy as np
from copy import deepcopy
import torch
from tqdm import tqdm

########################################################################################################################

def print_model_report(model):
    print('-'*100)
    print(model)
    print('Dimensions =',end=' ')
    count=0
    for p in model.parameters():
        print(p.size(),end=' ')
        count+=np.prod(p.size())
    print()
    print('Num parameters = %s'%(human_format(count)))
    print('-'*100)
    return count

def human_format(num):
    magnitude=0
    while abs(num)>=1000:
        magnitude+=1
        num/=1000.0
    return '%.1f%s'%(num,['','K','M','G','T','P'][magnitude])

def print_optimizer_config(optim):
    if optim is None:
        print(optim)
    else:
        print(optim,'=',end=' ')
        opt=optim.param_groups[0]
        for n in opt.keys():
            if not n.startswith('param'):
                print(n+':',opt[n],end=', ')
        print()
    return

########################################################################################################################

def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return

########################################################################################################################

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

########################################################################################################################

def compute_mean_std_dataset(dataset):
    # dataset already put ToTensor
    mean=0
    std=0
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for image, _ in loader:
        mean+=image.mean(3).mean(2)
    mean /= len(dataset)

    mean_expanded=mean.view(mean.size(0),mean.size(1),1,1).expand_as(image)
    for image, _ in loader:
        std+=(image-mean_expanded).pow(2).sum(3).sum(2)

    std=(std/(len(dataset)*image.size(2)*image.size(3)-1)).sqrt()

    return mean, std

########################################################################################################################

def fisher_matrix_diag(t,x,y,model,criterion,sbatch=20):
    # Init
    fisher={}
    for n,p in model.named_parameters():
        fisher[n]=0*p.data
    # Compute
    model.train()
    for i in tqdm(range(0,x.size(0),sbatch),desc='Fisher diagonal',ncols=100,ascii=True):
        b=torch.LongTensor(np.arange(i,np.min([i+sbatch,x.size(0)]))).cuda()
        images=torch.autograd.Variable(x[b],volatile=False)
        target=torch.autograd.Variable(y[b],volatile=False)
        # Forward and backward
        model.zero_grad()
        outputs=model.forward(images)
        loss=criterion(t,outputs[t],target)
        loss.backward()
        # Get gradients
        for n,p in model.named_parameters():
            if p.grad is not None:
                fisher[n]+=sbatch*p.grad.data.pow(2)
    # Mean
    for n,_ in model.named_parameters():
        fisher[n]=fisher[n]/x.size(0)
        fisher[n]=torch.autograd.Variable(fisher[n],requires_grad=False)
    return fisher

########################################################################################################################

def cross_entropy(outputs,targets,exp=1,size_average=True,eps=1e-5):
    out=torch.nn.functional.softmax(outputs)
    tar=torch.nn.functional.softmax(targets)
    if exp!=1:
        out=out.pow(exp)
        out=out/out.sum(1).view(-1,1).expand_as(out)
        tar=tar.pow(exp)
        tar=tar/tar.sum(1).view(-1,1).expand_as(tar)
    out=out+eps/out.size(1)
    out=out/out.sum(1).view(-1,1).expand_as(out)
    ce=-(tar*out.log()).sum(1)
    if size_average:
        ce=ce.mean()
    return ce

########################################################################################################################

def set_req_grad(layer,req_grad):
    if hasattr(layer,'weight'):
        layer.weight.requires_grad=req_grad
    if hasattr(layer,'bias'):
        layer.bias.requires_grad=req_grad
    return

########################################################################################################################

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False
########################################################################################################################
import numpy as np
from scipy.stats import sem
import scipy.stats as stats
def compute_performance(end_task_acc_arr):
    """
    Given test accuracy results from multiple runs saved in end_task_acc_arr,
    compute the average accuracy, forgetting, and task accuracies as well as their confidence intervals.

    :param end_task_acc_arr:       (list) List of lists
    :param task_ids:                (list or tuple) Task ids to keep track of
    :return:                        (avg_end_acc, forgetting, avg_acc_task)
    """
    n_run, n_tasks = end_task_acc_arr.shape[:2]
    t_coef = stats.t.ppf((1+0.95) / 2, n_run-1)     # t coefficient used to compute 95% CIs: mean +- t *

    # compute average test accuracy and CI
    end_acc = end_task_acc_arr[:, -1, :]                         # shape: (num_run, num_task)
    avg_acc_per_run = np.mean(end_acc, axis=1)      # mean of end task accuracies per run
    avg_end_acc = (np.mean(avg_acc_per_run), t_coef * sem(avg_acc_per_run))

    # compute forgetting
    best_acc = np.max(end_task_acc_arr, axis=1)
    final_forgets = best_acc - end_acc
    avg_fgt = np.mean(final_forgets, axis=1)
    avg_end_fgt = (np.mean(avg_fgt), t_coef * sem(avg_fgt))

    # compute ACC
    acc_per_run = np.mean((np.sum(np.tril(end_task_acc_arr), axis=2) /
                           (np.arange(n_tasks) + 1)), axis=1)
    avg_acc = (np.mean(acc_per_run), t_coef * sem(acc_per_run))


    # compute BWT+
    bwt_per_run = (np.sum(np.tril(end_task_acc_arr, -1), axis=(1,2)) -
                  np.sum(np.diagonal(end_task_acc_arr, axis1=1, axis2=2) *
                         (np.arange(n_tasks, 0, -1) - 1), axis=1)) / (n_tasks * (n_tasks - 1) / 2)
    bwtp_per_run = np.maximum(bwt_per_run, 0)
    avg_bwtp = (np.mean(bwtp_per_run), t_coef * sem(bwtp_per_run))

    # compute FWT
    fwt_per_run = np.sum(np.triu(end_task_acc_arr, 1), axis=(1,2)) / (n_tasks * (n_tasks - 1) / 2)
    avg_fwt = (np.mean(fwt_per_run), t_coef * sem(fwt_per_run))
    return avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt

from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import torch
import numpy as np
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

def compute_clustering_metrics(model, test_loader, device="cuda"):
    """
    计算轮廓系数 (Silhouette Score) 以及类内和类间距离。

    参数:
        model (torch.nn.Module): 训练好的模型
        test_loader (DataLoader): 测试集数据加载器
        device (str): 计算设备 ("cuda" 或 "cpu")

    返回:
        silhouette_avg (float): 轮廓系数
        intra_class_dist (float): 平均类内距离
        inter_class_dist (float): 平均类间距离
    """
    model.to(device)
    model.eval()

    features = []
    labels = []

    # 提取特征
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)  # 提取特征
            features.extend(output.cpu().numpy())  # 确保是(N, feature_dim) 形状
            labels.extend(target.cpu().numpy())

    features = np.array(features)
    labels = np.array(labels)

    # 确保数据量足够计算轮廓系数
    if len(np.unique(labels)) < 2 or len(features) < 2:
        silhouette_avg = 0.0
    else:
        silhouette_avg = silhouette_score(features, labels)

    # 计算类内和类间距离
    class_features = {}  # 存储每个类别的特征
    for feature, label in zip(features, labels):
        if label not in class_features:
            class_features[label] = []
        class_features[label].append(feature)

    # 计算类内距离 (Intra-class Distance)
    intra_distances = []
    for label, feats in class_features.items():
        feats = np.array(feats)
        if len(feats) > 1:
            dist_matrix = cdist(feats, feats, metric="euclidean")
            triu_indices = np.triu_indices_from(dist_matrix, k=1)  # 只取上三角部分
            if triu_indices[0].size > 0:  # 确保有多个样本
                avg_intra_dist = np.mean(dist_matrix[triu_indices])
                intra_distances.append(avg_intra_dist)
    intra_class_dist = np.mean(intra_distances) if intra_distances else 0.0

    # 计算类间距离 (Inter-class Distance)
    class_means = {label: np.mean(feats, axis=0) for label, feats in class_features.items()}
    inter_distances = []
    class_labels = list(class_means.keys())

    if len(class_labels) > 1:
        for i in range(len(class_labels)):
            for j in range(i + 1, len(class_labels)):
                dist = np.linalg.norm(class_means[class_labels[i]] - class_means[class_labels[j]])
                inter_distances.append(dist)
        inter_class_dist = np.mean(inter_distances) if inter_distances else 0.0
    else:
        inter_class_dist = 0.0  # 只有一个类别时，类间距离设为0

    print(f"轮廓系数 (Silhouette Score): {silhouette_avg:.4f}")
    print(f"平均类内距离 (Intra-class Distance): {intra_class_dist:.4f}")
    print(f"平均类间距离 (Inter-class Distance): {inter_class_dist:.4f}")
    return silhouette_avg, intra_class_dist, inter_class_dist
def print_normalized_class_norms(model=None, classifier_weights=None):
    # 获取分类器层的权重 (假设为最后一层线性层)
    if classifier_weights is None:
        classifier_weights = model.fc.weight.data  # [num_classes, num_features]
    
    # 计算每个类别的范数
    norms = []
    for class_idx in range(classifier_weights.size(0)):
        norm = torch.norm(classifier_weights[class_idx], p=2).item()  # L2范数
        norms.append(norm)
    
    # 计算所有范数的总和
    total_norm = sum(norms)
    
    # 对每个范数进行归一化
    normalized_norms = [norm / total_norm for norm in norms]

    # 打印归一化后的范数
    for class_idx, normalized_norm in enumerate(normalized_norms):
        print(f"Class {class_idx} normalized norm: {normalized_norm}")

    max_norm = max(norms)
    normalized_norms = [norm / max_norm for norm in norms]
    for class_idx, normalized_norm in enumerate(normalized_norms):
        print(f"Class {class_idx} normalized norm: {normalized_norm}")