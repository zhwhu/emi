import sys, argparse
import numpy as np
import torch
from torch.nn.functional import relu, avg_pool2d
from buffer import Buffer
# import utils
import datetime
from torch.nn.functional import relu
import torch
import torch.nn as nn
import torch.nn.functional as F
from InfoNCE import tao as TL
#from InfoNCE import classifier as C
from InfoNCE.utils import normalize
from InfoNCE.contrastive_learning import get_similarity_matrix,Supervised_NT_xent_pre,Supervised_NT_xent_n,Supervised_NT_xent_uni,Supervised_NT_xent_simb,Supervised_NT_xent_simn,Supervised_NT_xent_proto,Supervised_NT_xent_pp
import torch.optim.lr_scheduler as lr_scheduler
#from CSL.shedular import GradualWarmupScheduler
import torch
from apex import amp
import torchvision.transforms as transforms
import  torchvision
from torch.cuda.amp import GradScaler,autocast
import torchvision.transforms as transforms
import  torchvision
from asf3 import AdaptiveSimMixFeedback
# from APF import AdaptivePrototypicalFeedback
from OPE import OPELoss
from tqdm import tqdm
from utils import compute_performance, sample_similar_for_proto, compute_clustering_metrics, print_normalized_class_norms
# Arguments
class Logger(object):
    def __init__(self, logFile ="Default.log"):
        self.terminal = sys.stdout
        self.log = open(logFile,'a')
 
    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass
parser = argparse.ArgumentParser()
parser.add_argument('--logfile', type=str, default="runs/log.txt", help='(default=%(default)s)')
parser.add_argument('--seed', type=int, default=0, help='(default=%(default)d)')
parser.add_argument('--experiment', default='cifar-100', type=str, required=False, help='(default=%(default)s)')
parser.add_argument('--approach', default='OWM', type=str, required=False, help='(default=%(default)s)')
parser.add_argument('--nepochs', default=25, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--lr', default=0.0005, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--parameter', type=str, default='', help='(default=%(default)s)')
parser.add_argument('--dataset', type=str, default='cifar', help='(default=%(default)s)')
parser.add_argument('--input_size', type=str, default=[3, 32, 32], help='(default=%(default)s)')
parser.add_argument('--buffer_size', type=int, default=2000, help='(default=%(default)s)')
parser.add_argument('--gen', type=str, default=True, help='(default=%(default)s)')
parser.add_argument('--n_classes', type=int, default=512, help='(default=%(default)s)')
parser.add_argument('--buffer_batch_size', type=int, default=64, help='(default=%(default)s)')
parser.add_argument('--run_nums', type=int, default=64, help='(default=%(default)s)')
parser.add_argument('--is_rot', type=int, default=1, help='(default=%(default)s)')
parser.add_argument('--batch', type=int, default=10, help='(default=%(default)s)')
args = parser.parse_args()
import os

def cal_prototype(z1, z2, y, current_task_id, class_per_task):
        start_i = 0
        end_i = (current_task_id + 1) * class_per_task
        dim = z1.shape[1]
        current_classes_mean_z1 = torch.zeros((end_i, dim), device=z1.device)
        current_classes_mean_z2 = torch.zeros((end_i, dim), device=z1.device)
        proto_label = torch.zeros((end_i), device=z1.device)
        for i in range(start_i, end_i):
            indices = (y == i)
            if not any(indices):
                proto_label[i] = -1
                continue
            t_z1 = z1[indices]
            t_z2 = z2[indices]
            proto_label[i] = i
            mean_z1 = torch.mean(t_z1, dim=0)
            mean_z2 = torch.mean(t_z2, dim=0)

            current_classes_mean_z1[i] = mean_z1
            current_classes_mean_z2[i] = mean_z2

        nonZeroRows = torch.abs(current_classes_mean_z1).sum(dim=1) > 0
        nonZero_prototype_z1 = current_classes_mean_z1[nonZeroRows]
        nonZero_prototype_z2 = current_classes_mean_z2[nonZeroRows]
        nonZero_proto_label =  proto_label[nonZeroRows]
        return nonZero_prototype_z1, nonZero_prototype_z2, nonZero_proto_label, current_classes_mean_z1, current_classes_mean_z2
def cal_buffer_prototype(buffer_x, buffer_y, buffer_x_pair, task_id, model, class_per_task):
    buffer_fea, buffer_z = model.forward(buffer_x_pair, return_fast_feat=True)
    buffer_z_norm = F.normalize(buffer_z)
    buffer_z1 = buffer_z_norm[:buffer_x.shape[0]]
    buffer_z2 = buffer_z_norm[buffer_x.shape[0]:]

    buffer_z1_proto, buffer_z2_proto, label, zero_p1, zero_p2 = cal_prototype(buffer_z1, buffer_z2, buffer_y, task_id, class_per_task)
    classes_mean = (zero_p1 + zero_p2) / 2
    return buffer_z1_proto, buffer_z2_proto, label, classes_mean#, buffer_z1, buffer_z2

def cal_buffer_proto_loss(buffer_x, buffer_y, buffer_x_pair, task_id, model, opeloss):
    buffer_fea, buffer_z = model(buffer_x_pair, return_fast_feat=True)
    buffer_z_norm = F.normalize(buffer_z)
    buffer_z1 = buffer_z_norm[:buffer_x.shape[0]]
    buffer_z2 = buffer_z_norm[buffer_x.shape[0]:]

    buffer_proto_loss, buffer_z1_proto, buffer_z2_proto = opeloss(buffer_z1, buffer_z2, buffer_y, task_id)
    classes_mean = (buffer_z1_proto + buffer_z2_proto) / 2

    return buffer_proto_loss, buffer_z1_proto, buffer_z2_proto, classes_mean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignore warning
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use gpu0,1
def rot_inner_all(x):
    num=x.shape[0]

    #print(num)
    R=x.repeat(4,1,1,1)
    a=x.permute(0,1,3,2)
    a = a.view(num,3, 2, 16, 32)
    import pdb
   # pdb.set_trace()
  #  imshow(torchvision.utils.make_grid(a))
    a = a.permute(2,0, 1, 3, 4)
    s1 = a[0]  # .permute(1,0, 2, 3)#, 4)
    s2 = a[1]  # .permute(1,0, 2, 3)
    a= torch.rot90(a, 2, (3, 4))
    s1_1=a[0]#.permute(1,0, 2, 3)#, 4)
    s2_2=a[1]

    R[3*num:] = torch.cat((s1_1.unsqueeze(2), s2.unsqueeze(2)), dim=2).reshape(num,3, 32, 32).permute(0,1,3,2)
    R[num:2*num] = torch.cat((s1.unsqueeze(2), s2_2.unsqueeze(2)), dim=2).reshape(num,3, 32, 32).permute(0,1,3,2)
    R[2*num:3*num] = torch.cat((s1_1.unsqueeze(2), s2_2.unsqueeze(2)), dim=2).reshape(num,3, 32, 32).permute(0,1,3,2)

    return R

def get_buffer_dict(buffer, class_per_task, current_task_id):
    start_i = 0
    end_i = (current_task_id + 1) * class_per_task


    x_indices = torch.arange(buffer.x.shape[0])
    y_indices = torch.arange(buffer.y.shape[0])
    y = buffer.y
    _, y = torch.max(y, dim=1)

    class_x_list = []
    class_y_list = []
    class_id_map = {}
    for task_id in range(start_i, end_i):
        indices = (y == task_id)
        if not any(indices):
            continue

        class_x_list.append(x_indices[indices])
        class_y_list.append(y_indices[indices])
        class_id_map[task_id] = len(class_y_list) - 1
    return class_x_list, class_y_list, class_id_map
def Rotation(x,y):
   # print(x.shape)
  # if r<=-1:
        num=x.shape[0]
        X = rot_inner_all(x)#, 1, 0)
        ori_y=y
        y=y.repeat(16)
        for i in range(1,16):
            y[i*num:(i+1)*num]+=1000*i
        return torch.cat((X,torch.rot90(X,1,(2,3)),torch.rot90(X,2,(2,3)),torch.rot90(X,3,(2,3))),dim=0),y
   #else:
def loss_mixup(logits, y):
    criterion = F.cross_entropy
    loss_a = criterion(logits, y[:, 0].long(), reduction='none')
    loss_b = criterion(logits, y[:, 1].long(), reduction='none')
    return ((1 - y[:, 2]) * loss_a + y[:, 2] * loss_b).mean()

def sim_mixup(images, labels, mask, buffer_batch_size):
    # mask = mask[:buffer_batch_size, :buffer_batch_size]
    rows, cols = torch.where(mask == 1)
    colect = []
    mixed_images_list = []
    mixed_labels_list = []
    for row, col in zip(rows, cols):
        row_ = row
        col_ = col
        if row >= buffer_batch_size:
            row_ = row - buffer_batch_size
        if col >= buffer_batch_size:
            col_ = col - buffer_batch_size
        if [row_, col_] not in colect:
            x1 = images[row_]
            x2 = images[col_]
            y1 = labels[row_]
            y2 = labels[col_]
            lam = torch.tensor(np.random.uniform(0.1, 0.6), dtype=torch.float32).cuda()
            image = (1 - lam) * x1 + lam * x2
            mixed_images_list.append(image)
            mixed_labels_list.append(torch.tensor([y1, y2, lam.item()]))
            colect.append([row_, col_])
    r, l = mask.shape
    masksum = mask.sum(dim=1)
    for i in range(buffer_batch_size):
        if masksum[i] == 0 and masksum[i + buffer_batch_size] == 0:
            
            mixed_images_list.append(images[i])
            mixed_labels_list.append(torch.tensor([labels[i], labels[i], 0]))
    mixed_images_list = torch.stack(mixed_images_list).cuda()
    mixed_labels_list = torch.stack(mixed_labels_list).cuda()      
    return mixed_images_list, mixed_labels_list


gpus = [0, 1, 2, 3,5,6,7]
torch.cuda.set_device('cuda:{}'.format(gpus[0]))

print('=' * 100)
print('Arguments =')
for arg in vars(args):
    print('\t' + arg + ':', getattr(args, arg))
print('=' * 100)
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'GPU  ' + os.environ["CUDA_VISIBLE_DEVICES"])
print('=' * 100)
########################################################################################################################

# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
else:
    print('[CUDA unavailable]')
    sys.exit()
import cifar as dataloader
# import owm as approach
# import cnn_owm as network
#from minimodel import net as s_model
# from Resnet18 import resnet18 as b_model
import math
from common import MaskNet18 as b_model
from buffer import Buffer as buffer
# imagenet200 import SequentialTinyImagenet as STI
from torch.optim import Adam, SGD  # ,SparseAdam
import torch.nn.functional as F
from copy import deepcopy
import matplotlib.pyplot as plt
def imshow(img):
    img=img/2+0.5
    npimg=img.cpu().numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()
def test_model(loder,i,model):
    test_loss = 0
    correct = 0
    num = 0
    for batch_idx, (data, target) in enumerate(loder):

        data, target = data.cuda(), target.cuda()
        # data, target = Variable(data, volatile=True), Variable(target)
        model.eval()
        pred=model.forward(data)
        Pred = pred.data.max(1, keepdim=True)[1]
        num += data.size()[0]


        correct += Pred.eq(target.data.view_as(Pred)).cpu().sum()

    test_accuracy = 100. * correct / num  # len(data_loader.dataset)
    print(
        'Test set{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'
            .format(i,
            test_loss, correct, num,
            100. * correct / num, ))
    return test_accuracy

def get_cosine_lr(initial_lr, final_lr, num_tasks):
    lrs = []
    for t in range(num_tasks):
        # 计算当前任务的学习率
        lr = final_lr + 0.5 * (initial_lr - final_lr) * (1 + math.cos(math.pi * t / num_tasks))
        lrs.append(lr)
    return lrs
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
miun = 0.1
miub = 0.2
beta=0.05
lr = args.lr
lrs = get_cosine_lr(0.002, 0.001, 10)
print('is_rot:', args.is_rot)
# with torch.no_grad():
#     resize_scale = (0.3, 1.0)  # resize scaling factor,default [0.08,1]

#     color_jitter = TL.ColorJitterLayer(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8).cuda()
#     color_gray = TL.RandomColorGrayLayer(p=0.25).cuda()
#     resize_crop = TL.RandomResizedCropLayer(scale=resize_scale, size=[32, 32, 3]).cuda()
#     simclr_aug = transform = torch.nn.Sequential(
#         hflip,
#         color_gray,  
#         resize_crop, )
#for n,w in Basic_model.named_parameters():
 #   print(n,w.shape)
Max_acc=[]
print('=' * 100)
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'GPU  ' + os.environ["CUDA_VISIBLE_DEVICES"])
print('=' * 100)
class_holder=[]
buffer_per_class =7
run_nums = args.run_nums
acclist = []
cmp_acc = []
buffer_batch_size=64
def sample_from_buffer_for_prototypes(buffer):
    b_num = buffer.x.shape[0]
    if b_num <=  buffer_batch_size:
        buffer_x =  buffer.x
        buffer_y = buffer.y
        _, buffer_y = torch.max(buffer_y, dim=1)
    else:
        buffer_x, buffer_y, _ =  buffer.sample( buffer_batch_size, exclude_task=None)

    return buffer_x, buffer_y
sys.stdout = Logger(args.logfile) 
for run in range(run_nums):
   # rank=torch.randperm(len(Loder))
    print('Load data...')
    oop=16
    data, taskcla, inputsize, Loder, test_loder = dataloader.get_cifar100_10(seed=args.seed, batch=args.batch)
    print('Input size =', inputsize, '\nTask info =', taskcla)
    buffero = buffer(args).cuda()
    Basic_model = b_model(100).cuda()
    llabel = {}
    Optimizer = Adam(Basic_model.parameters(), lr=0.0005, betas=(0.9, 0.99),weight_decay=1e-4)#SGD(Basic_model.parameters(), lr=0.02, momentum=0.9)
    hflip = TL.HorizontalFlipLayer().cuda()
    Basic_model, Optimizer = amp.initialize(Basic_model, Optimizer,opt_level="O1")

    with torch.no_grad():
        resize_scale = (0.3, 1.0)  # resize scaling factor,default [0.08,1]

        color_jitter = TL.ColorJitterLayer(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8).cuda()
        color_gray = TL.RandomColorGrayLayer(p=0.25).cuda()
        resize_crop = TL.RandomResizedCropLayer(scale=resize_scale, size=[32, 32, 3]).cuda()
        simclr_aug = transform = torch.nn.Sequential(
            hflip,
            color_gray,  
            resize_crop, )
    rank = torch.arange(0, 10)
   # rank = torch.tensor([0,1,2,3,4,5,6,7,8,9])
    
    apf = AdaptiveSimMixFeedback(buffero, max_sim_mix=6, mixup_p=0.2, mixup_lower=0., mixup_upper=0.6, mixup_alpha=0.4, class_per_task=10)
    # apf = AdaptivePrototypicalFeedback(buffero, mixup_base_rate=0.9, mixup_p=0.2, mixup_lower=0, mixup_upper=0.6, mixup_alpha=0.4, class_per_task=10)
    tmp_acc = []
    for i in range(len(Loder)):
        task_id=i
        adjust_learning_rate(optimizer=Optimizer, lr=lrs[i])
        if i==0:
            train_loader = Loder[rank[i].item()]['train']
            for epoch in range(1):
                Basic_model.train()
                num_d=0
                for batch_idx, (x, y) in enumerate(train_loader):
                    num_d+=x.shape[0]
                    # if num_d > 10:
                    #     break
                    if num_d%5000==0:
                        print(num_d,num_d/10000)
                    llabel[i] = []

                    Y = deepcopy(y)
                    for j in range(len(Y)):
                        if Y[j] not in class_holder:
                            class_holder.append(Y[j].detach())

                    x, y = x.cuda(), y.cuda()
                    x = x.requires_grad_()

                ##########SLOW##########
                    weights_before = deepcopy(Basic_model.state_dict())
                    images_pair = torch.cat([x, simclr_aug(x)], dim=0)
                    feature_map,outputs_aux = Basic_model(images_pair, return_slow_feat=True)

                    simclr = normalize(outputs_aux)  # normalize
                    feature_map_out = normalize(feature_map[:images_pair.shape[0]]) 

                    num1 = feature_map_out.shape[1] - simclr.shape[1]
                    id1 = torch.randperm(num1)[0]
                    id1_2 = torch.randperm(num1)[1]
                    size = simclr.shape[1]
                    sim_matrix = torch.matmul(simclr, feature_map_out[:, id1 :id1+ 1 * size].t())

                    sim_matrix += 1 * get_similarity_matrix(simclr)  # *(1-torch.eye(simclr.shape[0]).cuda())#+0.5*get_similarity_matrix(feature_map_out)

                    cross_task_sim, _ = Supervised_NT_xent_simn(sim_matrix, labels=y,
                                                   temperature=0.07,miu=miun)
                
                    loss = cross_task_sim * 1
                    with amp.scale_loss(loss, Optimizer) as scaled_loss:
                        scaled_loss.backward()
                    Optimizer.step()         
                    Optimizer.zero_grad()           
                    weights_after = deepcopy(Basic_model.state_dict())
                    new_params = {name : weights_before[name] + ((weights_after[name] - weights_before[name]) * beta) for name in weights_before.keys()}
                    Basic_model.load_state_dict(new_params)
                #########fast################
                    if args.is_rot:
                        images1, rot_sim_labels = Rotation(x, y)
                    else:
                        images1, rot_sim_labels = x, y
                    images_pair = torch.cat([images1, simclr_aug(images1)], dim=0)
                    rot_sim_labels = rot_sim_labels.cuda()

                    feature_map,outputs_aux = Basic_model(images_pair, return_fast_feat=True)

                    simclr = normalize(outputs_aux)  # normalize
                    feature_map_out = normalize(feature_map[:images_pair.shape[0]])
                   # num1 = feature_map_out.shape[1] // simclr.shape[1]
                    num1 = feature_map_out.shape[1] - simclr.shape[1]
                    id1 = torch.randperm(num1)[0]
                    id1_2 = torch.randperm(num1)[1]
                    size = simclr.shape[1]

                    sim_matrix = torch.matmul(simclr, feature_map_out[:, id1 :id1+ 1 * size].t())

                    sim_matrix += 1 * get_similarity_matrix(simclr)  # *(1-torch.eye(simclr.shape[0]).cuda())#+0.5*get_similarity_matrix(feature_map_out)

                    loss_sim1 = Supervised_NT_xent_n(sim_matrix, labels=rot_sim_labels,
                                                   temperature=0.07)

                    lo1 = 1 * loss_sim1 #+0*loss_sim2

                    y_pred = Basic_model.forward(simclr_aug(x))
                    ce = F.cross_entropy(y_pred, y)
                    

                    loss_pp_buf = 0
                    if batch_idx != 0:
                        buffer_x, buffer_y = sample_from_buffer_for_prototypes(buffer=buffero)
                        buffer_x.requires_grad = True
                        buffer_x, buffer_y = buffer_x.cuda(), buffer_y.cuda()
                        buffer_x_pair = torch.cat([buffer_x, simclr_aug(buffer_x)], dim=0)

                        buffer_proto_z, buffer_proto_zt, _,classes_mean = cal_buffer_prototype(buffer_x, buffer_y, buffer_x_pair, task_id, Basic_model, class_per_task=10)
                        loss_pp_buf = Supervised_NT_xent_pp(buffer_proto_z, buffer_proto_zt)


                    z = simclr[:images1.shape[0]]
                    zt = simclr[images1.shape[0]:]
                    cur_new_proto_z, cur_new_proto_zt, new_proto_label, zero_p1,zero_p2 = cal_prototype(z[:x.shape[0]], zt[:x.shape[0]], y, task_id, class_per_task=10)
                    class_mean = (zero_p1+zero_p2)/2

                    newproto = F.normalize((cur_new_proto_z +  cur_new_proto_zt) / 2)
                    
                    newimgs = torch.cat([z[:x.shape[0]], zt[:x.shape[0]]], dim=0)


                    new_sim_proto = torch.matmul(newimgs, newproto.t())

                    new_simpp = torch.matmul(newproto, newproto.t())
                    loss_sim_proto_new = Supervised_NT_xent_proto(new_sim_proto, labels1=torch.cat([y, y],dim=0), labels2=new_proto_label, temperature=0.07)
                    loss_pp_new =Supervised_NT_xent_pp(cur_new_proto_z, cur_new_proto_zt)
                    loss_sim_proto = loss_pp_new + loss_pp_buf              
                    loss = ce+1*lo1 + loss_sim_proto #+ loss_sim_proto

                    with amp.scale_loss(loss, Optimizer) as scaled_loss:
                        scaled_loss.backward()

#                    print(i, epoch, loss)
                    #optimizer[i].step()
                    Optimizer.step()
                    Optimizer.zero_grad()

                    buffero.add_reservoir(x=x.detach(), y=y.detach(), logits=None, t=i)
            Previous_model = deepcopy(Basic_model)
            tmp_list = np.zeros(10)
            for j in range(i + 1):
                print("ori", rank[j].item())
                a = test_model(Loder[rank[j].item()]['test'], j,Basic_model)
                tmp_list[j] = a.item()
                if j == i:
                    Max_acc.append(a)
                if a > Max_acc[j]:
                    Max_acc[j] = a
            print('mean: ',tmp_list[:i+1].mean())
            tmp_acc.append(np.array(tmp_list))

        else:
            train_loader = Loder[rank[i].item()]['train']
            #optimizer.append(Adam(S_model[i].parameters(), lr=0.001, betas=(0.9, 0.99)))  # ,momentum=0.9))
            for epoch in range(1):
            #    S_model[i].train()
                num_d=0
                Basic_model.train()
                for batch_idx, (x, y) in tqdm(enumerate(train_loader)):
                    num_d+=x.shape[0]
                    # if num_d > 10:
                    #     break
                    if num_d%5000==0:
                        print(num_d,num_d/10000)



                    Y = deepcopy(y)
                    for j in range(len(Y)):
                        if Y[j] not in class_holder:
                            class_holder.append(Y[j].detach())
                    task_id=i

                    Optimizer.zero_grad()
                    # if args.cuda:
                    x, y = x.cuda(), y.cuda()
                    # x, y = Variable(x), Variable(y)
                    x = x.requires_grad_()
                    buffer_batch_size = min(64,buffer_per_class*len(class_holder))
                    mem_x, mem_y,_= buffero.sample(buffer_batch_size, exclude_task=None)
                    mem_x = mem_x.requires_grad_()

###############################slow
                    weights_before = deepcopy(Basic_model.state_dict())
                    images_pair = torch.cat([x, simclr_aug(x)], dim=0)
                    images_pair_r = torch.cat([mem_x, simclr_aug(mem_x)], dim=0)

                    t =torch.cat((images_pair,images_pair_r),dim=0)
                    feature_map, u = Basic_model.forward(t, return_slow_feat=True)

                    feature_map_out = normalize(feature_map[:images_pair.shape[0]])
                    feature_map_out_r = normalize(feature_map[images_pair.shape[0]:])

                    images_out = u[:images_pair.shape[0]]
                    images_out_r = u[images_pair.shape[0]:]
                    simclr = normalize(images_out)
                    simclr_r = normalize(images_out_r)

                    num1 = feature_map_out.shape[1] - simclr.shape[1]
                    id1 = torch.randperm(num1)[0]
                    id1_2 = torch.randperm(num1)[1]
                    id2=torch.randperm(num1)[0]
                    id2_2 = torch.randperm(num1)[0]
                    size = simclr.shape[1]

                    sim_matrix = torch.matmul(simclr, feature_map_out[:, id1:id1 + size].t())
                    sim_matrix_r = torch.matmul(simclr_r,
                                                     feature_map_out_r[:, id2:id2 + size].t())

                    sim_matrix += 1 * get_similarity_matrix(simclr) 
                    sim_matrix_r += 1 * get_similarity_matrix(simclr_r)

                    sim_n, simn_mask =Supervised_NT_xent_simn(sim_matrix,labels=y,temperature=0.07, miu=miun)
             #       loss_sim_mix2= Supervised_NT_xent(sim_matrix_mix,labels=rot_sim_labels_mix2,temperature=0.07)
                    sim_b, simb_mask = Supervised_NT_xent_simb(sim_matrix_r, labels=mem_y, temperature=0.07, miu=miub)

                    dmi = sim_b + sim_n
                    loss = 1 * dmi
                    with amp.scale_loss(loss, Optimizer) as scaled_loss:
                        scaled_loss.backward()
                    Optimizer.step()    
                    Optimizer.zero_grad()   
                    weights_after = deepcopy(Basic_model.state_dict())
                    new_params = {name : weights_before[name] + ((weights_after[name] - weights_before[name]) * beta) for name in weights_before.keys()}
                    Basic_model.load_state_dict(new_params)             
################################fast
                    
                    ori_mem_x = mem_x
                    ori_mem_y = mem_y
                    mem_x = ori_mem_x
                    mem_y = ori_mem_y

                    rot_sim_labels = torch.cat([y + 100 * i for i in range(oop)], dim=0)
                    rot_sim_labels_r = torch.cat([mem_y + 100 * i for i in range(oop)], dim=0)
                    # if batch_idx != 0:
                    #     mem_x, mem_y, mem_y_mix = apf(ori_mem_x, ori_mem_y, buffer_batch_size, simb_mask, class_mean,task_id)
                    #     # mem_x, mem_y, mem_y_mix = apf(ori_mem_x, ori_mem_y, buffer_batch_size, class_mean,task_id)
                    #     # mem_y = mem_y_mix[:, 0].long()
                    #     rot_sim_labels = torch.cat([y + 100 * i for i in range(oop)], dim=0)
                    #     rot_sim_labels_r = torch.cat([mem_y + 100 * i for i in range(oop)], dim=0)
                    # else:
                    #     mem_x = ori_mem_x
                    #     mem_y = ori_mem_y

                    #     rot_sim_labels = torch.cat([y + 100 * i for i in range(oop)], dim=0)
                    #     rot_sim_labels_r = torch.cat([mem_y + 100 * i for i in range(oop)], dim=0)

                    mem_x = mem_x.requires_grad_()

                    if args.is_rot:
                        images1, rot_sim_labels = Rotation(x, y)  # torch.cat([Rotation(x, r) for r in range(4)])


                        images1_r, rot_sim_labels_r = Rotation(mem_x,
                                                            mem_y)  # torch.cat([Rotation(mem_x, r) for r in range(4)])
                    else:
                        images1, rot_sim_labels = x, y


                        images1_r, rot_sim_labels_r = mem_x,mem_y

                    images_pair = torch.cat([images1, simclr_aug(images1)], dim=0)
                    images_pair_r = torch.cat([images1_r, simclr_aug(images1_r)], dim=0)

                    t =torch.cat((images_pair,images_pair_r),dim=0)
                    feature_map, u = Basic_model.forward(t, return_fast_feat=True)
                    pre_u_feature, pre_u = Previous_model.forward(images1_r, return_fast_feat=True)
                    feature_map_out = normalize(feature_map[:images_pair.shape[0]])
                    feature_map_out_r = normalize(feature_map[images_pair.shape[0]:])
                    pre_feature_map_out_r = pre_u_feature

                    images_out = u[:images_pair.shape[0]]
                    images_out_r = u[images_pair.shape[0]:]
                    pre_u = normalize(pre_u)#torch.cat((images_out_r,pre_u),dim=0)

                    simclr = normalize(images_out)
                    simclr_r = normalize(images_out_r)
                    simclr_pre = normalize(pre_feature_map_out_r)

                    num1 = feature_map_out.shape[1] - simclr.shape[1]
                    id1 = torch.randperm(num1)[0]
                    id1_2 = torch.randperm(num1)[1]
                    id2=torch.randperm(num1)[0]
                    id2_2 = torch.randperm(num1)[0]
                    size = simclr.shape[1]

                    sim_matrix = torch.matmul(simclr, feature_map_out[:, id1:id1 + size].t())
                    sim_matrix_r = torch.matmul(simclr_r,
                                                     feature_map_out_r[:, id2:id2 + size].t())

                    sim_matrix += 1 * get_similarity_matrix(
                        simclr) 
                    sim_matrix_r += 1 * get_similarity_matrix(simclr_r)
                    sim_matrix_r_pre = torch.matmul(simclr_r[:images1_r.shape[0]],pre_u.t())

                    loss_sim_r =Supervised_NT_xent_uni(sim_matrix_r,labels=rot_sim_labels_r,temperature=0.07)

                    loss_sim_pre = Supervised_NT_xent_pre(sim_matrix_r_pre, labels=rot_sim_labels_r, temperature=0.07)
                    loss_sim = Supervised_NT_xent_n(sim_matrix, labels=rot_sim_labels, temperature=0.07)

                    lo1 =1 * loss_sim_r+1*loss_sim+1*loss_sim_pre
#####################proto
                    z = simclr[:images1.shape[0]]
                    zt = simclr[images1.shape[0]:]

                    zr = simclr_r[:images1_r.shape[0]]
                    zrt = simclr_r[images1_r.shape[0]:]

                    # cur_new_proto_z, cur_new_proto_zt, new_proto_label, _, _ = cal_prototype(z[:x.shape[0]], zt[:x.shape[0]], y, task_id, class_per_task=10)
                    # buffer_x_pair = torch.cat([ori_mem_x, simclr_aug(ori_mem_x)], dim=0)
                    # buffer_z1_proto, buffer_z2_proto, buffer_proto_label, class_mean = cal_buffer_prototype(ori_mem_x, ori_mem_y, buffer_x_pair, task_id, Basic_model, class_per_task=10)
                    class_x_list, class_y_list, class_id_map = get_buffer_dict(buffer=buffero, class_per_task=10, current_task_id=task_id)
                    x_for_proto, y_for_proto=sample_similar_for_proto(x, y, buffero, 6, class_x_list, class_y_list, class_id_map)
                    bx_for_proto, by_for_proto=sample_similar_for_proto(ori_mem_x, ori_mem_y, buffero, 6, class_x_list, class_y_list, class_id_map)
                    x_proto, xt_proto, y_proto_label, _ = cal_buffer_prototype(x_for_proto, y_for_proto, torch.cat([x_for_proto, simclr_aug(x_for_proto)], dim=0), task_id, Basic_model, class_per_task=10)
                    bx_proto, bxt_proto, by_proto_label, class_mean = cal_buffer_prototype(bx_for_proto, by_for_proto, torch.cat([bx_for_proto, simclr_aug(bx_for_proto)], dim=0), task_id, Basic_model, class_per_task=10)
                    # proto_n = torch.cat([cur_new_proto_z, cur_new_proto_zt], dim=0)
                    # proto_buf = torch.cat([buffer_z1_proto, buffer_z2_proto], dim=0)

                    proto_n = normalize((x_proto+xt_proto)  / 2)
                    proto_buf = normalize((bx_proto+bxt_proto) / 2)

                    
                    z_n = torch.cat([z[:x.shape[0]], zt[:x.shape[0]]], dim=0)
                    z_buf=  torch.cat([zr[:ori_mem_x.shape[0]], zrt[:ori_mem_x.shape[0]]], dim=0)

                    new_sim_proto = torch.matmul(proto_n, z_n.t())
                    buffer_sim_proto = torch.matmul(proto_buf, z_buf.t())


                    loss_sim_proto_new = Supervised_NT_xent_proto(new_sim_proto, labels1=y_proto_label, labels2=torch.cat([y, y],dim=0), temperature=0.07)
                    loss_sim_proto_buffer = Supervised_NT_xent_proto(buffer_sim_proto, labels1=by_proto_label, labels2=torch.cat([ori_mem_y, ori_mem_y],dim=0), temperature=0.07)
                    sim_ps = loss_sim_proto_new + loss_sim_proto_buffer
                    # loss_pp_buf = Supervised_NT_xent_pp(buffer_z1_proto, buffer_z2_proto)
                    # loss_pp_new = Supervised_NT_xent_pp(cur_new_proto_z, cur_new_proto_zt)
                    loss_pp_buf = Supervised_NT_xent_pp(bx_proto, bxt_proto, temperature=0.07)
                    loss_pp_new = Supervised_NT_xent_pp(x_proto, xt_proto, temperature=0.07)
                    # sim_pp= loss_pp_new+loss_pp_buf
                    rmi = sim_ps
                    smi = loss_pp_new 
                    mi = 1*rmi + 1*smi
##### ####################proto######################                    
                    y_pred_r = Basic_model(simclr_aug(mem_x))
                    y_label_pre = Previous_model(simclr_aug(mem_x))
                    # if batch_idx != 0:
                    #     ce = loss_mixup(y_pred_r, mem_y_mix)
                    # else:
                    #     ce = F.cross_entropy(y_pred_r, mem_y)  
                    # if batch_idx != 0:
                    #     ce = F.cross_entropy(y_pred_r, mem_y_mix[:, 0].long())
                    # else:
                    #     ce = F.cross_entropy(y_pred_r, mem_y)  
                    ce = F.cross_entropy(y_pred_r, mem_y) 
                    loss = 1 * ce + lo1 + 1 * F.mse_loss(y_label_pre[:, :10 * task_id],
                                                                                       y_pred_r[:,
                                                                                       :10 * task_id])+  1 * mi
                    with amp.scale_loss(loss, Optimizer) as scaled_loss:
                        scaled_loss.backward()

                    Optimizer.step()
                    Optimizer.zero_grad()
                    buffero.add_reservoir(x=x.detach(), y=y.detach(), logits=None, t=i)
            Previous_model = deepcopy(Basic_model)
            tmp_list = np.zeros(10)
            for j in range(i + 1):
                print("ori", rank[j].item())
                a = test_model(Loder[rank[j].item()]['test'], j,Basic_model)
                tmp_list[j] = a.item()
                if j == i:
                    Max_acc.append(a)
                if a > Max_acc[j]:
                    Max_acc[j] = a
            print('mean: ',tmp_list[:i+1].mean())
            tmp_acc.append(np.array(tmp_list))
    cmp_acc.append(np.array(tmp_acc))
    print('=' * 100)
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'GPU  ' + os.environ["CUDA_VISIBLE_DEVICES"])
    print('=' * 100)
    test_loss = 0
    correct = 0
    num = 0
    for batch_idx, (data, target) in enumerate(test_loder):

        data, target = data.cuda(), target.cuda()
        Basic_model.eval()
        pred=Basic_model.forward(data)
        Pred = pred.data.max(1, keepdim=True)[1]
        num += data.size()[0]
        correct += Pred.eq(target.data.view_as(Pred)).cpu().sum()

    test_accuracy = 100. * correct / num  # len(data_loader.dataset)
    print(
        'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'
            .format(
            test_loss, correct, num,
            100. * correct / num, ))
    long=len(Max_acc)
    summ=0
    for i in range(long):
        summ+=Max_acc[i]
    print("total",summ)
    acclist.append(100. * correct / num)
    print(acclist)
    compute_clustering_metrics(model=Basic_model, test_loader=test_loder)
    cmparry = np.array(cmp_acc)
    accarry = torch.tensor(acclist)
    print(accarry.mean(), accarry.std())
    # avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt = compute_performance(cmparry)
    # print('=' * 100)
    # # print(f"total {i+1}runs test acc results: {accarry}")
    # print('----------- Avg_End_Acc {} Avg_End_Fgt {} Avg_Acc {} Avg_Bwtp {} Avg_Fwt {}-----------'
    #         .format(avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt))
    # print_normalized_class_norms(Basic_model)
    
print('=' * 100)
cmp_acc = np.array(cmp_acc)
acclist = torch.tensor(acclist)
print(acclist.mean(), acclist.std())
avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt = compute_performance(cmp_acc)
print('=' * 100)
print(f"total {run_nums}runs test acc results: {acclist}")
print('----------- Avg_End_Acc {} Avg_End_Fgt {} Avg_Acc {} Avg_Bwtp {} Avg_Fwt {}-----------'
        .format(avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt))
print('=' * 100)