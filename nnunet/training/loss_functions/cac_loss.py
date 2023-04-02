'''
Descripttion: Context Aware Consistency Loss
version: 
Author: Luckie
Date: 2021-06-21 22:08:51
LastEditors: Luckie
LastEditTime: 2021-11-29 16:01:51
'''
import math, time
import random
from itertools import chain
import torch
import torch.nn.functional as F
from torch import nn
import torch.utils.checkpoint
import numpy as np
import pickle


class CAC(nn.Module):
    def __init__(self, num_classes, out_dim=320, stride=16, selected_num=400, b = 500, step_save=1, temp=0.1, proj_final_dim=128, pos_thresh_value=0.1, weight_intra=0.1, weight_inter=0.05, sup_loss=None, ignore_index=None, testing=False, pretrained=True):

        super(CAC, self).__init__()
        self.num_classes = num_classes
        self.out_dim = out_dim
        self.proj_final_dim = proj_final_dim
        # self.project = nn.Sequential(
        #         nn.Conv3d(self.out_dim, self.out_dim, kernel_size=1, stride=1),
        #         nn.ReLU(inplace=True),
        #         nn.Conv3d(self.out_dim, self.proj_final_dim, kernel_size=1, stride=1)
        #     ).cuda()
        self.stride = stride
        self.selected_num = selected_num
        self.b = b
        self.step_save = step_save
        self.temp = temp
        self.step_count = 0
        # self.feature_bank = []
        # self.pseudo_label_bank = []
        self.feature_bank = {'liver':[],'spleen':[],'right kidney':[],'pancreas':[]}
        self.pseudo_label_bank = {'liver':[],'spleen':[],'right kidney':[],'pancreas':[]}
        self.tasks_feature_bank={'liver':[],'spleen':[],'right kidney':[],'left kidney':[],'pancreas':[]}
        self.tasks_label_bank={'liver':[],'spleen':[],'right kidney':[],'left kidney':[],'pancreas':[]}
        self.pos_thresh_value = pos_thresh_value
        self.weight_intra = weight_intra
        self.weight_inter = weight_inter
        self.task_label_id_dict = {'spleen':1,'right kidney':2,'left kidney':3,'liver':4,'pancreas':5}

    def forward(self,output_ul1=None, output_ul2=None, logits1=None, logits2=None,target1=None, target2=None, ul1=None, br1=None, \
                ul2=None, br2=None, tasks=None, partial_label_guide = True, save_feature_only=False):
        labels_id = []
        self.stride = target1.size()[2] // output_ul1.size()[2]
        for task in tasks:
            labels_id.append(self.task_label_id_dict[task]) #label id determined by task
        task = tasks[0]
        pseudo_logits_1 = F.softmax(logits1, 1).max(1)[0].detach() #[batch_size, h, w]
        pseudo_logits_2 = F.softmax(logits2, 1).max(1)[0].detach()          
        pseudo_label1 = logits1.max(1)[1].detach() #[batch_size, h, w]
        pseudo_label2 = logits2.max(1)[1].detach()
        target1 = F.interpolate(target1, size=output_ul1.size()[2:], mode='nearest')
        target2 = F.interpolate(target2, size=output_ul2.size()[2:], mode='nearest')
        class_num = len(labels_id)
        for label_id in labels_id[::-1]:
            target1[target1==class_num] = label_id
            target2[target2==class_num] = label_id
            class_num-=1
        #print("unique label:",torch.unique(pseudo_label1))


        # # get overlap part
        output_feature_list1 = []
        output_feature_list2 = []
        pseudo_label_list1 = []
        pseudo_label_list2 = []
        pseudo_logits_list1 = []
        pseudo_logits_list2 = []
        target_label_list1 = []
        target_label_list2 = []

        #TODO need to get target label overlap
        for idx in range(logits1.size(0)): # iterate use batch size
            output_ul1_idx = output_ul1[idx]
            output_ul2_idx = output_ul2[idx]
            pseudo_label1_idx = pseudo_label1[idx]
            pseudo_label2_idx = pseudo_label2[idx]
            target_label1_idx = target1[idx]
            target_label2_idx = target2[idx]
            pseudo_logits_1_idx = pseudo_logits_1[idx]
            pseudo_logits_2_idx = pseudo_logits_2[idx]
            output_feature_list1.append(output_ul1_idx[:, ul1[idx][0]//self.stride:br1[idx][0]//self.stride, ul1[idx][1]//self.stride:br1[idx][1]//self.stride, ul1[idx][2]//self.stride:br1[idx][2]//self.stride].permute(1, 2, 3, 0).contiguous().view(-1, output_ul1.size(1)))
            output_feature_list2.append(output_ul2_idx[:, ul2[idx][0]//self.stride:br2[idx][0]//self.stride, ul2[idx][1]//self.stride:br2[idx][1]//self.stride, ul2[idx][2]//self.stride:br2[idx][2]//self.stride].permute(1, 2, 3, 0).contiguous().view(-1, output_ul2.size(1)))
            pseudo_label_list1.append(pseudo_label1_idx[ul1[idx][0]//self.stride:br1[idx][0]//self.stride, ul1[idx][1]//self.stride:br1[idx][1]//self.stride, ul1[idx][2]//self.stride:br1[idx][2]//self.stride].contiguous().view(-1))
            pseudo_label_list2.append(pseudo_label2_idx[ul2[idx][0]//self.stride:br2[idx][0]//self.stride, ul2[idx][1]//self.stride:br2[idx][1]//self.stride, ul2[idx][2]//self.stride:br2[idx][2]//self.stride].contiguous().view(-1))
            target_label_list1.append(target_label1_idx[:,ul1[idx][0]//self.stride:br1[idx][0]//self.stride, ul1[idx][1]//self.stride:br1[idx][1]//self.stride, ul1[idx][2]//self.stride:br1[idx][2]//self.stride].contiguous().view(-1))
            target_label_list2.append(target_label2_idx[:,ul2[idx][0]//self.stride:br2[idx][0]//self.stride, ul2[idx][1]//self.stride:br2[idx][1]//self.stride, ul2[idx][2]//self.stride:br2[idx][2]//self.stride].contiguous().view(-1))# print("output feature list shape:", output_feature_list1[0].shape)
            # print("pseudo label list shape:", pseudo_label_list2[0].shape)
            pseudo_logits_list1.append(pseudo_logits_1_idx[ul1[idx][0]//self.stride:br1[idx][0]//self.stride, ul1[idx][1]//self.stride:br1[idx][1]//self.stride, ul1[idx][2]//self.stride:br1[idx][2]//self.stride].contiguous().view(-1))
            pseudo_logits_list2.append(pseudo_logits_2_idx[ul2[idx][0]//self.stride:br2[idx][0]//self.stride, ul2[idx][1]//self.stride:br2[idx][1]//self.stride, ul2[idx][2]//self.stride:br2[idx][2]//self.stride].contiguous().view(-1))
        output_feat1 = torch.cat(output_feature_list1, 0) #[n, c]
        output_feat2 = torch.cat(output_feature_list2, 0) #[n, c]
        """
        do save feature for t-sne visualization
        """
        # # print("output feat1 shape:", output_feat1.shape)
        # # print("output feat2 shape:", output_feat2.shape)
        pseudo_label1_overlap = torch.cat(pseudo_label_list1, 0) #[n,]
        pseudo_label2_overlap = torch.cat(pseudo_label_list2, 0) #[n,]
        pseudo_logits1_overlap = torch.cat(pseudo_logits_list1, 0) #[n,]
        pseudo_logits2_overlap = torch.cat(pseudo_logits_list2, 0) #[n,] 
        target_label1_overlap = torch.cat(target_label_list1, 0)
        target_label2_overlap = torch.cat(target_label_list2, 0)
        assert output_feat1.size(0) == output_feat2.size(0)
        assert pseudo_label1_overlap.size(0) == pseudo_label2_overlap.size(0)
        assert output_feat1.size(0) == pseudo_label1_overlap.size(0)

        # concat across multi-gpus
        #可以对output ul1 和 pseudo_label1 先进行一次过滤 把概率低的过滤掉 还有预测不正确的
        b, c, d, h, w = output_ul1.size()
        selected_num = self.selected_num
        output_ul1_flatten = output_ul1.permute(0, 2, 3, 4, 1).contiguous().view(b*d*h*w, c)
        output_ul2_flatten = output_ul2.permute(0, 2, 3, 4, 1).contiguous().view(b*d*h*w, c)
        selected_idx1 = np.random.choice(range(b*d*h*w), selected_num, replace=False)
        selected_idx2 = np.random.choice(range(b*d*h*w), selected_num, replace=False)
        output_ul1_flatten_selected = output_ul1_flatten[selected_idx1]
        output_ul2_flatten_selected = output_ul2_flatten[selected_idx2]
        output_ul_flatten_selected = torch.cat([output_ul1_flatten_selected, output_ul2_flatten_selected], 0) #[2*kk, c]
        #output_ul_all = self.concat_all_gather(output_ul_flatten_selected) #[2*N, c]
        output_ul_all = output_ul_flatten_selected
        pseudo_label1_flatten_selected = pseudo_label1.view(-1)[selected_idx1]
        pseudo_label2_flatten_selected = pseudo_label2.view(-1)[selected_idx2]
        pseudo_label_flatten_selected = torch.cat([pseudo_label1_flatten_selected, pseudo_label2_flatten_selected], 0) #[2*kk]
        
        # get selected pred logits
        pseudo_logits1_flatten_selected = pseudo_logits_1.view(-1)[selected_idx1]
        pseudo_logits2_flatten_selected = pseudo_logits_2.view(-1)[selected_idx2]
        pseudo_logits_flatten_selected = torch.cat([pseudo_logits1_flatten_selected, pseudo_logits2_flatten_selected], 0) #[2*kk]
        
        # get selected ground truth target label
        target_label1_flatten_selected = target1.view(-1)[selected_idx1]
        target_label2_flatten_selected = target2.view(-1)[selected_idx2]
        target_label_flatten_selected = torch.cat([target_label1_flatten_selected, target_label2_flatten_selected], 0) #[2*kk]
        
        # pseudo_label_all = self.concat_all_gather(pseudo_label_flatten_selected) #[2*N]
        pseudo_label_all = pseudo_label_flatten_selected

        # use only same dataset feature

        # save task feature get the correct predict feature and thresthhold greater than thresh hold value
        if partial_label_guide:
            a = (target_label_flatten_selected!=0) & (pseudo_label_flatten_selected == target_label_flatten_selected) #得到预测对的当前类别
            b = (target_label_flatten_selected==0)  # 判断未标记的数据中是否有预测的目标类别，如果有则预测错误
            for label_id in labels_id:
                b &= (pseudo_label_flatten_selected != label_id) #得到其他类别或者背景
            filter_idx = (a | b) & (pseudo_logits_flatten_selected > self.pos_thresh_value)
            target_label_flatten_selected_filter = target_label_flatten_selected[filter_idx]
            # pseudo_label_flatten_selected_filter = pseudo_label_flatten_selected[filter_idx]
            # pseudo_logits_flatten_selected_filter = pseudo_logits_flatten_selected[filter_idx]
            # output_ul_flatten_selected_filter = output_ul_flatten_selected[filter_idx]
            output_ul_all = output_ul_all[filter_idx]
            pseudo_label_all = pseudo_label_all[filter_idx]
        
        self.feature_bank[task].append(output_ul_all.detach()) # save feature to memory bank
        self.pseudo_label_bank[task].append(pseudo_label_all) # save feature to pseudo_label bank
        if save_feature_only:
            for task in tasks:
                #self.tasks_feature_bank[task].append(output_ul_flatten_selected_filter[target_label_flatten_selected_filter!=0].detach())
                self.tasks_label_bank[task].append(target_label_flatten_selected_filter[target_label_flatten_selected_filter!=0])
        if len(self.feature_bank[task]) > self.step_save:
            self.feature_bank[task] = self.feature_bank[task][1:]
            self.pseudo_label_bank[task] = self.pseudo_label_bank[task][1:]
            # if len(tasks) == 5 and len(self.tasks_label_bank[task]) > 20: # for feature align to full label dataset
            #     for task in tasks:
            #         self.tasks_feature_bank[task] = self.tasks_feature_bank[task][1:]
            #         self.tasks_label_bank[task] = self.tasks_label_bank[task][1:]
        # else:
        #     self.step_count += 1
        
        if save_feature_only:
            #if use all label we just save the feature for align
            return torch.FloatTensor([0.0]).cuda(),torch.FloatTensor([0.0]).cuda()
        output_ul_all = torch.cat(self.feature_bank[task], 0)
        pseudo_label_all = torch.cat(self.pseudo_label_bank[task], 0)
      
      
        eps = 1e-8
        pos1 = (output_feat1 * output_feat2.detach()).sum(-1, keepdim=True) / self.temp #[n, 1]
        pos2 = (output_feat1.detach() * output_feat2).sum(-1, keepdim=True) / self.temp #[n, 1]
        # print("pos1 shape:", pos1.shape)
        # print("pos2 shape:", pos2.shape)
        # compute loss1
        b = self.b
        def run1(pos, output_feat1, output_ul_idx, pseudo_label_idx, pseudo_label1_overlap, neg_max1):
            # print("gpu: {}, i_1: {}".format(gpu, i))
            mask1_idx = (pseudo_label_idx.unsqueeze(0) != pseudo_label1_overlap.unsqueeze(-1)).float() #[n, b]
            neg1_idx = (output_feat1 @ output_ul_idx.T) / self.temp #[n, b]
            logits1_neg_idx = (torch.exp(neg1_idx - neg_max1) * mask1_idx).sum(-1) #[n, ]
            return logits1_neg_idx

        def run1_0(pos, output_feat1, output_ul_idx, pseudo_label_idx, pseudo_label1_overlap):
            # print("gpu: {}, i_1_0: {}".format(gpu, i))
            mask1_idx = (pseudo_label_idx.unsqueeze(0) != pseudo_label1_overlap.unsqueeze(-1)).float() #[n, b]
            neg1_idx = (output_feat1 @ output_ul_idx.T) / self.temp #[n, b]
            neg1_idx = torch.cat([pos, neg1_idx], 1) #[n, 1+b]
            mask1_idx = torch.cat([torch.ones(mask1_idx.size(0), 1).float().cuda(), mask1_idx], 1) #[n, 1+b]
            if len(neg1_idx) == 0:
                 print("size neg1 idx is 0")
            neg_max1 = torch.max(neg1_idx, 1, keepdim=True)[0] #[n, 1]
            logits1_neg_idx = (torch.exp(neg1_idx - neg_max1) * mask1_idx).sum(-1) #[n, ]
            return logits1_neg_idx, neg_max1



        # computer inter domain contrastive loss
        inter_domain_loss1 = torch.FloatTensor([0]).cuda()
        inter_domain_loss2 = torch.FloatTensor([0]).cuda()
                # get other domain feat
        key = list(self.task_label_id_dict.keys())
        for task in tasks:
            key.remove(task)
        another_task = random.choice(key) # need to random choice
        another_task_id = self.task_label_id_dict[another_task]
        
        # filter the overlap region
        filter_currtent_label1_idx = (pseudo_label1_overlap==another_task_id) & (pseudo_logits1_overlap>self.pos_thresh_value)
        filter_currtent_label2_idx = (pseudo_label2_overlap==another_task_id) & (pseudo_logits2_overlap>self.pos_thresh_value)
        for label_id in labels_id:
             filter_currtent_label1_idx &= (target_label1_overlap!=label_id)
             filter_currtent_label2_idx &= (target_label2_overlap!=label_id) 

        if  self.weight_inter>0 and len(self.tasks_feature_bank[another_task]) > 0 and filter_currtent_label1_idx.sum()>0 and filter_currtent_label2_idx.sum()>0:
            # filter the overlap region
            pseudo_label1_overlap_inter_domain = pseudo_label1_overlap[filter_currtent_label1_idx]
            pseudo_label2_overlap_inter_domain = pseudo_label2_overlap[filter_currtent_label2_idx]
            out_feat1_overlap_inter_domain = output_feat1[filter_currtent_label1_idx]
            out_feat2_overlap_inter_domain = output_feat2[filter_currtent_label2_idx]
            pseudo_logits1_overlap_inter_domain = pseudo_logits1_overlap[filter_currtent_label1_idx]
            pseudo_logits2_overlap_inter_domain = pseudo_logits2_overlap[filter_currtent_label2_idx]
            # sample for inter domain contrastive loss
            another_task_feat =  torch.cat(self.tasks_feature_bank[another_task], 0)
            another_task_label = torch.cat(self.tasks_label_bank[another_task],0)
            another_task_feat_pos = another_task_feat[another_task_label==another_task_id]
            another_task_feat_neg = another_task_feat[another_task_label!=another_task_id]
            if len(another_task_feat_pos) > 0 and len(another_task_feat_neg) > 0:
                another_task_label_neg = another_task_label[another_task_label!=another_task_id]
                inter_domain_selected_idx1 = np.random.choice(range(another_task_feat_pos.size(0)), out_feat1_overlap_inter_domain.size(0), replace=True)
                inter_domain_selected_idx2 = np.random.choice(range(another_task_feat_pos.size(0)), out_feat2_overlap_inter_domain.size(0), replace=True)
                another_task_feat_pos_selected1 = another_task_feat_pos[inter_domain_selected_idx1]
                another_task_feat_pos_selected2 = another_task_feat_pos[inter_domain_selected_idx2]
                inter_domain_pos1 = (out_feat1_overlap_inter_domain * another_task_feat_pos_selected1).sum(-1, keepdim=True) / self.temp #[n, 1]
                inter_domain_pos2 = (out_feat2_overlap_inter_domain * another_task_feat_pos_selected2).sum(-1, keepdim=True) / self.temp #[n, 1]
                logits1_down, neg_max1 =  torch.utils.checkpoint.checkpoint(run1_0,inter_domain_pos1, out_feat1_overlap_inter_domain, another_task_feat_neg, another_task_label_neg, pseudo_label1_overlap_inter_domain)
                logits1 = torch.exp(inter_domain_pos1 - neg_max1).squeeze(-1) / (logits1_down + eps)
                inter_domain_loss1 += torch.mean(-torch.log(logits1 + eps))

                logits2_down, neg_max2 = torch.utils.checkpoint.checkpoint(run1_0,inter_domain_pos2, out_feat2_overlap_inter_domain, another_task_feat_neg, another_task_label_neg, pseudo_label2_overlap_inter_domain)
                logits2 = torch.exp(inter_domain_pos2 - neg_max2).squeeze(-1) / (logits2_down + eps)
                inter_domain_loss2 += torch.mean(-torch.log(logits2 + eps))

                # print("inter_domain_loss1:",inter_domain_loss1)

        N = output_ul_all.size(0)
        # print("n:",N)
        logits1_down = torch.zeros(pos1.size(0)).float().cuda()
        for i in range((N-1)//b + 1):
            # print("i:",i)
            # print("gpu: {}, i: {}".format(gpu, i))
            pseudo_label_idx = pseudo_label_all[i*b:(i+1)*b]
            output_ul_idx = output_ul_all[i*b:(i+1)*b]
            if i == 0:
                #logits1_neg_idx, neg_max1 = run1_0(pos1, output_feat1, output_ul_idx, pseudo_label_idx, pseudo_label1_overlap)
                logits1_neg_idx, neg_max1 = torch.utils.checkpoint.checkpoint(run1_0, pos1, output_feat1, output_ul_idx, pseudo_label_idx, pseudo_label1_overlap)
            else:
                #logits1_neg_idx = run1(pos1, output_feat1, output_ul_idx, pseudo_label_idx, pseudo_label1_overlap, neg_max1)
                logits1_neg_idx = torch.utils.checkpoint.checkpoint(run1, pos1, output_feat1, output_ul_idx, pseudo_label_idx, pseudo_label1_overlap, neg_max1)
            logits1_down += logits1_neg_idx

        logits1 = torch.exp(pos1 - neg_max1).squeeze(-1) / (logits1_down + eps)
        
        # use the partially label to filter the wrong predict
        pos_mask_1 = ((pseudo_logits2_overlap > self.pos_thresh_value) & (pseudo_logits1_overlap < pseudo_logits2_overlap)).float()
        if partial_label_guide:
            filter_a = (target_label2_overlap!=0) & (pseudo_label2_overlap == target_label2_overlap) #得到预测对的当前类别
            filter_b = (target_label2_overlap==0) 
            for label_id in labels_id:
                filter_b &= (pseudo_label2_overlap != label_id) #得到其他类别或者背景
            filter_idx = filter_a | filter_b
            pos_mask_1 = (filter_idx & (pseudo_logits2_overlap > self.pos_thresh_value) & (pseudo_logits1_overlap < pseudo_logits2_overlap)).float()
        
        loss1 = -torch.log(logits1 + eps)
        loss1 = (loss1 * pos_mask_1).sum() / (pos_mask_1.sum() + 1e-12)

        # compute loss2
        def run2(pos, output_feat2, output_ul_idx, pseudo_label_idx, pseudo_label2_overlap, neg_max2):
            # print("gpu: {}, i_2: {}".format(gpu, i))
            mask2_idx = (pseudo_label_idx.unsqueeze(0) != pseudo_label2_overlap.unsqueeze(-1)).float() #[n, b]
            neg2_idx = (output_feat2 @ output_ul_idx.T) / self.temp #[n, b]
            logits2_neg_idx = (torch.exp(neg2_idx - neg_max2) * mask2_idx).sum(-1) #[n, ]
            return logits2_neg_idx

        def run2_0(pos, output_feat2, output_ul_idx, pseudo_label_idx, pseudo_label2_overlap):
            # print("gpu: {}, i_2_0: {}".format(gpu, i))
            mask2_idx = (pseudo_label_idx.unsqueeze(0) != pseudo_label2_overlap.unsqueeze(-1)).float() #[n, b]
            neg2_idx = (output_feat2 @ output_ul_idx.T) / self.temp #[n, b]
            neg2_idx = torch.cat([pos, neg2_idx], 1) #[n, 1+b]
            mask2_idx = torch.cat([torch.ones(mask2_idx.size(0), 1).float().cuda(), mask2_idx], 1) #[n, 1+b]
            neg_max2 = torch.max(neg2_idx, 1, keepdim=True)[0] #[n, 1]
            logits2_neg_idx = (torch.exp(neg2_idx - neg_max2) * mask2_idx).sum(-1) #[n, ]
            return logits2_neg_idx, neg_max2

        N = output_ul_all.size(0)
        logits2_down = torch.zeros(pos2.size(0)).float().cuda()
        for i in range((N-1)//b + 1):
            pseudo_label_idx = pseudo_label_all[i*b:(i+1)*b]
            output_ul_idx = output_ul_all[i*b:(i+1)*b]
            if i == 0:
                logits2_neg_idx, neg_max2 = torch.utils.checkpoint.checkpoint(run2_0, pos2, output_feat2, output_ul_idx, pseudo_label_idx, pseudo_label2_overlap)
            else:
                logits2_neg_idx = torch.utils.checkpoint.checkpoint(run2, pos2, output_feat2, output_ul_idx, pseudo_label_idx, pseudo_label2_overlap, neg_max2)
            logits2_down += logits2_neg_idx

        logits2 = torch.exp(pos2 - neg_max2).squeeze(-1) / (logits2_down + eps)
        pos_mask_2 = ((pseudo_logits1_overlap > self.pos_thresh_value) & (pseudo_logits2_overlap < pseudo_logits1_overlap)).float()
        if partial_label_guide:
            filter_a = (target_label1_overlap!=0) & (pseudo_label1_overlap == target_label1_overlap) #得到预测对的当前类别
            filter_b = (target_label1_overlap==0) 
            for label_id in labels_id:
                filter_b &= (pseudo_label1_overlap != label_id) #得到其他类别或者背景
            filter_idx = filter_a | filter_b
            pos_mask_2 = (filter_idx & (pseudo_logits1_overlap > self.pos_thresh_value) & (pseudo_logits2_overlap < pseudo_logits1_overlap)).float()

        #print("pos mask 2:", pos_mask_2.sum())
        loss2 = -torch.log(logits2 + eps)
        loss2 = (loss2 * pos_mask_2).sum() / (pos_mask_2.sum() + 1e-12)


        loss_unsup = self.weight_intra * (loss1 + loss2 )
        loss_inter_domain = self.weight_inter * (inter_domain_loss1 + inter_domain_loss2)
        #loss_inter_domain_cl = 0.1 * (inter_domain_pos1 + inter_domain_pos2)

        # curr_losses['loss1'] = loss1
        # curr_losses['loss2'] = loss2
        # curr_losses['loss_unsup'] = loss_unsup
        # total_loss = total_loss + loss_unsup
        return loss_unsup, loss_inter_domain
        #return total_loss, curr_losses, outputs

        # else:
        #     raise ValueError("No such mode {}".format(self.mode))

    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        with torch.no_grad():
            tensors_gather = [torch.ones_like(tensor)
                for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

            output = torch.cat(tensors_gather, dim=0)
        return output

    def get_backbone_params(self):
        return self.encoder.get_backbone_params()

    def get_other_params(self):
        if self.mode == 'supervised':
            return chain(self.encoder.get_module_params(), self.classifier.parameters())
        elif self.mode == 'semi':
            return chain(self.encoder.get_module_params(), self.classifier.parameters(), self.project.parameters())
        else:
            raise ValueError("No such mode {}".format(self.mode))