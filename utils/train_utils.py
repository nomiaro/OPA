import numpy as np
import random
from pointnet2 import pointnet2_utils
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from scipy.spatial.distance import cdist
from torch import nn
from models.loss_helper_unlabeled import trans_center
from models.loss_helper_unlabeled import trans_size
from utils.extract_pc import extract_pc_in_box3d, my_compute_box_3d


# prepare the boxes to augment
def select_bboxes(batch_data_label, dataset, box_num, batch_size, pred_bboxes=None, labeled=True, labeled_num=0):
    batch_obj_index = []
    selected_bbox = {'center': [], 'size': []}
    sel_bbox_nums = 0

    for data_label_index in range(batch_size):
        if labeled:
            bbox_size = batch_data_label['instance_bboxes_size'][data_label_index]
        else:
            bbox_size = len(pred_bboxes['center'][data_label_index])

        if bbox_size < 1:
            if labeled:
                print('GT bbox size < 1')
            else:
                print('pseudo bbox size < 1')
            batch_obj_index.append(None)
            selected_bbox['center'].append(None)
            selected_bbox['size'].append(None)
            continue

        if bbox_size <= box_num:
            bbox_selected = [i for i in range(bbox_size)]
        else:
            bbox_selected = random.sample(range(bbox_size), k=box_num)
        
        # obj_index: [num_bbox, num_points, 3]
        obj_index = []
        center_list = []
        size_list = []
        for bbox_ind in bbox_selected:
            if labeled:
                center = batch_data_label['bbox'][data_label_index, bbox_ind, 0:3].cpu().detach().numpy()
                size = batch_data_label['bbox'][data_label_index, bbox_ind, 3:6].cpu().detach().numpy()
                if dataset == 'scannet':
                    size /= 2.0
            else:
                center = pred_bboxes['center'][data_label_index][bbox_ind]
                if center.shape[0] < 1:
                    continue
                size = pred_bboxes['size'][data_label_index][bbox_ind]

            box_3d_corners = my_compute_box_3d(center, size)
            pc = batch_data_label['point_clouds'][data_label_index+labeled_num].cpu().detach().numpy()

            try:
                index = extract_pc_in_box3d(pc, box_3d_corners)
                index = np.nonzero(index)[0]
            except:
                print('QJ error!')
                index = []

            if len(index) > 0:
                obj_index.append(index)
                center_list.append(center)
                size_list.append(size)
                sel_bbox_nums += 1

        if len(obj_index) > 0:
            selected_bbox['center'].append(center_list)
            selected_bbox['size'].append(size_list)
            batch_obj_index.append(obj_index)
        else:
            batch_obj_index.append(None)
            selected_bbox['center'].append(None)
            selected_bbox['size'].append(None)
            
    return batch_obj_index, selected_bbox, sel_bbox_nums

# Select bboxes in the pre-train stage
def select_bboxes_pretrain(batch_data_label, dataset, box_num):
    batch_size = batch_data_label['point_clouds'].size()[0]
    
    batch_obj_index, selected_bbox, sel_bbox_nums = select_bboxes(batch_data_label,
                                                                  dataset,
                                                                  box_num,
                                                                  batch_size,
                                                                  labeled=True)
    return batch_obj_index, selected_bbox, sel_bbox_nums

# Select bboxes in the SSL stage
def select_bboxes_train(batch_data_label, dataset, box_num, box_num_unlableed, pred_bboxes, labeled_num):
    batch_size_unlabeled = batch_data_label['point_clouds'].size()[0] - labeled_num
    
    batch_obj_index_labeled, selected_bbox_labeled, sel_bbox_nums_labeled = select_bboxes(batch_data_label,
                                                                                          dataset,
                                                                                          box_num,
                                                                                          labeled_num,
                                                                                          labeled=True)

    batch_obj_index_unlabeled, selected_bbox_unlabeled, sel_bbox_nums_unlabeled = select_bboxes(batch_data_label,
                                                                                                dataset,
                                                                                                box_num_unlableed,
                                                                                                batch_size_unlabeled,
                                                                                                pred_bboxes,
                                                                                                labeled=False,
                                                                                                labeled_num=labeled_num)
    batch_obj_index = batch_obj_index_labeled + batch_obj_index_unlabeled
    selected_bbox = {}
    selected_bbox['center'] = selected_bbox_labeled['center'] + selected_bbox_unlabeled['center']
    selected_bbox['size'] = selected_bbox_labeled['size'] + selected_bbox_unlabeled['size']
    sel_bbox_nums = sel_bbox_nums_labeled + sel_bbox_nums_unlabeled
    
    return batch_obj_index, selected_bbox, sel_bbox_nums


def adjust_nums_of_points(batch_data_label, batch_obj_index, OBJECT_POINTS_NUM):
    points = []
    near_points_list = []
    for batch_ind, obj_index in enumerate(batch_obj_index):
        if obj_index is not None:
            for index in obj_index:
                if len(index) < OBJECT_POINTS_NUM:
                    xyz = batch_data_label['point_clouds'][batch_ind, index, 0:3].detach().clone()
                    pc = [xyz[ind, 0:3].cpu().detach().numpy() for ind in range(len(index))]
                    near_points = [i for i in range(len(index))]
                    # padding by duplicated points
                    k = OBJECT_POINTS_NUM - len(index)
                    padding = random.choices(pc, k=k)
                    pc += padding
                    points.append(pc)
                    near_points_list.append(near_points)
                elif len(index) > OBJECT_POINTS_NUM:
                    # fps
                    xyz = batch_data_label['point_clouds'][batch_ind, index, 0:3].detach().clone()
                    _xyz = np.array([xyz[ind, 0:3].cpu().detach().numpy() for ind in range(len(index))])
                    '''
                    f = FPS(xyz)
                    pc = f.compute_fps(OBJECT_POINTS_NUM)
                    '''
                    xyz = xyz.view(-1, xyz.size()[0], 3)
                    xyz_flipped = xyz.transpose(1, 2).contiguous()
                    new_xyz = pointnet2_utils.gather_operation(
                        xyz_flipped,
                        pointnet2_utils.furthest_point_sample(xyz, OBJECT_POINTS_NUM)
                    ).transpose(1, 2).contiguous() if OBJECT_POINTS_NUM is not None else None
                    num_pc = new_xyz.size()[1]
                    new_xyz = new_xyz.view(num_pc, 3)
                    pc = [new_xyz[ind, 0:3].cpu().detach().numpy() for ind in range(num_pc)]
                    distances = cdist(_xyz, pc ,'euclidean')
                    near_points = np.argmin(distances, axis=1)
                    points.append(pc)
                    near_points_list.append(near_points)
                else:
                    xyz = batch_data_label['point_clouds'][batch_ind, index, 0:3].detach().clone()
                    pc = [xyz[ind, 0:3].cpu().detach().numpy() for ind in range(len(index))]
                    near_points = [i for i in range(len(index))]
                    points.append(pc)
                    near_points_list.append(near_points)

    points = torch.from_numpy(np.array(points)).to(device)
    points = points.transpose(2, 1).contiguous() # (B, 3, N)
    
    return points, near_points_list


# Augment selected objects
def augment_objs(augmentor, points, OBJECT_POINTS_NUM):
    noise = 0.02 * torch.randn(points.size()[0], OBJECT_POINTS_NUM).cuda()
    displacement = augmentor(points, noise) # (B, 3, N)
    if displacement is not None:
        displacement = displacement.transpose(2, 1).contiguous() # (B, N, 3)
    return displacement


# add displacement to points
def displace_points(batch_data_label, aug_batch_data_label, batch_obj_index, selected_bbox, near_points_list, displacement):
    dis_ind = 0
    for batch_ind, obj_index in enumerate(batch_obj_index):
        if obj_index is not None:
            for ind in range(len(obj_index)):
                center = selected_bbox['center'][batch_ind][ind]
                size = selected_bbox['size'][batch_ind][ind]
                length_x = size[0]
                length_y = size[1]
                length_z = size[2]
                if displacement is not None:
                    pc = aug_batch_data_label['point_clouds'][batch_ind, obj_index[ind], 0:3].clone()
                    _pc = pc.cpu().detach().numpy()
                    dis = displacement[dis_ind, near_points_list[dis_ind]]
                    _dis = displacement[dis_ind, near_points_list[dis_ind]].cpu().detach().numpy()

                    box_3d_corners = my_compute_box_3d(center, size)
                    temp = _pc + _dis
                    try:
                        index = extract_pc_in_box3d(temp, box_3d_corners)
                        index = np.nonzero(index)[0]
                    except:
                        print('QJ error!')
                        index = []
                    if len(index) > 0:
                        aug_batch_data_label['point_clouds'][batch_ind, obj_index[ind][index], 0:3] = pc[index] + dis[index]
                dis_ind += 1


# update point vote
def update_votes(dataset, aug_batch_data_label, DATASET_CONFIG, num_point, labeled_num):
    if dataset == 'scannet':
        point_votes = np.expand_dims(np.zeros([num_point, 9]), 0).repeat(labeled_num, axis=0)
        point_votes_mask = np.expand_dims(np.zeros(num_point), 0).repeat(labeled_num, axis=0)    

        for batch_id in range(labeled_num):
            scene_instance_labels = aug_batch_data_label["instance_labels"][batch_id].cpu().detach().numpy()
            scene_semantic_labels = aug_batch_data_label["semantic_labels"][batch_id].cpu().detach().numpy()
            scene_point_clouds = aug_batch_data_label["point_clouds"][batch_id].cpu().detach().numpy()

            s_point_votes = np.zeros([num_point, 3])
            s_point_votes_mask = np.zeros(num_point)
            
            for i_instance in np.unique(scene_instance_labels):
                # find all points belong to that instance
                ind = np.where(scene_instance_labels == i_instance)[0]
                # find the semantic label
                if scene_semantic_labels[ind[0]] in DATASET_CONFIG.nyu40ids:
                    x = scene_point_clouds[ind,:3]
                    center = 0.5*(x.min(0) + x.max(0))
                    s_point_votes[ind, :] = center - x
                    s_point_votes_mask[ind] = 1.0
                    
            point_votes[batch_id] = np.tile(s_point_votes, (1, 3)) # make 3 votes identical
            point_votes_mask[batch_id] = s_point_votes_mask

        aug_batch_data_label['vote_label'] = torch.from_numpy(point_votes.astype(np.float32)).to(device)
        aug_batch_data_label['vote_label_mask'] = torch.from_numpy(point_votes_mask.astype(np.int64)).to(device)


# calculate the loss for augmentor
def cal_aug_loss(end_points, loss, augLoss, Lambda, epoch, warm_up):
    parameters = torch.max(torch.tensor(1.).cuda(), torch.exp(end_points['bbox_conf']) * 1.).cuda()
    right = torch.abs(1.0 - torch.exp((augLoss - loss * parameters).clamp(min=-4, max=4)))
    lambda_right = Lambda
    if epoch >= warm_up:
        return augLoss + lambda_right * right
    else:
        return augLoss


def get_pseudo_bboxess(batch_data_label, ema_end_points, config_dict, config, box_num_unlableed):
    # produce pseudo ground truth label
    supervised_mask = batch_data_label['supervised_mask']
    labeled_num = torch.nonzero(supervised_mask).squeeze(1).shape[0]
    pred_center = ema_end_points['center'][labeled_num:]
    pred_sem_cls = ema_end_points['sem_cls_scores'][labeled_num:]
    pred_objectness = ema_end_points['objectness_scores'][labeled_num:]

    # obj score threshold
    pred_objectness = nn.Softmax(dim=2)(pred_objectness)
    # the second element is positive score
    pos_obj = pred_objectness[:, :, 1]
    objectness_mask = pos_obj > config_dict['obj_threshold']

    # cls score threshold
    pred_sem_cls = nn.Softmax(dim=2)(pred_sem_cls)
    max_cls, argmax_cls = torch.max(pred_sem_cls, dim=2)
    cls_mask = max_cls > config_dict['cls_threshold']
	
    unsupervised_inds = torch.nonzero(1 - supervised_mask).squeeze(1).long()

    iou_pred = nn.Sigmoid()(ema_end_points['iou_scores'][unsupervised_inds, ...])
    if iou_pred.shape[2] > 1:
        iou_pred = torch.gather(iou_pred, 2, argmax_cls.unsqueeze(-1)).squeeze(-1)  # use pred semantic labels
    else:
        iou_pred = iou_pred.squeeze(-1)

    iou_threshold = config_dict['iou_threshold']
    iou_mask = iou_pred > iou_threshold
    before_iou_mask = torch.logical_and(cls_mask, objectness_mask)
    final_mask = torch.logical_and(before_iou_mask, iou_mask)


    # calculate size and center
    pred_center = trans_center(pred_center,
                               batch_data_label['flip_x_axis'][labeled_num:],
                               batch_data_label['flip_y_axis'][labeled_num:],
                               batch_data_label['rot_mat'][labeled_num:],
                               batch_data_label['scale'][labeled_num:])
    

    size_scores = ema_end_points['size_scores'][labeled_num:]
    size_residuals = ema_end_points['size_residuals'][labeled_num:]
    B, K = size_scores.shape[:2]
    mean_size_arr = config.mean_size_arr
    mean_size_arr = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda()  # (num_size_cluster,3)
    size_class = torch.argmax(size_scores, -1)  # B,num_proposal
    size_residual = torch.gather(size_residuals, 2,
                                 size_class.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 3))  # B,num_proposal,1,3
    size_residual = size_residual.squeeze(2)
    
    size_residual = trans_size(size_class, size_residual, batch_data_label['scale'][labeled_num:], config)
    
    size_base = torch.index_select(mean_size_arr, 0, size_class.view(-1))
    size_base = size_base.view(B, K, 3)
    size = (size_base + size_residual) / 2  # half of the size
    size[size < 0] = 1e-6


    # find top k iou_pred index
    from utils.nms import nms_3d_faster_samecls
    center_np = []
    size_np = []
    for idx, iou_score in enumerate(iou_pred):
        bbox_num = torch.sum(final_mask[idx])
        # iou_score size is 12*128*18
        iou_score = iou_score[final_mask[idx]].cpu().detach().numpy()
        obj_score = pos_obj[idx][final_mask[idx]].cpu().detach().numpy()
        center = pred_center[idx]
        center = center[final_mask[idx]].cpu().detach().numpy()
        _size = size[idx]
        _size = _size[final_mask[idx]].cpu().detach().numpy()
        _cls = argmax_cls[idx]
        _cls = _cls[final_mask[idx]].cpu().detach().numpy()

        if bbox_num >= 2*box_num_unlableed:
            boxes = np.zeros((bbox_num, 8))
            for ind in range(bbox_num):
                boxes[ind, 0] = center[ind, 0] - _size[ind, 0]
                boxes[ind, 1] = center[ind, 1] - _size[ind, 1]
                boxes[ind, 2] = center[ind, 2] - _size[ind, 2]
                boxes[ind, 3] = center[ind, 0] + _size[ind, 0]
                boxes[ind, 4] = center[ind, 1] + _size[ind, 1]
                boxes[ind, 5] = center[ind, 2] + _size[ind, 2]
                boxes[ind, 6] = iou_score[ind] * obj_score[ind]
                boxes[ind, 7] = _cls[ind]
            pick = nms_3d_faster_samecls(boxes, config_dict['nms_iou'], config_dict['use_old_type_nms'])
            center_np.append(center[pick[:2*box_num_unlableed]])
            size_np.append(_size[pick[:2*box_num_unlableed]])
        elif bbox_num == 0:
            center_np.append(np.array([]))
            size_np.append(np.array([]))
        else:
            center_np.append(center[:bbox_num])
            size_np.append(_size[:bbox_num])
    center_np = np.array(center_np, dtype=object)
    size_np = np.array(size_np, dtype=object)

    pred_bboxes = {}
    pred_bboxes['center'] = center_np
    pred_bboxes['size'] = size_np

    return pred_bboxes


class FPS:
    def __init__(self, points):
        self.points = points

    def get_max_distance(self, a, b):
        distances = cdist(a, b ,'euclidean')
        distances = np.min(distances, axis=0)
        return np.argmax(distances)

    def compute_fps(self, num_points):
        center = self.points.mean(0)
        A = np.array([center])
        B = self.points

        ind = self.get_max_distance(A, B)
        A = np.append(A, np.array([B[ind]]), axis=0)
        # delete center
        A = np.delete(A, 0, axis=0)
        B = np.delete(B, ind, axis=0)
        for i in range(num_points-1):
            ind = self.get_max_distance(A, B)
            A = np.append(A, np.array([B[ind]]), axis=0)
            B = np.delete(B, ind, axis=0)
        return A