from __future__ import print_function
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retina import Retina
from utils.box_utils import decode, decode_landm
import time
import torchvision
import os
print(torch.__version__, torchvision.__version__)

parser = argparse.ArgumentParser(description='RetinaPL')

parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=1000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=500, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')

args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def show_files(path, all_files):
    # 首先遍历当前目录所有文件及文件夹
    file_list = os.listdir(path)
    # 准备循环判断每个元素是否是文件夹还是文件，是文件的话，把名称传入list，是文件夹的话，递归
    for file in file_list:
        # 利用os.path.join()方法取得路径全名，并存入cur_path变量，否则每次只能遍历一层目录
        cur_path = os.path.join(path, file)
        # 判断是否是文件夹
        if os.path.isdir(cur_path):
            show_files(cur_path, all_files)
        else:
            if cur_path.find(".jpg") > 0:
                all_files.append(cur_path)

    return all_files

if __name__ == '__main__':
    
    torch.set_grad_enabled(False)
    cfg = cfg_mnet

    # net and model
    net = Retina(cfg=cfg, phase='test')
    net = load_model(net, './weights/mnet_plate.pth', False)
    net.eval()
    print('Finished loading model!')
    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    resize = 1

    # testing begiN
    contents = show_files("./imgs", [])
    for img_name in contents:
        
        img_raw = cv2.imread(img_name, cv2.IMREAD_COLOR)
        print(img_name)
        
        img = np.float32(img_raw)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        tic = time.time()
        loc, conf, landms = net(img)  # forward pass
        print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        landms = landms[:args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        print('priorBox time: {:.4f}'.format(time.time() - tic))
        # show image
        if args.save_image:
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                print(text)
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                            

                # landms
                #cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                #cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
               
                #cv2.circle(img_raw, (b[9], b[10]), 1, (0, 255, 0), 4)
                #cv2.circle(img_raw, (b[11], b[12]), 1, (255, 0, 0), 4)
                
                x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
                w = int(x2 - x1 + 1.0)
                h = int(y2 - y1 + 1.0)
                img_box = np.zeros((h, w, 3))
                img_box = img_raw[y1:y2 + 1, x1:x2 + 1, :]
                # cv2.imshow("img_box",img_box)  
                # print('+++',b[9],b[10])
                
                new_x1, new_y1 = b[9] - x1, b[10] - y1
                new_x2, new_y2 = b[11] - x1, b[12] - y1
                new_x3, new_y3 = b[7] - x1, b[8] - y1
                new_x4, new_y4 = b[5] - x1, b[6] - y1
                print(new_x1, new_y1)
                print(new_x2, new_y2)
                print(new_x3, new_y3)
                print(new_x4, new_y4)
                        
                # 定义对应的点
                points1 = np.float32([[new_x1, new_y1], [new_x2, new_y2], [new_x3, new_y3], [new_x4, new_y4]])
                points2 = np.float32([[0, 0], [94, 0], [0, 24], [94, 24]])
                
                # 计算得到转换矩阵
                #M = cv2.getPerspectiveTransform(points1, points2)
                
                # 实现透视变换转换
                #processed = cv2.warpPerspective(img_box, M, (94, 24))
                img = img_raw[y1:y2,x1:x2]
                # 显示原图和处理后的图像
                #cv2.imshow("processed", img)  
                
            cv2.imwrite( "res.jpg",img_raw)
            cv2.imshow('image', img_raw)
            if cv2.waitKey(1000000) & 0xFF == ord('q'):
                cv2.destroyAllWindows()

