from sys import prefix
import torch
import os
import numpy as np

def load_network_and_optimizer(net, opt, pretrained_dir, gpu):
    pretrained = torch.load(
        pretrained_dir, 
        map_location=torch.device("cuda:"+str(gpu)))
    pretrained_dict = pretrained['state_dict']
    model_dict = net.state_dict()
    pretrained_dict_update = {}
    pretrained_dict_remove = []
    for k, v in pretrained_dict.items():
        if k in model_dict:
            pretrained_dict_update[k] = v
        elif k[:7] == 'module.':
            if k[7:] in model_dict:
                pretrained_dict_update[k[7:]] = v
        else:
            pretrained_dict_remove.append(k)
    model_dict.update(pretrained_dict_update)
    net.load_state_dict(model_dict)
    opt.load_state_dict(pretrained['optimizer'])
    del(pretrained)
    return net.cuda(gpu), opt, pretrained_dict_remove

def load_network(net, pretrained_dir, gpu):
    pretrained = torch.load(
        pretrained_dir, 
        map_location=torch.device("cuda:"+str(gpu)))
    pretrained_dict = pretrained['state_dict']
    model_dict = net.state_dict()
    pretrained_dict_update = {}
    pretrained_dict_remove = []
    
    for k, v in pretrained_dict.items():
        if k in model_dict:
            pretrained_dict_update[k] = v
        elif k[:7] == 'module.':
            if k[7:] in model_dict:
                pretrained_dict_update[k[7:]] = v
        else:
            pretrained_dict_remove.append(k)
    print(pretrained_dict_remove)
    model_dict.update(pretrained_dict_update)
    net.load_state_dict(model_dict)
    del(pretrained)
    return net.cuda(gpu), pretrained_dict_remove

# def load_network_trn(net, pretrained_dir, gpu):
#     pretrained = torch.load(
#         pretrained_dir, 
#         map_location=torch.device("cuda:"+str(gpu)))
#     pretrained_dict = pretrained
#     model_dict = net.state_dict()
#     pretrained_dict_update = {}
#     pretrained_dict_remove = []
#     for k, v in pretrained_dict.items():
#         prefix1= 'module.'
#         prefix2 = 'module.feature_extractor.'
#         # if "feature_extractor" not in k:
#         #     pretrained_dict_remove.append(k)
#         #     continue

#         if k in model_dict:
#             pretrained_dict_update[k] = v
#             # print(k)
#         elif prefix2 + k in model_dict:
#             pretrained_dict_update[prefix2 + k] = v
#         elif k[:7] == 'module.':
#             if k[7:] in model_dict:
#                 pretrained_dict_update[k[7:]] = v
#             else:
#                 pretrained_dict_remove.append(k)
#         else:
#             pretrained_dict_remove.append(k)
#     # print(pretrained_dict_remove)
#     model_dict.update(pretrained_dict_update)
#     net.load_state_dict(model_dict)
#     del(pretrained)
#     return net.cuda(gpu), pretrained_dict_remove

def load_network_trn(net, pretrained_dir, gpu):
    pretrained = torch.load(
        pretrained_dir, 
        map_location=torch.device("cuda:"+str(gpu)))
    pretrained_dict = pretrained
    model_dict = net.state_dict()
    pretrained_dict_update = {}
    pretrained_dict_remove = []
    for k, v in pretrained_dict.items():
        prefix1= ''
        prefix2 = 'feature_extractor.'

        if k in model_dict:
            pretrained_dict_update[k] = v
            # print(k)
        elif prefix2 + k in model_dict:
            pretrained_dict_update[prefix2 + k] = v
        elif k[:7] == 'module.':
            if k[7:] in model_dict:
                pretrained_dict_update[k[7:]] = v
            else:
                pretrained_dict_remove.append(k)
        else:
            pretrained_dict_remove.append(k)
    model_dict.update(pretrained_dict_update)
    net.load_state_dict(model_dict)
    del(pretrained)
    return net.cuda(gpu), pretrained_dict_remove

def save_network(net, opt, step, save_path, max_keep=8):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = 'latest.pth'
    save_dir = os.path.join(save_path, save_file)
    torch.save({'state_dict': net.state_dict(), 'optimizer': opt.state_dict()}, save_dir)

