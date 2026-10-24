import os
import json
import time
import datetime
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms

from clip import clip
from timm.models.vision_transformer import vit_base_patch16_224, vit_base_patch16_384, vit_large_patch16_224

import datasets
from models import *

from utils.meter import AverageMeter
from utils.samplers import DownSampler
from utils.losses import *
from utils.evaluator import Evaluator
from utils.templates import ZEROSHOT_TEMPLATES
from imagenet_templates import IMAGENET_TEMPLATES



def load_clip_to_cpu(backbone_name, prec):
    backbone_name = backbone_name.lstrip("CLIP-")
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu").eval()

    model = clip.build_model(state_dict or model.state_dict())

    assert prec in ["fp16", "fp32", "amp"]
    if prec == "fp32" or prec == "amp":
        # CLIP's default precision is fp16
        model.float()

    return model


def load_vit_to_cpu(backbone_name, prec):
    if backbone_name == "IN21K-ViT-B/16":
        model = vit_base_patch16_224(pretrained=True).eval()
    elif backbone_name == "IN21K-ViT-B/16@384px":
        model = vit_base_patch16_384(pretrained=True).eval()
    elif backbone_name == "IN21K-ViT-L/16":
        model = vit_large_patch16_224(pretrained=True).eval()

    assert prec in ["fp16", "fp32", "amp"]
    if prec == "fp16":
        # ViT's default precision is fp32
        model.half()
    
    return model


class Trainer:
    def __init__(self, cfg):

        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        elif cfg.gpu is None:
            self.device = torch.device("cuda")
        else:
            torch.cuda.set_device(cfg.gpu)
            self.device = torch.device("cuda:{}".format(cfg.gpu))

        self.cfg = cfg
        self.build_data_loader()
        self.build_model()
        self.evaluator = Evaluator(cfg, self.many_idxs, self.med_idxs, self.few_idxs)
        self._writer = None

    def normalize_to_range(self, tensor, min_val=0.0, max_val=0.9):
        a = tensor.min()
        b = tensor.max()
        return min_val + (tensor - a) * (max_val - min_val) / (b - a)
    
    def build_data_loader(self):
        cfg = self.cfg
        root = cfg.root
        resolution = cfg.resolution
        expand = cfg.expand

        if cfg.backbone.startswith("CLIP"):
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        print("mean:", mean)
        print("std:", std)

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        transform_plain = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


        if cfg.tte:
            if cfg.tte_mode == "fivecrop":
                transform_test = transforms.Compose([
                    transforms.Resize(resolution + expand),
                    transforms.FiveCrop(resolution),
                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                    transforms.Normalize(mean, std),
                ])

            elif cfg.tte_mode == "tencrop":
                transform_test = transforms.Compose([
                    transforms.Resize(resolution + expand),
                    transforms.TenCrop(resolution),
                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                    transforms.Normalize(mean, std),
                ])
            elif cfg.tte_mode == "randaug":
                _resize_and_flip = transforms.Compose([
                    transforms.RandomResizedCrop(resolution),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])
                transform_test = transforms.Compose([
                    transforms.Lambda(lambda image: torch.stack([_resize_and_flip(image) for _ in range(cfg.randaug_times)])),
                    transforms.Normalize(mean, std),
                ])
        else:
            transform_test = transforms.Compose([
                transforms.Resize(resolution * 8 // 7),
                transforms.CenterCrop(resolution),
                transforms.Lambda(lambda crop: torch.stack([transforms.ToTensor()(crop)])),
                transforms.Normalize(mean, std),
            ])

        train_dataset = getattr(datasets, cfg.dataset)(root, train=True, transform=transform_train)
        train_init_dataset = getattr(datasets, cfg.dataset)(root, train=True, transform=transform_plain)
        train_test_dataset = getattr(datasets, cfg.dataset)(root, train=True, transform=transform_test)
        test_dataset = getattr(datasets, cfg.dataset)(root, train=False, transform=transform_test)

        self.num_classes = train_dataset.num_classes
        self.cls_num_list = train_dataset.cls_num_list
        self.classnames = train_dataset.classnames

        cls_list = torch.cuda.FloatTensor(self.cls_num_list)
        # self.cls_idx = self.normalize_to_range(cls_list, max_val=0.99)# positively prop to sample size

        self.cls_idx = self.normalize_to_range(cls_list.max() - cls_list, max_val=0.99)# negatively prop to sample size


        if cfg.dataset in ["CIFAR100", "CIFAR100_IR10", "CIFAR100_IR50"]:
            split_cls_num_list = datasets.CIFAR100_IR100(root, train=True).cls_num_list
        elif cfg.dataset in ["CIFAR10", "CIFAR10_IR10", "CIFAR10_IR50"]:
            split_cls_num_list = datasets.CIFAR10_IR100(root, train=True).cls_num_list
        else:
            split_cls_num_list = self.cls_num_list
        self.many_idxs = (np.array(split_cls_num_list) > 100).nonzero()[0]
        self.med_idxs = ((np.array(split_cls_num_list) >= 20) & (np.array(split_cls_num_list) <= 100)).nonzero()[0]
        self.few_idxs = (np.array(split_cls_num_list) < 20).nonzero()[0]

        if cfg.init_head == "1_shot":
            init_sampler = DownSampler(train_init_dataset, n_max=1)
        elif cfg.init_head == "10_shot":
            init_sampler = DownSampler(train_init_dataset, n_max=10)
        elif cfg.init_head == "100_shot":
            init_sampler = DownSampler(train_init_dataset, n_max=100)
        else:
            init_sampler = None

        self.train_loader = DataLoader(train_dataset,
            batch_size=cfg.micro_batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True)

        self.train_init_loader = DataLoader(train_init_dataset,
            batch_size=64, sampler=init_sampler, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True)

        self.train_test_loader = DataLoader(train_test_dataset,
            batch_size=64, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True)

        self.test_loader = DataLoader(test_dataset,
            batch_size=64, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True)
        
        assert cfg.batch_size % cfg.micro_batch_size == 0
        self.accum_step = cfg.batch_size // cfg.micro_batch_size


    def build_model(self):
        cfg = self.cfg
        classnames = self.classnames
        num_classes = len(classnames)

        print("Building model")
        if cfg.zero_shot:
            assert cfg.backbone.startswith("CLIP")
            print(f"Loading CLIP (backbone: {cfg.backbone})")
            clip_model = load_clip_to_cpu(cfg.backbone, cfg.prec)
            self.model = ZeroShotCLIP(clip_model)
            self.model.to(self.device)
            self.tuner = None
            self.head = None

            template = "a photo of a {}."
            prompts = self.get_tokenized_prompts(classnames, template)
            self.model.init_text_features(prompts)

        elif cfg.backbone.startswith("CLIP"):
            print(f"Loading CLIP (backbone: {cfg.backbone})")
            clip_model = load_clip_to_cpu(cfg.backbone, cfg.prec)
            self.model = PeftModelFromCLIP_Sage(cfg, clip_model, num_classes)
            self.model.to(self.device)
            self.tuner = self.model.tuner
            self.head = self.model.head


        elif cfg.backbone.startswith("IN21K-ViT"):
            print(f"Loading ViT (backbone: {cfg.backbone})")
            vit_model = load_vit_to_cpu(cfg.backbone, cfg.prec)
            self.model = PeftModelFromViT(cfg, vit_model, num_classes)
            self.model.to(self.device)
            self.tuner = self.model.tuner
            self.head = self.model.head

        if not (cfg.zero_shot or cfg.test_train or cfg.test_only):
            self.build_optimizer()
            self.build_criterion()

            if cfg.init_head == "text_feat":
                self.init_head_text_feat()
            elif cfg.init_head in ["class_mean", "1_shot", "10_shot", "100_shot"]:
                self.init_head_class_mean()
            elif cfg.init_head == "linear_probe":
                self.init_head_linear_probe()
            elif cfg.init_head == "ours":
                self.init_head_ours()
            else:
                print("No initialization with head")
            
            torch.cuda.empty_cache()
        
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1 and cfg.gpu is None:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def build_optimizer(self):
        cfg = self.cfg

        print("Turning off gradients in the model")
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
        print("Turning on gradients in the tuner")
        for name, param in self.tuner.named_parameters():
            param.requires_grad_(True)
        print("Turning on gradients in the head")
        for name, param in self.head.named_parameters():
            param.requires_grad_(True)


        # print parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        tuned_params = sum(p.numel() for p in self.tuner.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())


        self.optim = torch.optim.SGD([{"params": self.tuner.parameters()},
                                    #   {"params": self.zs_scale},
                                      {"params": self.head.parameters()}],
                                      lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, cfg.num_epochs)

        self.scaler = GradScaler() if cfg.prec == "amp" else None


    def build_criterion(self):
        cfg = self.cfg
        cls_num_list = torch.Tensor(self.cls_num_list).to(self.device)

        if cfg.loss_type == "CE":
            self.criterion = nn.CrossEntropyLoss()
        elif cfg.loss_type == "Focal": # https://arxiv.org/abs/1708.02002
            self.criterion = FocalLoss()
        elif cfg.loss_type == "LDAM": # https://arxiv.org/abs/1906.07413
            self.criterion = LDAMLoss(cls_num_list=cls_num_list, s=cfg.scale)
        elif cfg.loss_type == "CB": # shttps://arxiv.org/abs/1901.05555
            self.criterion = ClassBalancedLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "GRW": # https://arxiv.org/abs/2103.16370
            self.criterion = GeneralizedReweightLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "BS": # https://arxiv.org/abs/2007.10740
            self.criterion == BalancedSoftmaxLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "LA": # https://arxiv.org/abs/2007.07314
            self.criterion = LogitAdjustedLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "LADE": # https://arxiv.org/abs/2012.00321
            self.criterion = LADELoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "ours": 
            print('using our loss...')
            self.criterion = OursLoss(cls_num_list=cls_num_list, Sage_gamma=cfg.Sage_gamma, Sage_mu=cfg.Sage_mu)


    def get_tokenized_prompts(self, classnames, template):
        prompts = [template.format(c.replace("_", " ")) for c in classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)
        return prompts

    @torch.no_grad()
    def init_head_text_feat(self):
        cfg = self.cfg
        classnames = self.classnames

        print("Initialize head with text features")
        if cfg.prompt == "ensemble":
            all_text_features = []
            for template in tqdm(ZEROSHOT_TEMPLATES['imagenet']):
                prompts = self.get_tokenized_prompts(classnames, template)
                text_features = self.model.encode_text(prompts)
                text_features = F.normalize(text_features, dim=-1)
                all_text_features.append(text_features)
            all_text_features = torch.stack(all_text_features)
            text_features = all_text_features.mean(dim=0)
            print('ensemble')
        elif cfg.prompt == "descriptor":
            with open("utils/descriptors_imagenet.json") as f:
                descriptors = json.load(f)
            template = "{}"
            all_class_features = []
            for cn in tqdm(classnames):
                prompts = self.get_tokenized_prompts(descriptors[cn], template)
                text_features = self.model.encode_text(prompts)
                text_features = F.normalize(text_features, dim=-1)
                all_class_features.append(text_features.mean(dim=0))
            text_features = torch.stack(all_class_features)
        elif cfg.prompt == "classname":
            template = "{}"
            prompts = self.get_tokenized_prompts(classnames, template)
            text_features = self.model.encode_text(prompts)
            text_features = F.normalize(text_features, dim=-1)
        elif cfg.prompt == "default":
            template = "a photo of a {}."
            prompts = self.get_tokenized_prompts(classnames, template)
            text_features = self.model.encode_text(prompts)
            text_features = F.normalize(text_features, dim=-1)

        if cfg.backbone.startswith("CLIP-ViT"):
            text_features = text_features @ self.model.image_encoder.proj.t()
            text_features = F.normalize(text_features, dim=-1)

        self.head.apply_weight(text_features)

        ###
        all_teacher_features = []
        for single_template in IMAGENET_TEMPLATES:# 60 tempates
            prompts = [single_template.format(c.replace("_", " ")) for c in classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts])
            text_features = self.model.encode_text(prompts.cuda())# size (num_cls, feat_dim)
            all_teacher_features.append(text_features.unsqueeze(1))

        all_text_features = torch.cat(all_teacher_features, dim=1)# size (num_cls, 60, feat_dim)
        all_text_features = F.normalize(all_text_features, dim=-1)

        if cfg.backbone.startswith("CLIP-ViT"):
            text_features = all_text_features @ self.model.image_encoder.proj.t()
            text_features = F.normalize(text_features, dim=-1)

        self.templates_text_feat = F.normalize(text_features, dim=-1)


    @torch.no_grad()
    def init_head_ours(self):
        cfg = self.cfg
        classnames = self.classnames

        print("Initialize head with ours text features !!!!!!!")
        all_teacher_features = []
        for single_template in IMAGENET_TEMPLATES:# 60 tempates
            prompts = [single_template.format(c.replace("_", " ")) for c in classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts])
            text_features = self.model.encode_text(prompts.cuda())# size (num_cls, feat_dim)
            all_teacher_features.append(text_features.unsqueeze(1))

        all_text_features = torch.cat(all_teacher_features, dim=1)# size (num_cls, 60, feat_dim)
        all_text_features = F.normalize(all_text_features, dim=-1)

        if cfg.backbone.startswith("CLIP-ViT"):
            text_features = all_text_features @ self.model.image_encoder.proj.t()
            text_features = F.normalize(text_features, dim=-1)

        self.templates_text_feat = F.normalize(text_features, dim=-1)
        mean_text_feat = text_features.mean(1) 

        self.head.apply_weight(mean_text_feat)
        self.mean_text_feat = text_features.mean(1)


    @torch.no_grad()
    def init_head_class_mean(self):
        print("Initialize head with class means")
        all_features = []
        all_labels = []

        for batch in tqdm(self.train_init_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            feature = self.model(image, use_tuner=False, return_feature=True)

            all_features.append(feature)
            all_labels.append(label)

        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        sorted_index = all_labels.argsort()
        all_features = all_features[sorted_index]
        all_labels = all_labels[sorted_index]

        unique_labels, label_counts = torch.unique(all_labels, return_counts=True)

        class_means = [None] * self.num_classes
        idx = 0
        for i, cnt in zip(unique_labels, label_counts):
            class_means[i] = all_features[idx: idx+cnt].mean(dim=0, keepdim=True)
            idx += cnt
        class_means = torch.cat(class_means, dim=0)
        class_means = F.normalize(class_means, dim=-1)

        self.head.apply_weight(class_means)

    @torch.no_grad()
    def init_head_linear_probe(self):
        print("Initialize head with linear probing")
        all_features = []
        all_labels = []

        for batch in tqdm(self.train_init_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            feature = self.model(image, use_tuner=False, return_feature=True)

            all_features.append(feature)
            all_labels.append(label)

        all_features = torch.cat(all_features, dim=0).cpu()
        all_labels = torch.cat(all_labels, dim=0).cpu()

        clf = LogisticRegression(solver="lbfgs", max_iter=100, penalty="l2", class_weight="balanced").fit(all_features, all_labels)
        class_weights = torch.from_numpy(clf.coef_).to(all_features.dtype).to(self.device)
        class_weights = F.normalize(class_weights, dim=-1)

        self.head.apply_weight(class_weights)


    def train(self):
        cfg = self.cfg
        cos = torch.nn.CosineSimilarity(dim=1,eps=1e-07)

        self.simpler = normal.Normal(0, 1/3)
        cls_list = torch.cuda.FloatTensor(self.cls_num_list)
        m_list = torch.log(cls_list)
        m_list = m_list.max()-m_list
        self.m_list = m_list

        # Initialize summary writer
        writer_dir = os.path.join(cfg.output_dir, "tensorboard")
        os.makedirs(writer_dir, exist_ok=True)
        print(f"Initialize tensorboard (log_dir={writer_dir})")
        self._writer = SummaryWriter(log_dir=writer_dir)

        # Initialize average meters
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_meter = AverageMeter(ema=True)
        acc_meter = AverageMeter(ema=True)
        cls_meters = [AverageMeter(ema=True) for _ in range(self.num_classes)]

        # Remember the starting time (for computing the elapsed time)
        time_start = time.time()
        num_epochs = cfg.num_epochs
        sampling_ratios = torch.cuda.FloatTensor(self.cls_num_list).min() / torch.cuda.FloatTensor(self.cls_num_list)# size (num_cls, )
        for epoch_idx in range(num_epochs):
            self.tuner.train()
            end = time.time()

            num_batches = len(self.train_loader)
            for batch_idx, batch in enumerate(self.train_loader):
                data_time.update(time.time() - end)

                image = batch[0]
                label = batch[1]
                image = image.to(self.device)
                label = label.to(self.device)


                if cfg.prec == "amp":
                    with autocast():

                        output, [raw_output, f_zs_output, w_zs_output] = self.model(image, init_text=self.mean_text_feat)

                        if cfg.loss_type == "ours": 
                            loss = self.criterion(raw_output, label) + cfg.Sage_lambda_1 *self.criterion(f_zs_output, label) +\
                                        cfg.Sage_lambda_2 *self.criterion(w_zs_output, label) + cfg.Sage_lambda_3 * self.criterion(output, label)

                        loss_micro = loss / self.accum_step
                        self.scaler.scale(loss_micro).backward()


                    if ((batch_idx + 1) % self.accum_step == 0) or (batch_idx + 1 == num_batches):
                        self.scaler.step(self.optim)
                        self.scaler.update()
                        self.optim.zero_grad()

                else:
                    output = self.model(image)
                    loss = self.criterion(output, label)
                    loss_micro = loss / self.accum_step
                    loss_micro.backward()
                    if ((batch_idx + 1) % self.accum_step == 0) or (batch_idx + 1 == num_batches):
                        self.optim.step()
                        self.optim.zero_grad()

                with torch.no_grad():
                    pred = output.argmax(dim=1)
                    correct = pred.eq(label).float()
                    acc = correct.mean().mul_(100.0)

                current_lr = self.optim.param_groups[0]["lr"]
                loss_meter.update(loss.item())
                acc_meter.update(acc.item())
                batch_time.update(time.time() - end)

                for _c, _y in zip(correct, label):
                    cls_meters[_y].update(_c.mul_(100.0).item(), n=1)
                cls_accs = [cls_meters[i].avg for i in range(self.num_classes)]


                mean_acc = np.mean(np.array(cls_accs))
                many_acc = np.mean(np.array(cls_accs)[self.many_idxs])
                med_acc = np.mean(np.array(cls_accs)[self.med_idxs])
                few_acc = np.mean(np.array(cls_accs)[self.few_idxs])

                meet_freq = (batch_idx + 1) % cfg.print_freq == 0
                only_few_batches = num_batches < cfg.print_freq
                if meet_freq or only_few_batches:
                    nb_remain = 0
                    nb_remain += num_batches - batch_idx - 1
                    nb_remain += (
                        num_epochs - epoch_idx - 1
                    ) * num_batches
                    eta_seconds = batch_time.avg * nb_remain
                    eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                    info = []
                    info += [f"epoch [{epoch_idx + 1}/{num_epochs}]"]
                    info += [f"batch [{batch_idx + 1}/{num_batches}]"]
                    info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                    info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                    info += [f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})"]
                    info += [f"acc {acc_meter.val:.4f} ({acc_meter.avg:.4f})"]
                    info += [f"(mean {mean_acc:.4f} many {many_acc:.4f} med {med_acc:.4f} few {few_acc:.4f})"]
                    info += [f"lr {current_lr:.4e}"]
                    info += [f"eta {eta}"]
                    print(" ".join(info))

                n_iter = epoch_idx * num_batches + batch_idx
                self._writer.add_scalar("train/lr", current_lr, n_iter)
                self._writer.add_scalar("train/loss.val", loss_meter.val, n_iter)
                self._writer.add_scalar("train/loss.avg", loss_meter.avg, n_iter)
                self._writer.add_scalar("train/acc.val", acc_meter.val, n_iter)
                self._writer.add_scalar("train/acc.avg", acc_meter.avg, n_iter)
                self._writer.add_scalar("train/mean_acc", mean_acc, n_iter)
                self._writer.add_scalar("train/many_acc", many_acc, n_iter)
                self._writer.add_scalar("train/med_acc", med_acc, n_iter)
                self._writer.add_scalar("train/few_acc", few_acc, n_iter)
                
                end = time.time()

            self.sched.step()
            torch.cuda.empty_cache()



        print("Finish training")
        print("Note that the printed training acc is not precise.",
              "To get precise training acc, use option ``test_train True``.")

        # show elapsed time
        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Time elapsed: {elapsed}")

        # save model
        self.save_model(cfg.output_dir)

        score_lst = self.test()

        # Close writer
        self._writer.close()

        return score_lst, self.head.scale_1, self.head.scale_2
    

    @torch.no_grad()
    def test(self, mode="test"):
        if self.tuner is not None:
            self.tuner.eval()
        if self.head is not None:
            self.head.eval()
        
        # if self.feat_center is not None:
        #     self.feat_center.eval()
        self.evaluator.reset()

        if mode == "train":
            print(f"Evaluate on the train set")
            data_loader = self.train_test_loader
        elif mode == "test":
            print(f"Evaluate on the test set")
            data_loader = self.test_loader

        self.total_logits = torch.empty((0, self.num_classes)).cuda()
        self.total_labels = torch.empty(0, dtype=torch.long).cuda()
        self.total_sim = torch.empty((self.num_classes)).cuda()# size (num_cls, )
        for batch in tqdm(data_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            _bsz, _ncrops, _c, _h, _w = image.size()
            image = image.view(_bsz * _ncrops, _c, _h, _w)

            if _ncrops <= 5:
                output, _ = self.model(image, init_text=self.mean_text_feat, test=True)
                output = output.view(_bsz, _ncrops, -1).mean(dim=1)
                
              
            else:
                # CUDA out of memory
                output = []
                image = image.view(_bsz, _ncrops, _c, _h, _w)
                for k in range(_ncrops):
                    output.append(self.model(image[:, k], test=True)[-1])
                output = torch.stack(output).mean(dim=0)


            self.total_logits = torch.cat((self.total_logits, output))
            self.total_labels = torch.cat((self.total_labels, label))

            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()


        for k, v in results.items():
            tag = f"test/{k}"
            if self._writer is not None:
                self._writer.add_scalar(tag, v)

        return [results["many_acc"], results["med_acc"], results["few_acc"], results["mean_acc"]]
    

    def save_model(self, directory, token=None):
        tuner_dict = self.tuner.state_dict()
        head_dict = self.head.state_dict()

        checkpoint = {
            "tuner": tuner_dict,
            "head": head_dict
        }


        for key in ["tuner", "head"]:
            state_dict = checkpoint[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[7:]
                new_state_dict[k] = v
            checkpoint[key] = new_state_dict


        
        if token is not None:
            save_path = os.path.join(directory, "checkpoint_"+str(token)+".pth.tar")
        else:
            save_path = os.path.join(directory, "checkpoint.pth.tar")

        # save model
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")


    def load_model(self, directory):
        load_path = os.path.join(directory, "checkpoint.pth.tar")

        if not os.path.exists(load_path):
            raise FileNotFoundError('Checkpoint not found at "{}"'.format(load_path))

        checkpoint = torch.load(load_path, map_location=self.device)
        tuner_dict = checkpoint["tuner"]
        head_dict = checkpoint["head"]


        print("Loading weights to from {}".format(load_path))
        self.tuner.load_state_dict(tuner_dict, strict=False)

        if head_dict["weight"].shape == self.head.weight.shape:
            self.head.load_state_dict(head_dict, strict=False)


