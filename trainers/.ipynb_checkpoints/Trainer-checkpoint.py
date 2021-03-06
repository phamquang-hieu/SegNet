import torch
from torchvision.utils import make_grid
from torchvision import transforms
import json
import os
import numpy as np
from transformers import SegformerForSemanticSegmentation
from torch.nn.functional import interpolate
from utils.TverskyLoss import TverskyLoss

class Trainer():
    def __init__(self, model, optimizer, lr_scheduler, loss, args, resume:str, train_loader, valid_loader, logger=None):
        self.model = model
        self.loss = loss
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.logger = logger
        self.start_epoch = 1
        self.best_mIoU = -1
        self.best_cIoU = np.array([-1]*self.args.num_classes)
        self.cur_cIoU = np.array([-1]*self.args.num_classes)

        if resume:
            self._resume_checkpoint(resume)
            
        
    def _train_epoch(self, epoch):
        self.model.train()
        train_length = len(self.train_loader)

        for i, (image, target) in enumerate(self.train_loader):
            if isinstance(self.model, SegformerForSemanticSegmentation):
                output = self.model(image).logits
            else:
                output = self.model(image)
            
            if output.shape[2: 4] != target.shape[1:3]:
                if output.shape[2: 4] != target.shape[1:3]:
                    output = interpolate(output, size=(target.shape[1], target.shape[2]), mode='bilinear', align_corners=True)
            
            if self.args.loss == 'combined':
                loss = self.loss[0](output, target).mean()
                loss_1 = self.loss[1](output, target)
                if isinstance(self.loss[1], TverskyLoss) and epoch > 1:
                    loss_1 = loss_1.pow(self.args.gamma)
                loss += loss_1.mean()
            else:
                loss = self.loss(output, target) 

            # if np.sum(self.cur_cIoU > self.focal_IoU_threshold) > self.focal_IoU_classes:
            #     probs = torch.exp(-loss)
            #     loss *= self.args.a_focal*(1-probs).pow(self.args.gamma)
            
            if epoch > 10 and isinstance(self.loss, TverskyLoss): 
                loss = loss.pow(self.args.gamma)
            
            loss = loss.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # self.logger.log_metric("training-loss", loss, epoch=epoch, step=(epoch-1)*train_length+i+1)

            print(f'[Epoch {epoch}/{self.args.num_epoch}] [Batch {(i+1)}/{len(self.train_loader)}] {loss}')
        

        
    def _valid_epoch(self, epoch):
        self.model.eval()
        # class_IoU = torch.cuda.FloatTensor([0]*self.args.num_classes) # it will soon be a 1D tensor after addition
        class_IoU = 0
        valid_len = 0
        epoch_mIoU = 0
        num_valid_instances = len(self.valid_loader)
        for i, (image, target) in enumerate(self.valid_loader):
            if isinstance(self.model, SegformerForSemanticSegmentation):
                output_valid = self.model(image).logits
            else:
                output_valid = self.model(image)
            if output_valid.shape[2: 4] != target.shape[1:3]:
                output_valid = interpolate(output_valid, size=(target.shape[1], target.shape[2]), mode='bilinear', align_corners=True) 
            
            # with torch.no_grad():
            #     self.logger.log_metric("valid_loss", self.loss(output_valid, target).mean(), step=(epoch-1)*num_valid_instances+i+1, epoch=epoch)
            
            batch_class_IoU, num_samples_per_class = self._eval_metrics(output_valid, target)
            class_IoU += batch_class_IoU
            valid_len += num_samples_per_class
        print("num_instances_per_class", valid_len)
        class_IoU /= valid_len
        epoch_mIoU = class_IoU[1:].mean()
        print("epoch_mIoU", epoch_mIoU, epoch_mIoU.shape)
        self.cur_cIoU = class_IoU
        
        # self.logger.log_metric("mIoU", epoch_mIoU, step=epoch)
        # for idx, c in enumerate(class_IoU):
        #     self.logger.log_metric(f"class-{idx}-IoU", c, step=epoch)

        return epoch_mIoU, class_IoU
        
    def train(self):
        print("---start-training--")
        for epoch in range(self.start_epoch, self.args.num_epoch+1):
            self._train_epoch(epoch)
            if (epoch % self.args.eval_freq == 0):
                mIoU, c_IoU = self._valid_epoch(epoch)
                
                if mIoU > self.best_mIoU:
                    self.best_mIoU = mIoU
                    self.best_cIoU =  c_IoU
                    print("best cIoU", self.best_cIoU)
                    self._save_checkpoint(epoch, save_best=True)
                else:
                    self._save_checkpoint(epoch, save_best=False)   
            
            self.lr_scheduler.step()
    
    def _eval_metrics(self, output:torch.cuda.FloatTensor, target:torch.cuda.FloatTensor):
        # return a dictionary with keys being name of corresponding metrics.
        output = output.argmax(dim=1)
        assert len(output.shape) == len(target.shape) == 3, "_eval_metrics expect input of size 3"
        class_IoU = []
        num_samples_per_class = []
        for label in range(self.args.num_classes):
            output_label = (output==label).long()
            target_label = (target==label).long()
            
            label_intersection = output_label.logical_and(target_label).sum(dim=(1, 2))
            label_union = output_label.logical_or(target_label).sum(dim=(1, 2))
            # print("---------------------------")
            # print("label_union", label_union)
            # print("label_union==0", label_union==0)
            # print("label_union!=0", label_union!=0)
            # print("label_union!=0.sum()", (label_union!=0).sum())
            # print("---------------------------")
            num_samples_per_class.append((label_union != 0).sum().cpu().numpy())

            class_IoU.append(((label_intersection)/(label_union+1e-5)).cpu().numpy())   
        class_IoU = np.array(class_IoU)
        num_samples_per_class = np.array(num_samples_per_class)
        # batch_sum_mIoU = class_IoU.mean(axis=0).sum() # mean along the label dimension
        class_IoU = class_IoU.sum(axis=1)
        return class_IoU, num_samples_per_class    
    
    def drive_path(self, path):
        return os.path.join("../../content/drive/MyDrive/ComputerVision", path)

    def _save_checkpoint(self, epoch, save_best=False):
        model_dir = "checkpoints"
        state = {
            'epoch': epoch, 
            'model': self.model.state_dict(),
            'optimzer': self.optimizer.state_dict(),
            'lr_scheduler':self.lr_scheduler.state_dict(),
            'best_IoU': self.best_mIoU,
            'best_cIoU':self.best_cIoU
        }
        if save_best:
            save_path = f"{model_dir}/{self.args.name}/checkpoint-best.pth"
        else:
            save_path = f"{model_dir}/{self.args.name}/checkpoint.pth"
        if self.args.drive_mounted:
            save_path = self.drive_path(save_path)
            model_dir = self.drive_path(model_dir)
        os.makedirs(os.path.join(model_dir, self.args.name), exist_ok=True)
        torch.save(state, save_path)
        
    def _resume_checkpoint(self, resume_path):
        state = torch.load(resume_path)
        self.start_epoch = state['epoch']+1
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimzer'])
        self.lr_scheduler.load_state_dict(state['lr_scheduler'])
        self.best_IoU = state['best_IoU']
        self.best_cIoU = state['best_cIoU']

        