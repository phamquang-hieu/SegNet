import torch
from torchvision.utils import make_grid
from torchvision import transforms
import json
from comet_ml import ExistingExperiment
import os

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
        if resume:
            self._resume_checkpoint(resume)
            with open("/content/drive/MyDrive/ComputerVision/{}.json".format(args.name)) as f:
                EXPERIMENT_KEY = json.load(f)
            self.logger = ExistingExperiment(api_key="zZTzevPBE5M14bjosVgWeyg3u",
                                                            previous_experiment=EXPERIMENT_KEY)
        
    def _train_epoch(self, epoch):
        self.model.train()
        train_length = len(self.train_loader)
        for i, (image, target) in enumerate(self.train_loader):
            output = self.model(image)
            
            loss = self.loss(output, target)  # CELoss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.logger.log_metric("training-loss", loss, epoch=epoch, step=(epoch-1)*train_length+i+1)
            print(f'[Epoch {epoch}/{self.args.num_epoch}] [Batch {(i+1)}/{len(self.train_loader)}] {loss}')
            
    def _valid_epoch(self, epoch):
        self.model.eval()
        # class_IoU = torch.cuda.FloatTensor([0]*self.args.num_classes) # it will soon be a 1D tensor after addition
        class_IoU = 0
        valid_len = 0
        epoch_mIoU = 0
        for i, (image, target) in enumerate(self.valid_loader):
            output_valid = self.model(image)
            
            batch_sum_mIoU, batch_class_IoU = self._eval_metrics(output_valid, target)
            class_IoU += batch_class_IoU
            epoch_mIoU += batch_sum_IoU
            
            valid_len += image.shape[0]
            
        class_IoU /= valid_len
        epoch_mIoU /= valid_len
    
        self.logger.log_metric("mIoU", epoch_mIoU, step=epoch)
        for idx, c in enumerate(class_IoU):
            self.logger.log_metric(f"class-{idx}-IoU", c, step=epoch)

        return epoch_mIoU, class_IoU
        
    def train(self):
        print("---start-training--")
        for epoch in range(self.start_epoch, self.args.num_epoch+1):
            
            if (epoch % self.args.eval_freq == 0):
                mIoU = self._valid_epoch(epoch)
                
                if mIoU > self.best_mIoU:
                    self.best_mIoU = mIoU
                    self.best_cIoU =  None #cIoU
                    print("best cIoU", self.best_cIoU)
                    self._save_checkpoint(epoch, save_best=True)
                else:
                    self._save_checkpoint(epoch, save_best=False)
            self._train_epoch(epoch)
            
            self.lr_scheduler.step()
    
    def _eval_metrics(self, output:torch.cuda.FloatTensor, target:torch.cuda.FloatTensor):
        # return a dictionary with keys being name of corresponding metrics.
        output = output.argmax(dim=1)
        assert len(output.shape) == len(target.shape) == 3, "_eval_metrics expect input of size 3"
        class_IoU = []
        for label in range(self.args.num_classes):
            output_label = (output==label).long()
            target_label = (target==label).long()
            
            label_intersection = output_label.logical_and(target_label).sum(dim=(1, 2))
            label_union = output_label.logical_or(target_label).sum(dim=(1, 2))
            
            class_IoU.append(((label_intersection)/(label_union)).item())   
        class_IoU = np.array(class_IoU)
        batch_sum_mIoU = class_IoU.mean(axis=0).sum() # mean along the label dimension
        class_IoU = class_IoU.sum(axis=1)
        return batch_sum_mIoU, class_IoU    
    
    def infer(self):
        pass
    
    def drive_path(self, path):
        return os.path.join("content/drive/MyDrive/ComputerVision", path)

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
            save_path = f"{model_dir}/checkpoint-best.pth"
        else:
            save_path = f"{model_dir}/checkpoint.pth"
        if self.args.drive_mounted:
            save_path = self.drive_path(save_path)
            model_dir = self.drive_path(model_dir)
        os.makedirs(model_dir, exist_ok=True)
        torch.save(state, save_path)
        
    def _resume_checkpoint(self, resume_path):
        state = torch.load(resume_path)
        self.start_epoch = state['epoch']
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.lr_scheduler.load_state_dict(state['lr_scheduler'])
        self.best_IoU = state['best_IoU']
        self.best_cIoU = state['best_cIoU']
