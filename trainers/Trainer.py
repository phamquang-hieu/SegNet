import torch
from torchvision.utils import make_grid
from torchvision import transforms
import json
from comet_ml import ExistingExperiment

class Trainer():
    def __init__(self, model, optimizer, lr_scheduler, loss, args, resume:str, train_loader, valid_loader, logger=None):
        self.model = model
        self.loss = loss
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimzer = optimizer
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.logger = logger
        self.start_epoch = 1
        
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
            print(f'[Epoch {epoch}/{self.args.num_epoch}] [Batch {(i+1)/len(self.train_loader)}] {loss}')
            
    def _valid_epoch(self, epoch):
        self.model.eval()
        class_IoU = torch.cuda.FloatTensor(0) # it will soon be a 1D tensor after addition
        valid_len = 0
        for i, (image, target) in enumerate(self.valid_loader):
            output_valid = self.model(image)
            
            metrics = self._eval_metrics(ouput_valid, target)
            epoch_mIou += metrics[0]
            class_IoU += metrics[1]
            valid_len += image.shape[0]
            
        class_IoU /= valid_len
        epoch_mIoU /= valid_len
        class_IoU = class_IoU.cpu().numpy()
        epoch_mIoU = epoch_mIoU.cpu().numpy()
        self.logger.log_metric("mIoU", epoch_mIoU, step=epoch)
        self.logger.log_metric("classIoU", class_IoU, step=epoch)

        return epoch_mIoU, class_IoU
        
    def train(self):
        print("---start-training--")
        for epoch in range(self.start_epoch, self.args.num_epoch+1):
            self._train_epoch(epoch)
            
            if (epoch % self.args.eval_freq == 0):
                best_mIoU = -1
                mIoU, cIoU = self._valid_epoch(epoch)
                
                if mIoU > best_mIoU:
                    self.best_mIoU = mIoU
                    self.best_cIoU = cIoU
                    self._save_checkpoint(epoch, save_best=True)
                else:
                    self._save_checkpoint(epoch, save_best=False)
            
            self.lr_scheduler.step()
    
    def _eval_metrics(self, output:torch.cuda.FloatTensor, target:torch.cuda.FloatTensor):
        # return a dictionary with keys being name of corresponding metrics.
        assert len(output.shape) == len(target.shape) == 4, "_eval_metrics expect input of size 4"
        
        intersection = output.logical_and(target).sum(dim=(2,3), dtype=torch.float16).mean(dim=0)
        union = output.logical_or(target).sum(dim=(2,3), dtype=torch.float16).mean(dim=0)
        IoU = (intersection/union) # of type Tensor
        
        mIoU = IoU.mean()
        
        return mIoU, IoU
    
    def infer(self):
        pass
    
    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch, 
            'model': self.model.state_dict(),
            'optimzer': self.optimizer.state_dict(),
            'lr_scheduler':self.lr_scheduler.state_dict(),
            'best_IoU': self.best_mIoU,
            'best_cIoU':self.best_cIoU
        }
        
        if save_best:
            save_path = f"./checkpoints/checkpoint-best.pth"
        else:
            save_path = f"./checkpoints/checkpoint.pth"
        torch.save(save_path)
        
    def _resume_checkpoint(self, resume_path):
        state = torch.load(resume_path)
        self.start_epoch = state['epoch']
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.lr_scheduler.load_state_dict(state['lr_scheduler'])
        self.best_IoU = state['best_IoU']
        self.best_cIoU = state['best_cIoU']
