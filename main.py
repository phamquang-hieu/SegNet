from trainers.Trainer import Trainer
from networks.segnet import SegNet
from networks import UnetDecoder
from datasets.CamVid import CamVid
from torch.utils.data import DataLoader
import argparse
import torch
import torch.nn as nn
# import torchvision.transforms as transforms
import albumentations as A
from albumentations.augmentations.geometric.resize import Resize
import random
from comet_ml import Experiment
import json
from utils.schedulers import OneCycle
from transformers import SegformerForSemanticSegmentation
from huggingface_hub import cached_download, hf_hub_url
from comet_ml import ExistingExperiment
from torch.optim.lr_scheduler import ExponentialLR
import json

def main(args, logger):
    transform_train = A.Compose([
                                  A.OneOf([
                                      A.RandomSizedCrop(min_max_height=(576, 684), height=720, width=960, p=0.5),
                                      A.PadIfNeeded(min_height=720, min_width=960, p=0.5)
                                  ], p=1),    
                                  A.VerticalFlip(p=0.5),              
                                  A.RandomRotate90(p=0.5),
                                  A.OneOf([
                                      A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                                      A.GridDistortion(p=0.5),
                                      A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)                  
                                      ], p=0.8),
                                  A.CLAHE(p=0.8),
                                  A.RandomBrightnessContrast(p=0.8),    
                                  A.RandomGamma(p=0.8),
                                  Resize(height=360, width=480)
                                ]
                                )
    transform_test = A.Compose([
        Resize(height=360, width=480)
    ])
    # transform = None
    train_loader = DataLoader(CamVid(mode='train', transform=transform_train), batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(CamVid(mode='valid', transform=transform_test), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(CamVid(mode='test', transform=transform_test), batch_size=args.batch_size, shuffle=False)
    
    with open("/content/SegNet/datasets/CamVid/id2label.json") as f:
        id2label = json.load(f)
    label2id = {v: k for k, v in id2label.items()}
    # model = SegNet(args.num_classes)
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0",
                                                         num_labels=args.num_classes,
                                                         id2label=id2label,
                                                         label2id=label2id)
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    model.decode_head = UnetDecoder(num_labels=args.num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00006)

    # lr_scheduler = OneCycle(optimizer, num_epochs=args.num_epoch, iters_per_epoch=1, phase1=15/args.num_epoch)
    lr_scheduler = ExponentialLR(optimizer, gamma=1)
    # [0,  0.28457743, 0.17831436, 4.13987536, 0.14145816, 0.57983627, 0.39328795, 3.74674816, 2.5740319 , 1., 6.31815479, 8.99454291]
    # [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    loss = nn.CrossEntropyLoss(reduction='none', weight=torch.cuda.FloatTensor([0,  0.28457743, 0.17831436, 4.13987536, 0.14145816, 0.57983627, 0.39328795, 3.74674816, 2.5740319 , 1., 6.31815479, 8.99454291]), ignore_index=0) 

    loss.cuda()
    model.cuda()
    
    trainer = Trainer(model=model, 
                      optimizer=optimizer, 
                      lr_scheduler=lr_scheduler, 
                      loss = loss, 
                      args = args,
                      resume=args.resume,
                      train_loader=train_loader,
                      valid_loader=valid_loader, 
                      logger=logger
                    )
    print("stay-tuned")
    trainer.train()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', "--resume", default="", help='path to the directory to restore ')
    parser.add_argument('-dr', "--drive_mounted", default=True, help="whether or not connected to google drive")
    parser.add_argument('-name', "--name", type=str, default="segnet")
    parser.add_argument('-n', "--num_epoch", type=int, default=120, help='number of epoch')
    parser.add_argument('-bs', "--batch_size", type=int, default=4, help='batch size')
    parser.add_argument('-lr', "--learning_rate", type=float, default=1e-3, help='learning rate')
    parser.add_argument('-mmt', "--momentum", type=float, default=0.9, help='momentum')
    parser.add_argument('-n_c', "--num_classes", type=int, default=11, help='number of classes in the segmentation problem')
    parser.add_argument('-eval', "--eval_freq", type=int, default=1, help='frequency of evaluation (epoch)')
    parser.add_argument('-a', '--a_focal', type=float, default=0.25, help="alpha in focal loss")
    parser.add_argument('-gm', '--gamma', type=float, default=2, help='eponential part of focal loss')
    args = parser.parse_args()
    
    if args.resume:
        with open("../../content/drive/MyDrive/ComputerVision/{}.json".format(args.name)) as f:
            EXPERIMENT_KEY = json.load(f)
            logger = ExistingExperiment(api_key="zZTzevPBE5M14bjosVgWeyg3u",
                                                            previous_experiment=EXPERIMENT_KEY)
    else:
        # Create an experiment with your api key
        logger = Experiment(
            api_key="zZTzevPBE5M14bjosVgWeyg3u",
            project_name="semanticsegmentation",
            workspace="phamquang-hieu",
        )
        logger.set_name(args.name)
        with open("/content/drive/MyDrive/ComputerVision/{}.json".format(args.name), "w") as f:
            json.dump(logger.get_key(), f)
    main(args, logger)
