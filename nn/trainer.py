from numpy import squeeze
import yaml
from shufflenetv2 import ShuffleNetModified
from dataset import ImageDataset
from torch.utils.data.dataloader import DataLoader
from data_prefetcher import DataPrefetcher
import torch
import os
import time
import cv2
import datetime
from tqdm import tqdm
from loguru import logger
from log import setup_logger
from misc import get_model_info, gpu_mem_usage, save_checkpoint, MeterBuffer
from torch.utils.tensorboard import SummaryWriter
class Trainer:
    def __init__(self, config):
        args = []
        with open(config, "r") as file:
            args = yaml.load(file,Loader=yaml.FullLoader)
        self.args = args
        
        self.start_epoch = 0
        self.max_iter = 0
        self.device = "cuda:0"
        self.max_epoch = int(args["epoch"])
        self.lr = float(args["lr"])
        self.batch_size = int(args["batch"])
        self.weight_decay = float(args["weight_decay"])
        self.print_interval = int(args["print_interval"])
        self.eval_interval = int(args["eval_interval"])
        self.data_type = torch.float16 if (args["data_type"] == "fp16") else torch.float32
        self.amp_training = True if (args["data_type"] == "fp16") else False
        self.act = args["act"]
        self.train_txt = args["train_txt"]
        self.val_txt = args["val_txt"]
        self.exp_dir = args["exp_dir"]
        self.num_workers = int(args["num_workers"])
        self.train_dataset = []
        self.val_dataset = []
        self.train_loader = []
        self.val_loader = []
        
        self.model = ShuffleNetModified(act=self.act)
        self.model.to(self.device)
        self.cls_loss = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr,weight_decay=self.weight_decay)
        
        setup_logger(self.exp_dir)
        
        
    def before_train(self):
        logger.info("Preparing dataset...")
        self.train_dataset = ImageDataset(self.train_txt)
        self.val_dataset = ImageDataset(self.val_txt)
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.batch_size,
                                       num_workers=self.num_workers,
                                       shuffle=True)
        self.val_loader = DataLoader(self.val_dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers)

        self.max_iter = len(self.train_loader)
        logger.info(
            "Model Summary: {}".format(get_model_info(self.model, (128,128)))
        )
        self.meter = MeterBuffer(window_size=self.print_interval)
        self.model.train()
        self.tblogger = SummaryWriter(self.exp_dir)
        logger.info("Starting...")
    
    def evaluate_and_save_model(self):
        logger.info("Evaluating... ")
        self.model.eval()
        
        gts = []
        preds = []
            
        with torch.no_grad():
            for inps, targets in tqdm(self.val_loader):
                inps = torch.unsqueeze(inps,1)
                inps = inps.to(self.device, self.data_type)
                targets = targets.to(self.data_type)
                outputs = []
                outputs = self.model(inps)
                gts.append(targets)
                preds.append(squeeze(outputs))

        gts = torch.cat(gts)
        preds = torch.cat(preds)
        self.tblogger.add_pr_curve(tag="val/PR Curve", labels=gts, predictions=preds, global_step=self.epoch + 1)
                
        self.model.train()
                
    def save_ckpt(self, ckpt_name, update_best_ckpt=False):
        save_model = self.model
        logger.info("Save weights to {}".format(self.exp_dir))
        ckpt_state = {
            "start_epoch": self.epoch + 1,
            "model": save_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        save_checkpoint(
            ckpt_state,
            update_best_ckpt,
            self.exp_dir,
            ckpt_name,
        )

    
    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()
    
    def before_epoch(self):
        logger.info("---> start train epoch{}".format(self.epoch + 1))
        self.prefetcher = DataPrefetcher(self.train_loader)
        
    def after_epoch(self):
        self.save_ckpt(ckpt_name="latest")
        self.tblogger.add_scalar("train/loss", self.loss, self.epoch + 1)
        if (self.epoch + 1) % self.eval_interval == 0:
            self.evaluate_and_save_model()

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            # self.before_iter()
            self.train_one_iter()
            self.after_iter()
            
    def train_one_iter(self):
        outputs = []
        iter_start_time = time.time()
        inps, targets = self.prefetcher.next()
        inps = torch.unsqueeze(inps,1)
        inps = inps.to(self.device, self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        data_end_time = time.time()

        with torch.cuda.amp.autocast(enabled=self.amp_training):
            outputs = self.model(inps)
        
        self.loss = self.cls_loss(squeeze(outputs), targets)
        cv2.waitKey(0)
        self.loss.backward() # 反向传播
        self.optimizer.step()
        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=self.lr,
            **{"loss":self.loss},
        )

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter
    
    
    def after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        
        
        if (self.iter + 1) % self.print_interval == 0:
            # TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.4f}".format(k, v.latest) for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            logger.info(
                "{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    gpu_mem_usage(),
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", {}".format(eta_str))
            )
            self.meter.clear_meters()

    def after_train(self):
        # logger.info(
        #     "Training of experiment is done and the best AP is {:.2f}".format(self.best_ap * 100)
        # )
        self.tblogger.close()
            