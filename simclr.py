import logging
import os
import sys
import numpy as np 
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint
import wandb 

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader, load=False, epoch_start=0, dataset=None): 
        length = len(train_loader) 
        
        best_loss = np.inf 
        
        wandb.init(project="HSpace-SAEs", entity="a-ijishakin",
                        name='SimCLR')  
        
        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        if load:
            checkpoint = torch.load(f'/home/rmapaij/sae_bench/SimCLR/model_best.pth.tar{dataset}')
            # Load the state_dict into the model
            self.model.load_state_dict(checkpoint['state_dict'])  
            epoch_start = checkpoint['epoch'] + 1 
            
        for epoch_counter in range(epoch_start, self.args.epochs): 
            epoch_loss = 0 
            with tqdm(total=len(train_loader), desc=f'Epoch {epoch_counter}') as pbar:
                for idx, data in enumerate(train_loader):
                    images = data['imgs'] 
                    images = torch.cat(images, dim=0)

                    images = images.to(self.args.device)

                    with autocast(enabled=self.args.fp16_precision):
                        features = self.model(images)
                        logits, labels = self.info_nce_loss(features)
                        loss = self.criterion(logits, labels)

                    self.optimizer.zero_grad()

                    scaler.scale(loss).backward()

                    scaler.step(self.optimizer)
                    scaler.update() 
                    
                    wandb.log({'loss': 
                        loss.item()}, 
                            step= (epoch_counter * length) + idx)  
                    epoch_loss += loss.item()
                    
                    
                    n_iter += 1 
                    pbar.update(1) 

                if epoch_loss < best_loss: 
                    best_loss = epoch_loss 
                    save_checkpoint({
                        'epoch': epoch_counter,
                        'arch': self.args.arch,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                    }, is_best=True,  dataset=dataset, filename=os.path.join(self.writer.log_dir, f'checkpoint-{dataset}.pth.tar')) 
                

                # warmup for the first 10 epochs
                if epoch_counter >= 10:
                    self.scheduler.step()

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}_{}.pth.tar'.format(self.args.epochs, dataset)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
