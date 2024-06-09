from tqdm import tqdm
import torch 
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import os
# Additional Scripts
from kvasir import transforms as T
from utils.utils import EpochCallback


from train_transunet import TransUNetSeg
import matplotlib.pyplot as plt

class TrainTestPipe:
    def __init__(self,model_path, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim, class_num, device, lr, momentum, weight_decay):
        self.device = device
        self.model_path = model_path

        self.transunet = TransUNetSeg(img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim, class_num, device, lr, momentum, weight_decay)


    def __loop(self, loader, step_func, t):
        total_loss = 0
        metrics = [0,0,0,0,0]
        for step, data in enumerate(loader):
            img, img_sail, mask = data['img'], data['img_sail'], data['mask']
            img = img.to(self.device)
            img_sail = img_sail.to(self.device)
            mask = mask.to(self.device)

            loss, cls_pred , metric = step_func(img=img, img_sail=img_sail, mask=mask)
            metrics = [sum(x) for x in zip(metrics, metric)]
            total_loss += loss

            t.update()

        return total_loss , metrics

    def train(self, train_loader, test_loader, epoch, patience):
        # Load pre-trained model weights before starting training
        # if os.path.exists(self.model_path):
        #     self.transunet.load_model(self.model_path)  

        # # Freeze the weights of the earlier layers, if desired
        # for param in self.transunet.model.parameters():
        #     param.requires_grad = True
        # for param in self.transunet.model.fc.parameters():
        #     param.requires_grad = True    

        # num_features = self.transunet.model.fc.in_features
        # self.transunet.model.fc = nn.Linear(num_features, cfg.transunet.class_num)


        train_loss_plot = []
        train_acc_plot = []
        test_loss_plot = []
        test_aac_plot = []
        
        callback = EpochCallback(self.model_path, epoch,
                                 self.transunet.model, self.transunet.optimizer, 'test_loss', patience)

        train_loss_plot.append(1)
        test_loss_plot.append(1)

        train_acc_plot.append(0)


        for epoch in range(epoch):
            with tqdm(total=len(train_loader) + len(test_loader)) as t:
                train_loss ,  metrics = self.__loop(train_loader, self.transunet.train_step, t)

                test_loss = self.__loop(test_loader, self.transunet.test_step, t)

            callback.epoch_end(epoch + 1,
                               {'train_loss': train_loss / len(train_loader),
                                'test_loss': test_loss[0] / len(train_loader), 
                                "IOU": metrics[0] / len(train_loader), 
                                "DSC": 1 -  train_loss / len(train_loader),
                                "F1-score": metrics[1] / len(train_loader), 
                                "accuracy": metrics[2] / len(train_loader), 
                                "recall": metrics[3] / len(train_loader), 
                                "precision": metrics[4] / len(train_loader)})

            train_loss_plot.append(train_loss / len(train_loader))
            test_loss_plot.append(test_loss[0] / len(train_loader))

            train_acc_plot.append(1 -  train_loss / len(train_loader))


            # Plot the training and testing losses
            plt.figure()  # Create a new figure to avoid overlap
            plt.plot(train_loss_plot, label='Train Loss')
            plt.plot(test_loss_plot, label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
    
            # Save the plot to the same file, overwriting the previous plot
            plt.savefig('/kaggle/working/plot1.png')
            plt.close()  # Close the figure to free memory      





            plt.figure()  # Create a new figure to avoid overlap
            plt.plot(train_acc_plot, label='dice similarity coefficient')
            plt.xlabel('Epochs')
            plt.ylabel('DSC')
            plt.legend()
    
            # Save the plot to the same file, overwriting the previous plot
            plt.savefig('/kaggle/working/plot2.png')
            plt.close()  # Close the figure to free memory 

            if callback.end_training:
                break

