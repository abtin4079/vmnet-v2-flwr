import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils_vmunet import save_imgs

def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    step,
                    logger, 
                    config,
                    writer):
    '''
    Train the model for one epoch
    '''
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Switch to train mode and move model to device
    model.train()
    model.to(device)

    loss_list = []

    for iter , data in enumerate(train_loader):
        step += 1  # Increment step outside the loop to ensure correct counting
        optimizer.zero_grad()

        # Unpack data and move to device
        img, msk = data['img'], data['mask']
        img = img.to(device)
        msk = msk.to(device)

        # Forward pass
        outputs = model(img)
        loss = criterion(outputs, msk)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Record loss
        loss_list.append(loss.item())

        # Log current learning rate
        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        # Log loss to TensorBoard
        writer.add_scalar('loss', loss.item(), global_step=step)

        # Print log information
        if step % config.print_interval == 0:
          log_info = f'train: epoch {epoch}, iter:{step}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
          print(log_info)
          logger.info(log_info)

    # Step the scheduler
    scheduler.step()
    
    return step


def val_one_epoch(test_loader,
                    model,
                    criterion, 
                    epoch, 
                    logger,
                    config,
                    val_data_name=None):
    # switch to evaluate mode

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

    model.eval()
    model.to(device)
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out = model(img)
                   
            loss = criterion(out, msk)

            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out) 

    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        # accuracy = accuracy_score(y_true, y_pre)
        # _, recall, precision = all_score(y_true, y_pre)
        # miou = intersection_over_union(y_true, y_pre)
        # dsc = dice_similarity_coefficient(y_true, y_pre)


        if val_data_name is not None:
            log_info = f'val_datasets_name: {val_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = f' val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                sensitivity: {sensitivity}, specificity: {specificity}'
        print(log_info)
        logger.info(log_info)

    else:
        log_info = f' val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)
    
    return np.mean(loss_list)


def test_one_epoch(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out = model(img)
            loss = criterion(out, msk)

            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out) 
            if i % config.save_interval == 0:
                save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)