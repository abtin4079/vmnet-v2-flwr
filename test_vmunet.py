from engine_vmunet import val_one_epoch
from hydra.utils import instantiate, call
from utils_vmunet import *
import tqdm
from sklearn.metrics import confusion_matrix
from omegaconf import DictConfig, OmegaConf


def testing_preprocess(config):
    print('#----------Prepareing Model for server----------#')
    model_cfg = config.model_config
    if config.network == 'vmunet-v2':
        # model = VMUNetV2(
        #     num_classes=model_cfg['num_classes'],
        #     input_channels=model_cfg['input_channels'],
        #     depths=model_cfg['depths'],
        #     depths_decoder=model_cfg['depths_decoder'],
        #     drop_path_rate=model_cfg['drop_path_rate'],
        #     #load_ckpt_path=model_cfg['load_ckpt_path'],
        #     load_ckpt_path=parser.pretrained_weight_path, # comment so we can use transfer learning
        #     deep_supervision = model_cfg['deep_supervision'],
        # )

        model = instantiate(model_cfg)

        model.load_from()    

        criterion = instantiate(config.criterion)

    else: raise Exception('network in not right!')

    return model, criterion


def test_vmunet(config, test_loader, model, criterion):
    model.eval()
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

    metrics = [accuracy, sensitivity, specificity, f1_or_dsc, miou]

    return np.mean(loss_list), metrics