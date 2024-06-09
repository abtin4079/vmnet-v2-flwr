from utils.metrics import *





def test_server(model, criterion, testloader, device: str):
    """Validate the network on the entire test set. """
    total_loss = 0
    metrics = [0,0,0,0,0]
    for step, data in enumerate(testloader):
        img, img_sail, mask = data['img'], data['img_sail'], data['mask']
        img = img.to(device)
        img_sail = img_sail.to(device)
        mask = mask.to(device)

        # test_step 

        model.eval()
        pred_mask = model(data['img'], data['img_sail'])
        loss = criterion(pred_mask, data['mask']).item()

        # metrics
        IOU = intersection_over_union(pred_mask, data['mask'])
        acc = accuracy(pred_mask, data['mask'])
        F1, recall, precision = f1_score(pred_mask, data['mask'])

        metric = [IOU , F1 , acc, recall, precision]

        metrics = [sum(x) for x in zip(metrics, metric)]
        total_loss += loss
    
    total_loss = total_loss / len(testloader)
    metrics = [metric / len(testloader) for metric in metrics]

    return total_loss , metrics