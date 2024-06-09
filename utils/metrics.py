import torch
from utils.utils import dice_loss

def intersection_over_union(pred, target):
    pred = torch.sigmoid(pred)
    
    pred_binary = (pred > 0.5).float()  # Convert probabilities to binary predictions
    target_binary = target.float()

    intersection = torch.sum(pred_binary * target_binary)
    union = torch.sum(pred_binary) + torch.sum(target_binary) - intersection

    epsilon = 1e-5
    iou = intersection / (union + epsilon)

    return iou.item()


def dice_similarity_coefficient(pred, target):
    dice_score = dice_loss(pred, target)
    DSC = 1 - dice_score
    return DSC.item()
    

def accuracy(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred)
    
    pred_binary = (pred > threshold).float()  # Convert probabilities to binary predictions
    target_binary = target.float()

    correct = torch.sum((pred_binary == target_binary).float())
    total = target.numel()

    accuracy = correct / total

    return accuracy.item()


def f1_score(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred)
    
    pred_binary = (pred > threshold).float()  # Convert probabilities to binary predictions
    target_binary = target.float()

    tp = torch.sum(pred_binary * target_binary)
    fp = torch.sum((1 - target_binary) * pred_binary)
    fn = torch.sum(target_binary * (1 - pred_binary))
    
    epsilon = 1e-5
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    return f1.item(), recall.item(), precision.item()






if __name__ == '__main__':
    # Example usage:
    y_true = torch.tensor([[0, 1, 1], [1, 0, 1]])
    y_pred = torch.tensor([[0, 1, 0], [1, 1, 1]])

    print("Intersection over Union:", intersection_over_union(y_true, y_pred))
    print("Dice Similarity Coefficient:", dice_similarity_coefficient(y_true, y_pred))
    print("F1 Score:", f1_score(y_true.flatten(), y_pred.flatten()))