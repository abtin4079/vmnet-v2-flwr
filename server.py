from collections import OrderedDict
from hydra.utils import instantiate
from omegaconf import DictConfig
from test import test_server
from train import TrainTestPipe
import torch

from model import test

def get_on_fit_config(config: DictConfig):
    """Return function that prepares config to send to clients."""
    def fit_config_fn(server_round: int):
        # This function will be executed by the strategy in its
        # `configure_fit()` method.

        # Here we are returning the same config on each round but
        # here you might use the `server_round` input argument to
        # adapt over time these settings so clients. For example, you
        # might want clients to use a different learning rate at later
        # stages in the FL process (e.g. smaller lr after N rounds)

        return {
            "lr": config.lr,
            "momentum": config.momentum,
            "local_epochs": config.local_epochs,
        }

    return fit_config_fn

def get_evaluate_fn(model_cfg, testloader):
    """Define function for global evaluation on the server."""
    def evaluate_fn(server_round: int, parameters, config):
        # Determine the device to use
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initialize the model using the TrainTestPipe
        ttp = TrainTestPipe(block_num=model_cfg.block_num,
                            class_num=model_cfg.class_num, 
                            device=device,
                            lr=model_cfg.lr,
                            momentum=model_cfg.momentum,
                            head_num=model_cfg.head_num,
                            img_dim=model_cfg.img_dim,
                            in_channels=model_cfg.in_channels,
                            mlp_dim=model_cfg.mlp_dim, 
                            model_path=model_cfg.model_path, 
                            out_channels=model_cfg.out_channels,
                            patch_dim=model_cfg.patch_dim, 
                            weight_decay=model_cfg.weight_decay)
        
        model = ttp.transunet.model.to(device)
        criterion = ttp.transunet.criterion

        # Ensure that the parameters match the model's state dict keys
        try:
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            
            # Verify if the state_dict is correctly populated
            if not state_dict:
                raise ValueError("State dictionary is empty. Ensure parameters are correctly passed.")
            
            # Load the state dict into the model
            model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            print(f"Error loading state_dict: {e}")
            print(f"State dictionary keys: {state_dict.keys()}")
            raise e

        # Evaluate the model using the provided test_server function
        loss, metrics = test_server(model,criterion ,testloader, device)

        # Return the loss and the metrics as a dictionary
        return loss, {
            "IOU": metrics[0],
            "F1": metrics[1],
            "accuracy": metrics[2],
            "recall": metrics[3],
            "precision": metrics[4]
        }

    return evaluate_fn
