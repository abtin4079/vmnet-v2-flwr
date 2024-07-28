from collections import OrderedDict
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar

import torch
import flwr as fl
from train import TrainTestPipe
from hydra.utils import instantiate


from train_vmunet import training_preprocess, train_vmunet

class FlowerClient(fl.client.NumPyClient):
    """Define a Flower Client."""

    def __init__(self,
                trainloader,
                vallodaer,
                model_cfg, 
                cid) -> None:
        super().__init__()

        # the dataloaders that point to the data associated to this client
        self.model_cfg = model_cfg
        self.trainloader = trainloader
        self.valloader = vallodaer
        self.cid = cid
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        # Initializing the train test pipeline 
        # self.ttp = TrainTestPipe(block_num=model_cfg.block_num,
        #                          class_num=model_cfg.class_num, 
        #                          device=self.device,
        #                          lr=model_cfg.lr,
        #                          momentum=model_cfg.momentum,
        #                          head_num=model_cfg.head_num,
        #                          img_dim=model_cfg.img_dim,
        #                          in_channels=model_cfg.in_channels,
        #                          mlp_dim=model_cfg.mlp_dim, 
        #                          model_path=model_cfg.model_path, 
        #                          out_channels=model_cfg.out_channels,
        #                          patch_dim=model_cfg.patch_dim, 
        #                           weight_decay=model_cfg.weight_decay)

        # self.model = self.ttp.transunet.model

        #self.model = instantiate(model_cfg.model_config)
        
        # figure out if this client has access to GPU support or not

        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model , self.resume_model, self.chechpoint_dir, self.logger, self.writer, self.work_dir = training_preprocess(config=self.model_cfg, cid=self.cid)
        

################################################################## federated learning ##########################################################
    def set_parameters(self, parameters):
        """Receive parameters and apply them to the local model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)

        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Extract model parameters and return them as a list of numpy arrays."""

        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """Train model received by the server (parameters) using the data.

        that belongs to this client. Then, send it back to the server.
        """

        # copy parameters sent by the server into client's local model
        self.set_parameters(parameters)

        # fetch elements in the config sent by the server. Note that having a config
        # sent by the server each time a client needs to participate is a simple but
        # powerful mechanism to adjust these hyperparameters during the FL process. For
        # example, maybe you want clients to reduce their LR after a number of FL rounds.
        # or you want clients to do more local epochs at later stages in the simulation
        # you can control these by customizing what you pass to `on_fit_config_fn` when
        # defining your strategy.



        lr = config["lr"]
        momentum = config["momentum"]
        weight_decay = config["weight_decay"]
        epochs = config["local_epochs"]



        # a very standard looking optimizer

        # optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        # do local training. This function is identical to what you might
        # have used before in non-FL projects. For more advance FL implementation
        # you might want to tweak it but overall, from a client perspective the "local
        # training" can be seen as a form of "centralized training" given a pre-trained
        # model (i.e. the model received from the server)

        train_vmunet(model_cfg=self.model_cfg,
                     model=self.model,
                     epochs= 1,
                     trainloader=self.trainloader, 
                     valloader=self.valloader, 
                     resume_model=self.resume_model, 
                     checkpoint_dir=self.chechpoint_dir, 
                     logger=self.logger, 
                     writer=self.writer, 
                     work_dir=self.work_dir
                     )

        # self.ttp.train(train_loader=self.trainloader, test_loader=self.valloader, epoch=self.model_cfg.epochs, patience=self.model_cfg.patience)
        
        #train(self.model, self.trainloader, optim, epochs, self.device)

        # Flower clients need to return three arguments: the updated model, the number
        # of examples in the client (although this depends a bit on your choice of aggregation
        # strategy), and a dictionary of metrics (here you can add any additional data, but these
        # are ideally small data structures)
        return self.get_parameters({}), len(self.trainloader), {}

    # def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
    #     self.set_parameters(parameters)

    #     loss, accuracy = test(self.model, self.valloader, self.device)

    #     return float(loss), len(self.valloader), {"accuracy": accuracy}

################################################################################################################################################
def generate_client_fn(trainloaders, valloaders, model_cfg):
    """Return a function that can be used by the VirtualClientEngine.

    to spawn a FlowerClient with client id `cid`.
    """

    def client_fn(cid: str):
        # This function will be called internally by the VirtualClientEngine
        # Each time the cid-th client is told to participate in the FL
        # simulation (whether it is for doing fit() or evaluate())

        # Returns a normal FLowerClient that will use the cid-th train/val
        # dataloaders as it's local data.
        return FlowerClient(
            trainloader=trainloaders[int(cid)],
            vallodaer=valloaders[int(cid)],
            model_cfg=model_cfg,  
            cid= int(cid)
        )

    # return the function to spawn client
    return client_fn
