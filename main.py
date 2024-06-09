from pathlib import Path
import pickle
import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from kvasir.dataset_kvasir import prepare_dataset_kvasir

from dataset import prepare_dataset
from client import generate_client_fn
from server import get_on_fit_config, get_evaluate_fn

import flwr as fl

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    ##1. parse config & get experiment output dir
    print(OmegaConf.to_yaml(cfg))
    ##2. prepare your dataset
    #trainloaders, validationloaders, testloader = prepare_dataset(cfg.num_clients, cfg.batch_size)
    
    trainloaders, validationloaders, testloader = prepare_dataset_kvasir(cfg.num_clients, cfg.batch_size, output_size=cfg.output_size)

    print(len(trainloaders), len(trainloaders[0].dataset))

    ##3. Define your clients

    #initialize the model for training
    
    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.model)

    ##4. Define your strategy



    # strategy = fl.server.strategy.FedAvg( 
    #     fraction_fit=0.0,  # in simulation, since all clients are available at all times, we can just use `min_fit_clients` to control exactly how many clients we want to involve during fit
    #     min_fit_clients=cfg.num_clients_per_round_fit,  # number of clients to sample for fit()
    #     fraction_evaluate=0.0,  # similar to fraction_fit, we don't need to use this argument.
    #     min_evaluate_clients=cfg.num_clients_per_round_eval,  # number of clients to sample for evaluate()
    #     min_available_clients=cfg.num_clients,  # total clients in the simulation
    #     on_fit_config_fn=get_on_fit_config(
    #         cfg.config_fit
    #     ),  # a function to execute to obtain the configuration to send to the clients during fit()
    #     evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader),
    # ) # a function to run on the server side to evaluate the global model.


    # implement with hydra config file

    strategy = instantiate(cfg.strategy,
                           on_fit_config_fn=get_on_fit_config(cfg.config_fit), 
                           evaluate_fn=get_evaluate_fn(cfg.model, testloader),
                           )

    ##5. start Simulation 
    history = fl.simulation.start_simulation(
        client_fn= client_fn,
        num_clients= cfg.num_clients,
        config = fl.server.ServerConfig(num_rounds=cfg.num_rounds), 
        strategy= strategy,
        client_resources = {'num_cpus': 2.0, "num_gpus": 1.0},
    )

    ##6. save your results
    save_path = HydraConfig.get().runtime.output_dir
    results_path = Path(save_path) / 'results.pkl'

    results = {'history': history, "anything else": "here"}

    with open(str(results_path), 'wb') as h : 
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()