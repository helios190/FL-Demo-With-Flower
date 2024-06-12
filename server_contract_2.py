from typing import Dict, Optional, Tuple
import flwr as fl
from utils import model, getMnistDataSet, plotServerData
from web3 import Web3
import json

# Connect to local Ethereum node
w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))

# Smart contract address and ABI
contract_address = "0x673De118fe9Fb764667f0F5259fDBC062920D7Cb"
contract_address = w3.to_checksum_address(contract_address)

with open('/Users/bintangrestubawono/Documents/fl7/FL-Demo-With-Flower/contracts/artifacts/contracts/FederatedLearning.sol/FederatedLearning.json') as f:
    contract_abi = json.load(f)["abi"]

contract = w3.eth.contract(address=contract_address, abi=contract_abi)

model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

results_list = []

def get_eval_fn(model):
    x_train, y_train, x_test, y_test = getMnistDataSet()
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        print("After round {}, Global accuracy = {} ".format(server_round, accuracy))
        results = {"round": server_round, "loss": loss, "accuracy": accuracy}
        results_list.append(results)

        # Check if the "client_ids" key is present in the config
        if "client_ids" in config:
            for client in config["client_ids"]:
                tx_hash = contract.functions.registerClient().transact({'from': client})
                w3.eth.waitForTransactionReceipt(tx_hash)

        return loss, {"accuracy": accuracy}
    return evaluate

strategy = fl.server.strategy.FedAvg(evaluate_fn=get_eval_fn(model))

fl.server.start_server(
    config=fl.server.ServerConfig(num_rounds=21),
    strategy=strategy
)

plotServerData(results_list)
