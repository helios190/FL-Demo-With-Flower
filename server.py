from typing import Dict, Optional, Tuple
import flwr as fl
from utils import model, getMnistDataSet, plotServerData

model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

results_list = []

def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""
    x_train, y_train, x_test, y_test = getMnistDataSet()

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_test, y_test)
        print("After round {}, Global accuracy = {} ".format(server_round,accuracy))
        results = {"round":server_round,"loss": loss, "accuracy": accuracy}
        results_list.append(results)
        return loss, {"accuracy": accuracy}

    return evaluate

strategy = fl.server.strategy.FedAvg(evaluate_fn=get_eval_fn(model))

# Start Flower server for three rounds of federated learning
fl.server.start_server(
        config=fl.server.ServerConfig(num_rounds=21),
        strategy=strategy
    )

plotServerData(results_list)

