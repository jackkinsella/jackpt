import os
import json


def model_filename(n_embed: int, n_head: int, n_layer: int, block_size: int, num_steps: int) -> str:
    # Filename encodes the hyperparameters so different runs don't overwrite each other.
    # e.g. "models/jackpt_embed16_heads4_layers1_block16_steps1000.json"
    return f"models/jackpt_embed{n_embed}_heads{n_head}_layers{n_layer}_block{block_size}_steps{num_steps}.json"


def save_model(state_dict: dict, filename: str) -> None:
    # Serialise every weight's .data value to JSON.
    # We only save .data (the float), not the full Value graph — the graph is
    # rebuilt fresh each run and we just overwrite .data on load.
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    serialised = {
        key: [[cell.data for cell in row] for row in mat]
        for key, mat in state_dict.items()
    }
    with open(filename, 'w') as f:
        json.dump(serialised, f)
    print(f"model saved to {filename}")


def load_model(state_dict: dict, filename: str) -> None:
    # Read saved float values back and overwrite .data on each Value in place.
    # The Value graph structure (children, local_grads) is left untouched so
    # backprop still works correctly if you continue training after loading.
    with open(filename, 'r') as f:
        serialised = json.load(f)
    for key, mat in serialised.items():
        for row_idx, row in enumerate(mat):
            for col_idx, value in enumerate(row):
                state_dict[key][row_idx][col_idx].data = value
    print(f"model loaded from {filename}")
