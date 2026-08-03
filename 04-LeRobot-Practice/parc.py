from pathlib import Path
import inspect

import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


MODEL_DIR = Path(__file__).resolve().parent / "smolvla_libero_plus_baseline"


def main() -> None:
    model = SmolVLAPolicy.from_pretrained(str(MODEL_DIR))
    model.eval()

    print("=== signatures ===")
    print("select_action:", inspect.signature(model.select_action))
    print("forward:", inspect.signature(model.forward))
    print("predict_action_chunk:", inspect.signature(model.predict_action_chunk))

    print("\n=== action queue ===")
    print("has _action_queue:", hasattr(model, "_action_queue"))
    if hasattr(model, "_action_queue"):
        print("queue type:", type(model._action_queue))
        print("queue length:", len(model._action_queue))

    print("\n=== useful attributes ===")
    for name in [
        "config",
        "language_tokenizer",
        "reset",
        "select_action",
        "predict_action_chunk",
    ]:
        value = getattr(model, name, None)
        print(name, ":", type(value))

    print("\n=== config essentials ===")
    print("input_features:", model.config.input_features)
    print("output_features:", model.config.output_features)
    print("normalization_mapping:", model.config.normalization_mapping)
    print("max_state_dim:", model.config.max_state_dim)
    print("max_action_dim:", model.config.max_action_dim)
    print("chunk_size:", model.config.chunk_size)
    print("n_action_steps:", model.config.n_action_steps)
    print("empty_cameras:", model.config.empty_cameras)


if __name__ == "__main__":
    main()