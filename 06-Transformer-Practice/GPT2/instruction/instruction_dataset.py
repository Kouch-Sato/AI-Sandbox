from load_data import download_and_load_file 
from pathlib import Path

def format_input(entry):
    instructon_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instructon_text + input_text


file_path = Path(__file__).parent / "instruction-data.json"
url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)

data = download_and_load_file(file_path, url)
entry = data[0]
desired_response = f"\n\n### Response\n{entry['output']}"
print(format_input(entry) + desired_response)

num_data = len(data)
train_portion = int(num_data * 0.85)
test_portion = int(num_data * 0.10)
val_portion = int(num_data - train_portion - test_portion)

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

print(len(train_data))
print(len(test_data))
print(len(val_data))
