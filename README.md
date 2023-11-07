# Stable Diffusion fine-tuning

This project shows how to fine-tune stable diffusion model on your own dataset.

> Note: This script is experimental. The script fine-tunes the whole model and often times the model overfits and runs into issues like catastrophic forgetting. It's recommended to try different hyperparamters to get the best result on your dataset.

## Prerequisites

Before running the scripts, make sure to install the library's training dependencies (such as [PyTorch](https://pytorch.org/) and [🤗Transformers](https://huggingface.co/docs/transformers/installation)):

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

Then run

```bash
cd ..
pip install -r requirements.txt
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

## Making your own datasets

To fine-tune stable diffusion model on your own dataset, you need to prepare your dataset in the following format:

In `dataset` directory, you should create three subdirectory `jpg`, `hint`, and `train`. In `jpg` directory, you should put all the target images in your dataset. In `hint` directory, you should put all the source images (condition images) in your dataset. In `train` directory, you should put a `metadata.jsonl`. The `metadata.jsonl` should be in the following format:

```json
{"jpg": "./dataset/jpg/<name_of_target_image_1>", "txt": "<prompt_1>", "hint": "./dataset/hint/<name_of_source_image_1>"}
{"jpg": "./dataset/jpg/<name_of_target_image_2>", "txt": "<prompt_2>", "hint": "./dataset/hint/<name_of_source_image_1>"}
…
```

### Dataset example

Here is the structure of the `dataset` directory:

```bash
.
├── hint
│   ├── a.png
│   ├── b.png
│   └── c.png
├── jpg
│   ├── a.png
│   ├── b.png
│   └── c.png
└── train
    └── metadata.jsonl
```

In `metadata.jsonl` file:

```json
{"jpg": "./dataset/jpg/a.png", "txt": "a", "hint": "./dataset/hint/a.png"}
{"jpg": "./dataset/jpg/b.png", "txt": "b", "hint": "./dataset/hint/b.png"}
{"jpg": "./dataset/jpg/c.png", "txt": "c", "hint": "./dataset/hint/c.png"}
```

## Training

### LoRA

To train LoRA model, run:

```bash
./train_lora.sh
```

You can change some hyperparameters in `run_lora.sh` file. For example, you can change `--num_train_epochs` to change the number of training epochs.

### ControlNet

To train ControlNet model, run:

```bash
./train_controlnet.sh
```

You can change some hyperparameters in `run_controlnet.sh` file. For example, you can change `--num_train_epochs` to change the number of training epochs.

### Train ControlNet and LoRA at the same time

To train ControlNet and LoRA at the same time, run:

```bash
./train_controlnet_and_lora.sh
```

Note that you should change the output directory of the ControlNet and LoRA model to start your own training.

### All in One

Just run:

```bash
./train_lora.sh && ./train_controlnet.sh && ./train_controlnet_and_lora.sh
```

You will get all the models in the `controlnet-lora-output` directory.

## Inference

You can change the path of the model and the condition image in `inference.py` file. Then run:

```bash
python inference.py
```

And you will get `output.png` in the root directory.

## Acknowledgements

This project is based on [diffusers](https://github.com/huggingface/diffusers) and it's examples.
