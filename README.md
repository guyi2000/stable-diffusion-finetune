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

