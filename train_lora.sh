#! /bin/bash

export MODEL_PATH="./dataroot/models/runwayml/stable-diffusion-v1-5"
export DATASET_PATH="./struct"
export OUTPUT_PATH="./output"

accelerate launch --mixed_precision="fp16" train_lora.py \
  --pretrained_model_name_or_path=$MODEL_PATH \
  --dataset_name=$DATASET_PATH \
  --output_dir=$OUTPUT_PATH \
  --caption_column="text" \
  --resolution=512 \
  --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=100 \
  --checkpointing_steps=5000 \
  --learning_rate=1e-04 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --seed=42 \

  --validation_prompt="High house in intensity 8.0"
