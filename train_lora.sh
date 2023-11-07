#! /bin/bash

export MODEL_PATH="./dataroot/models/runwayml/stable-diffusion-v1-5"
export DATASET_PATH="./lora-dataset"
export OUTPUT_PATH="./lora-output"

accelerate launch --multi_gpu train_lora.py \
  --pretrained_model_name_or_path=$MODEL_PATH \
  --train_data_dir=$DATASET_PATH \
  --output_dir=$OUTPUT_PATH \
  --num_train_epochs=200 \
  --checkpointing_steps=500 \
  --resolution=512 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=100 \
  --validation_prompt="High house in intensity 8.0" \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 \
  --image_column="jpg" \
  --caption_column="txt" \
  --seed=42 \
  --random_flip \
  --rank=8
