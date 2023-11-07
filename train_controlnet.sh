#! /bin/bash

export MODEL_PATH="./dataroot/models/runwayml/stable-diffusion-v1-5"
export DATASET_PATH="./dataset"
export OUTPUT_PATH="./controlnet-output"

accelerate launch --multi_gpu train_controlnet.py \
  --pretrained_model_name_or_path=$MODEL_PATH \
  --train_data_dir=$DATASET_PATH \
  --output_dir=$OUTPUT_PATH \
  --num_train_epochs=200 \
  --checkpointing_steps=500 \
  --resolution=512 \
  --learning_rate=1e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=100 \
  --validation_image "./dataset/hint/bjy_1_1_p1.png" "./dataset/hint/frst_T1_T45_T35.png" \
  --validation_prompt "High house in intensity 8.0" "Low house in intensity 8.0" \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 \
  --image_column="jpg" \
  --conditioning_image_column="hint" \
  --caption_column="txt" \
  --seed=42
