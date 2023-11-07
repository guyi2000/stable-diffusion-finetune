#! /bin/bash

export MODEL_PATH="./dataroot/models/runwayml/stable-diffusion-v1-5"
export DATASET_PATH="./dataset"
export CONTROLNET_PATH="./controlnet-output"
export LORA_PATH="./lora-output/pytorch_lora_weights.safetensors"
export OUTPUT_PATH="./controlnet-lora-output"

accelerate launch --multi_gpu train_controlnet_and_lora.py \
  --pretrained_model_name_or_path=$MODEL_PATH \
  --controlnet_model_name_or_path=$CONTROLNET_PATH \
  --lora_model_name_or_path=$LORA_PATH \
  --train_data_dir=$DATASET_PATH \
  --output_dir=$OUTPUT_PATH \
  --num_train_epochs=200 \
  --checkpointing_steps=500 \
  --resolution=512 \
  --learning_rate=5e-6 \
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
