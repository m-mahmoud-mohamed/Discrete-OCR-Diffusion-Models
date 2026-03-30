import os
from llava.model.builder import load_pretrained_model

from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch
from llava.model.language_model.llada.generate import generate as llada_generate
from llava.model.language_model.llada.log_likelyhood import get_logits as llada_get_logits
import json
import time
pretrained = "lavida-ckpts/lavida-llada-hd"
model_name = "llava_llada"
device = "cuda"
device_map = "cuda:0"

conv_template = "llada" 
question = DEFAULT_IMAGE_TOKEN + "\nDescribe the image in detail."
# question = DEFAULT_IMAGE_TOKEN + "\nExtract all text from this image in natural reading order. Preserve paragraph, heading, list, table, and formula structure as best as possible. Do not describe images without text. Output clean plain text or Markdown."


conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()
print(prompt_question)
vision_kwargs = None
vision_kwargs = dict(
    mm_vision_tower="/mnt/lustre-grete/projects/nii00224/mahmoud/LaViDa/lavida-ckpts/google-siglip-so400m-patch14-384",
    mm_resampler_type=None,
    mm_projector_type='mlp2x_gelu',
    mm_hidden_size=1152,
    use_mm_proj=True
)
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map,vision_kwargs=vision_kwargs,torch_dtype='bfloat16') # Add any other thing you want to pass in llava_model_args

model.eval()
model.tie_weights()
model.to(torch.bfloat16)


image = Image.open('/mnt/lustre-grete/projects/nii00224/mahmoud/LaViDa/data/olmocr/images/ct-put-2024-03-16-18-40-34-c9a556cb-f08b-46f1-aaaf-6207e0e26f38-169.png').convert('RGB')
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.bfloat16, device=device) for _image in image_tensor]






input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size]
#warmup
_ = model.generate(
    input_ids,
    images=image_tensor,
    image_sizes=image_sizes,
    do_sample=False,
    temperature=0,
    max_new_tokens=64,
    block_length=64,
    step_ratio=1.0, # 32 steps
    tokenizer=tokenizer,
    prefix_lm=True,
    verbose=True,
)

t0 = time.time()
cont,hist = model.generate(
    input_ids,
    images=image_tensor,
    image_sizes=image_sizes,
    do_sample=False,
    temperature=0.1,
    max_new_tokens=64,
    block_length=64,
    step_ratio=0.5, # 32 steps
    tokenizer=tokenizer,
    prefix_lm=True,
    verbose=True,
    schedule='shift',
)
t1 = time.time()
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)


text_outputs = [text_output.lstrip('!') for text_output in text_outputs]
print(text_outputs)

print("Time taken for generation (s): ", t1-t0)


# print('---------hist-------')
# for i, v in enumerate(hist):
#     print(i,tokenizer.batch_decode(v, skip_special_tokens=False)[0].lstrip('!').replace("<|mdm_mask|>",'*'))