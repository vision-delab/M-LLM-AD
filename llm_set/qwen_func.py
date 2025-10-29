from qwen_vl_utils import process_vision_info


def qwen_make_messages_one_image(prompt, image):
    messages = [
        {
            "role": "system", 
            "content": "You are a time series anomaly detector."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text", 
                    "text": prompt
                },
            ],
        }
    ]
    return messages


def qwen_make_messages_images(prompt, image1, image2, image3=None):
    if image3 == None:
        messages = [
            {
                "role": "system", 
                "content": "You are a time series anomaly detector."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image1,
                    },
                    {
                        "type": "image",
                        "image": image2,
                    },
                    {
                        "type": "text", 
                        "text": prompt
                    },
                ],
            }
        ]
    else:
        messages = [
            {
                "role": "system", 
                "content": "You are a time series anomaly detector."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image1,
                    },
                    {
                        "type": "image",
                        "image": image2,
                    },
                    {
                        "type": "image",
                        "image": image3,
                    },
                    {
                        "type": "text", 
                        "text": prompt
                    },
                ],
            }
        ]
    return messages


def qwen_inference(model, processor, messages, device='cuda'):
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]
