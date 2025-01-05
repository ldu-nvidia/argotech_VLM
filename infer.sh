curl -X 'POST' \
'http://0.0.0.0:8000/v1/chat/completions' \
-H 'accept: application/json' \
-H 'Content-Type: application/json' \
-d '{
    "model": "meta/llama-3.2-11b-vision-instruct",
    "messages": [{"role":"user", "content":[
        {"type": "text", "text": "Describe this image"},
        {
          "type": "image_url",
          "image_url": {"url": "https://assets.ngc.nvidia.com/products/api-catalog/phi-3-5-vision/example1b.jpg"}
        }
    ]}],
    "max_tokens": 256
}'
