from openai import OpenAI
client = OpenAI(base_url="http://0.0.0.0:8000/v1", api_key="not-used")
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What is in this image?"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                }
            }
        ]
    }
]
chat_response = client.chat.completions.create(
    model="meta/llama-3.2-11b-vision-instruct",
    messages=messages,
    max_tokens=256,
    stream=False
)
assistant_message = chat_response.choices[0].message
print(assistant_message)



## alternative to pass images is to encode images into base64 format then send messages along with other params
'''
{
    "type": "image_url",
    "image_url": {
        "url": "data:image/jpeg;base64,SGVsbG8gZGVh...ciBmZWxsb3chIQ=="
    }
}
with open("image.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()
'''