from openai import OpenAI
import os
import base64

class AgroVLMAgent:
    def __init__(self) -> None:
        self.client = OpenAI(base_url="http://0.0.0.0:8000/v1", api_key="not-used")
        self.prompts = ["you are an expert in identifying plant diseases, I will provide you with an image and i want you to answer following questions.", "can you tell what plant is this from the leaf?", "is the leaf healthy or not?", "what kind of disease do you think the leaf has?"]
        # directory to image folder https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset?resource=download
        # could also be streaming images
        self.image_directory = "/home/ldu/Repos/argotech_VLM/images/train/Apple___Apple_scab/"
        self.model = "meta/llama-3.2-11b-vision-instruct"
      

    # image files named in certain way, strip string to get ground truth plant and condition
    def stripPlantInfo(self, plant_string: str) -> tuple:
        plant, condition = plant_string.split("___")[0], plant_string.split("___")[1]
        return (plant, condition)
    
    # image need to be encoded to send as request
    def encodeImage(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()
        return image_b64


    def formMessage(self, role: str, text_prompt: str, image_b64: str):
        return  [
                    {
                        "role": role,
                        "content": [
                            {
                                "type": "text",
                                "text": text_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/jpeg;base64," + image_b64
                                }
                            }
                        ]
                    }
                ]

    def getVLMResponse(self, messages):
        chat_response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1024,
            stream=False
        )
        assistant_message = chat_response.choices[0].message
        print(assistant_message)
        return assistant_message

def main():
    vlm_agent = AgroVLMAgent()
    plant, disease = vlm_agent.stripPlantInfo(vlm_agent.image_directory[:-1].split("/")[-1])
    
    # example is the 100th image, change index for specific image or loop through a batch of images
    one_image = vlm_agent.image_directory + os.listdir(vlm_agent.image_directory)[100]
    print("Ground Truth: the plant is ", plant, " and the disease is ", disease, "\n")
    print("Image directory is: ", one_image)
    decoded_image = vlm_agent.encodeImage(one_image)
    
    for prompt_idx in range(len(vlm_agent.prompts)):
        prompt = vlm_agent.prompts[prompt_idx]
        message = vlm_agent.formMessage(role="user", text_prompt=prompt, image_b64=decoded_image)
        print("\n")
        print("PROMPT: ", prompt)
        vlm_agent.getVLMResponse(message)


if __name__ == "__main__":
    main()
