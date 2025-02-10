from openai import OpenAI
import os
import base64

class AgroVLMAgent:
    def __init__(self) -> None:
        self.client = OpenAI(base_url="http://0.0.0.0:8000/v1", api_key="not-used")
        self.prompts = ["You are an expert in identifying plant diseases. I will provide you with an image of a plant leaf, and I want you to follow these steps to answer the questions",
                        
                        "Identify the Plant: Examine the leaf's shape, size, color, texture or any unique features that can help determine the plant species",
                        
                        "Assess the Leaf's Health: Look for signs of damage or discoloration like yellowing, browning, spots, or wilting. Consider the uniformity of the leaf's color and texture. Healthy leaves generally have vibrant, consistent color and smooth texture. Based on these visual clues, determine whether the leaf appears healthy or unhealthy", 
                        
                        "Diagnose the Disease: Analyze any abnormalities or symptoms such as spots, lesions, mold, or unusual patterns. Consider the possible causes for these symptoms, which could include fungal, bacterial, or viral infections, pests, or environmental stress. Provide a hypothesis about what kind of disease or condition the leaf may have. After completing each of these steps, provide your answers to the following questions:",
                        
                        "Can you tell what plant this is from the leaf? Is the leaf healthy or not? What kind of disease do you think the leaf has? Describe the disease"]
        
        # directory to image folder https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset?resource=download
        # could also be streaming images
        self.image_directory = "./images/train/Apple___Apple_scab/"
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
