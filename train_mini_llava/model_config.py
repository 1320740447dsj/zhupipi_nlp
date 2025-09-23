from transformers import  PretrainedConfig

class VLMConfig(PretrainedConfig):
    model_type = "vlm_model"

    def __init__(self, llm_model_path='./Qwen2.5-0.5B-Instruct',
                 vision_model_path='./siglip-base-patch16-224',
                 freeze_vision_model=True,
                 image_pad_num=49,
                 **kwargs):
        self.vision_model_path = vision_model_path
        self.llm_model_path = llm_model_path
        self.freeze_vision_model = freeze_vision_model
        self.image_pad_num = image_pad_num
        super().__init__(**kwargs)
