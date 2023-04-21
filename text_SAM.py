from  PIL  import  Image
from lang_sam import LangSAM

model = LangSAM()
image_pil = Image.open('/home/zhi/nas/PEAS/PEAS_images/OneDrive_6_23-03-2023/NGB106117_1_8_SSD_seed_1.png').convert("RGB")

text_prompt = 'peas'
masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)