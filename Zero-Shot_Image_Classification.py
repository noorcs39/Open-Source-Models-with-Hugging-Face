
# Visual Question Answering
from transformers import BlipForQuestionAnswering

model = BlipForQuestionAnswering.from_pretrained("./models/Salesforce/blip-vqa-base")
processor = AutoProcessor.from_pretrained("./models/Salesforce/blip-vqa-base")

image = Image.open("./beach.jpeg")

question = "how many dogs are in the picture?"
inputs = processor(image, question, return_tensors="pt")
out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))

# Zero-Shot Image Classification
from transformers import CLIPModel, AutoProcessor

model = CLIPModel.from_pretrained("./models/openai/clip-vit-large-patch14")
processor = AutoProcessor.from_pretrained("./models/openai/clip-vit-large-patch14")

image = Image.open("./kittens.jpeg")

labels = ["a photo of a cat", "a photo of a dog"]
inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)

probs = outputs.logits_per_image.softmax(dim=1)[0]
probs = list(probs)
for i in range(len(labels)):
    print(f"label: {labels[i]} - probability of {probs[i].item():.4f}")
