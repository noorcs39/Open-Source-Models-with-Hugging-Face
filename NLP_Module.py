from transformers import pipeline

# Use a regular text-generation or BlenderBot pipeline
pipe = pipeline("text2text-generation", model="facebook/blenderbot-400M-distill")

# Your input prompt
user_query = "What is Sukkur?"

# Get response
response = pipe(user_query)

# Print response
print(response[0]["generated_text"])
