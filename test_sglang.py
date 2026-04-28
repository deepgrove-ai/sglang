import openai

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")

# response = client.chat.completions.create(
#     model="/scratch/ansh/models/hf_checkpoint_3K",
#     messages=[
#         {"role": "user", "content": "List 3 countries and their capitals."},
#     ],
#     temperature=0,
#     max_tokens=64,
# )
# print(response.choices[0].message.content)

response = client.completions.create(
    model="/scratch/ansh/models/maple_reference_model",
    prompt="List 3 countries and their capitals.",
    temperature=0,
    max_tokens=64,
)

print(response.choices[0].text)
