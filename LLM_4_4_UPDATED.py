import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import accelerate
import torch
print(torch.backends.mps.is_available())  # Should print: True
from huggingface_hub import login
login("")

device = "cuda"
# Load the data
df = pd.read_csv("400 data.csv")

# Load the model
model_name = "google/gemma-3-12b-it"
# I use this model instead of google/gemma-3-12b-it because my laptop RAM is not enough for running
# If your laptop has enough memory, you can still use google/gemma-3-12b-it
tokenizer = AutoTokenizer.from_pretrained(model_name)  # Load tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map=device, torch_dtype=torch.bfloat16
)  # Load model

# Define the few-shot prompt
few_shot_intro = """You're a helpful assistant trained to detect stance in Reddit posts. The possible stances are: [Favor, Neutral, Oppose, Irrelevant].

Here are some examples:

Post: "Abortion access is a fundamental part of healthcare."
Stance: Favor

Post: "I can see both sides and think we need more thoughtful conversations around abortion."
Stance: Neutral

Post: "Abortion ends a human life, and I believe we should protect the unborn."
Stance: Oppose

Post: "Just got a positive pregnancy test before my expected period and I'm super anxious."
Stance: Irrelevant

Now analyze the stance of this new post:
"""

# Set up the generation pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Create an output list
stances = []

# Loop through each post and generate stance
for post in tqdm(df["text"]):
    prompt = few_shot_intro + f'Post: "{post}"\nStance:'
    output = pipe(prompt, max_new_tokens=10, do_sample=True)[0]['generated_text']

    # Extract stance from the output
    extracted = output.split("Stance:")[-1].strip().split()[0]  # Grab just the stance label
    stances.append(extracted)

# Save results
df["stance"] = stances
df.to_csv("reddit_stance_output.csv", index=False)
print("Done! Output saved to reddit_stance_output.csv")
