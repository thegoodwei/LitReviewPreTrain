import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the fine-tuned BioMedLM model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("fine_tuned_BioMedLM")
model = AutoModelForCausalLM.from_pretrained("fine_tuned_BioMedLM")

# Function to generate text based on a prompt
def generate_text(prompt, max_length=1000, num_return_sequences=1):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        output_sequences = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            no_repeat_ngram_size=3,
        )
    return [tokenizer.decode(sequence) for sequence in output_sequences]

# Define the research data and user prompts
research_data = """
Here is your research data...
"""

user_prompts = [
    "Discuss the impact of the research on the field of study.",
    "Explain the methodology used in the research.",
    "Analyze the limitations of the research.",
]

# Generate the literature review
literature_review = ""
for prompt in user_prompts:
    # Combine research data and user prompt
    full_prompt = research_data + prompt
    # Generate text using the fine-tuned model
    generated_text = generate_text(full_prompt, max_length=200)[0]
    # Add generated text to the literature review
    literature_review += generated_text + "\n\n"

# Print the literature review
print(literature_review)
