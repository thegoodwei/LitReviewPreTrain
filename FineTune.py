import os
import glob
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from datasets import load_dataset, DatasetBuilder

# Create a dataset script to load the article text files
class ArticleTextDataset(DatasetBuilder):
    def _info(self):
        return datasets.DatasetInfo(features=datasets.Features({"text": datasets.Value("string")}))

    def _split_generators(self, dl_manager):
        data_dir = 'articles'
        files = glob.glob(os.path.join(data_dir, "*.txt"))
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"files": files}),
        ]

    def _generate_examples(self, files):
        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                yield {"text": text}

# Register and load the dataset
dataset = load_dataset("article_text.py")
                
# Load the pre-trained BioMedLM model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("stanford-crfm/BioMedLM")
model = AutoModelForCausalLM.from_pretrained("stanford-crfm/BioMedLM")

# Tokenize the dataset
tokenized_dataset = dataset.map(lambda x: tokenizer(x['text']), batched=True)

# Create training and validation datasets
train_dataset = TextDataset(tokenized_dataset['train'], tokenizer.pad_token_id, block_size=tokenizer.model_max_length)

# Define data collator
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Configure training arguments
training_args = TrainingArguments(
    output_dir="output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=5e-5,
    weight_decay=0.01,
    save_total_limit=3,
)

# Create the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("fine_tuned_BioMedLM")
