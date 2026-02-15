#Example ECG dataset
ecg_dataset = [{"report": "PR interval 160ms, QRS duration 90ms, QT interval 400ms, heart rate 70 bpm. Normal sinus rhythm."},
               {"report": "PR interval 200ms, QRS duration 120ms, QT interval 450ms, heart rate 80 bpm. Sinus bradycardia."},
               {"report": "PR interval 180ms, QRS duration 100ms, QT interval 420ms, heart rate 75 bpm. Sinus tachycardia."},
               {"report": "PR interval 150ms, QRS duration 80ms, QT interval 380ms, heart rate 65 bpm. Normal sinus rhythm."},
               {"report": "PR interval 220ms, QRS duration 130ms, QT interval 470ms, heart rate 85 bpm. Sinus arrhythmia."}
               ]

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

#LoRa configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    init_lora_weights="gaussian",
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

#Apply LoRa to the base model
peft_model = get_peft_model(base_model, lora_config)

#Fine-Tune on ECG Dataset
from transformers import Trainer, TrainingArguments

# Tokenise dataset
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    tokens = tokenizer(
        examples["report"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

# Now you can tokenize with padding
tokenized_dataset = [tokenize_function(d) for d in ecg_dataset]



#Training arguments
training_args = TrainingArguments(
    output_dir="./ecg_lora",
    num_train_epochs=3,
    learning_rate=5e-4,
    logging_steps=1,
    save_strategy="no"
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()

#Create RAG Knowledge Base
import faiss
import numpy as np
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")


#Example Knowledge chunks
ecg_chunks = [
    "PR interval: 160ms, QRS duration: 90ms, QT interval: 400ms, heart rate: 70 bpm. Interpretation: Normal sinus rhythm.",
    "PR interval: 200ms, QRS duration: 120ms, QT interval: 450ms, heart rate: 80 bpm. Interpretation: Sinus bradycardia.",
    "PR interval: 180ms, QRS duration: 100ms, QT interval: 420ms, heart rate: 75 bpm. Interpretation: Sinus tachycardia.",
    "PR interval: 150ms, QRS duration: 80ms, QT interval: 380ms, heart rate: 65 bpm. Interpretation: Normal sinus rhythm.",
    "PR interval: 220ms, QRS duration: 130ms, QT interval: 470ms, heart rate: 85 bpm. Interpretation: Sinus arrhythmia."
]
#Generate embeddings
chunk_embeddings = []
for chunk in ecg_chunks:
    response = openai.embeddings.create(input=chunk,model="text-embedding-3-small")
    embedding = response.data[0].embedding
    chunk_embeddings.append(np.array(embedding))

#Create FAISS index
dimension = len(chunk_embeddings[0])
index = faiss.IndexFlatL2(dimension)
embedding_matrix = np.vstack(chunk_embeddings).astype("float32")
index.add(embedding_matrix)

#Retrieval function
def retrieve_relevant_chunks(query, top_k=3):
    query_emb = np.array(openai.Embedding.create(input=query, model="text-embedding-3-small")["data"][0]["embedding"]).astype("float32")
    distances, indices = index.search(np.expand_dims(query_emb, axis=0), top_k)
    return [ecg_chunks[i] for i in indices[0]]

#Patient Query Analysis with LoRa and RAG

from transformers import pipeline

Generator = pipeline("text-generation", model=peft_model, tokenizer=tokenizer, device=0)

def analyze_ecg(report):
    context_chunks = retrieve_relevant_chunks(report)
    context_text = "\n".join(context_chunks)

    prompt = f"""
You are a cardiologist analyzing an ECG report. Use the following context to interpret the ECG findings and provide a diagnosis.

Context:
{context_text}

Patient ECG Report:
{report}

Provide analysis and advice
"""
    result = Generator(prompt, max_length=200, temprature=0.2)
    return result[0]["generated_text"]


#Example patient query
Patient_report = "PR interval 210ms, QRS duration 110ms, QT interval 430ms, heart rate 78 bpm."
print(analyze_ecg(Patient_report))
