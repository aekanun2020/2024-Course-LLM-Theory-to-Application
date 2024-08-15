#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(' rm -rf results/ wandb/ fine_tuned_model_10k/ fine_tuned_model_improved/')


# In[2]:


import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

# กำหนดให้ใช้เฉพาะ GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)

# 1. โหลดและเตรียมข้อมูล
def load_data(business_file, review_file, sample_size=500000):
    # โหลดข้อมูล business
    businesses = {}
    with open(business_file, 'r') as f:
        for line in f:
            business = json.loads(line)
            businesses[business['business_id']] = business

    # โหลดและสุ่มเลือกข้อมูล review
    reviews = []
    with open(review_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            review = json.loads(line)
            if review['business_id'] in businesses:
                reviews.append(review)

    # สร้าง DataFrame
    df = pd.DataFrame(reviews)
    
    # เพิ่มข้อมูล business เข้าไปใน DataFrame (ยกเว้น business_name)
    df['categories'] = df['business_id'].map(lambda x: businesses[x].get('categories', ''))
    df['city'] = df['business_id'].map(lambda x: businesses[x]['city'])

    # สร้าง full_text ที่ไม่มี business_name
    df['full_text'] = df.apply(lambda row: f"ประเภท: {row['categories']}\nเมือง: {row['city']}\nคะแนน: {row['stars']}/5\nรีวิว: {row['text']}\n\n", axis=1)
    
    return df, businesses

# โหลดและเตรียมข้อมูล
df, businesses = load_data('../yelp_academic_dataset_business.json', '../yelp_academic_dataset_review.json', sample_size=500000)
print(f"จำนวนรีวิวที่โหลด: {len(df)}")

# 2. เตรียมข้อมูลสำหรับ fine-tuning
df['label'] = df['business_id']
label_to_id = {label: id for id, label in enumerate(df['label'].unique())}
id_to_label = {id: label for label, id in label_to_id.items()}
df['label_id'] = df['label'].map(label_to_id)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# 3. เตรียม Dataset
train_dataset = Dataset.from_pandas(train_df[['full_text', 'label_id']])
val_dataset = Dataset.from_pandas(val_df[['full_text', 'label_id']])

# 4. เตรียม Tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def tokenize_and_prepare(examples):
    tokenized = tokenizer(examples['full_text'], padding="max_length", truncation=True, max_length=512)
    tokenized['labels'] = examples['label_id']
    return tokenized

tokenized_train = train_dataset.map(tokenize_and_prepare, batched=True, remove_columns=train_dataset.column_names)
tokenized_val = val_dataset.map(tokenize_and_prepare, batched=True, remove_columns=val_dataset.column_names)

# 5. เตรียมโมเดลและ Trainer
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_to_id))
model.to('cuda:0')

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    remove_unused_columns=False,
    no_cuda=False,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
)

# 6. Fine-tune โมเดล
print("เริ่มการ fine-tune โมเดล...")
trainer.train()
print("Fine-tune เสร็จสิ้น")

# 7. บันทึกโมเดล
trainer.save_model("./fine_tuned_model_improved")
print("บันทึกโมเดลเรียบร้อย")


# In[3]:


# 8. ฟังก์ชันสำหรับการ matching
def match_business_to_text(user_text, top_n=5):
    inputs = tokenizer(user_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to('cuda:0')
    with torch.no_grad():
        outputs = model(**inputs)
    
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    top_n_probs, top_n_indices = torch.topk(probabilities, top_n)
    
    results = []
    for prob, idx in zip(top_n_probs[0], top_n_indices[0]):
        business_id = id_to_label[idx.item()]
        print ("MODEL's ANSWER ============> ", business_id)
        business_info = businesses[business_id]
        results.append({
            'business_id': business_id,
            'name': business_info['name'],
            'categories': business_info.get('categories', ''),
            'city': business_info['city'],
            'probability': prob.item()
        })
    
    return results

# 9. ฟังก์ชันสำหรับแสดงผลลัพธ์
def display_results(results):
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['name']}")
        print(f"   Business ID: {result['business_id']}")
        print(f"   Categories: {result['categories']}")
        print(f"   City: {result['city']}")
        print(f"   Probability: {result['probability']:.4f}")
        print(f"   Match level: {get_match_level(result['probability'])}")
        
        # แสดงรีวิวทั้งหมดของธุรกิจนี้
        print("\n   Reviews:")
        business_reviews = df[df['business_id'] == result['business_id']]['text'].tolist()
        for j, review in enumerate(business_reviews[:3], 1):  # แสดง 3 รีวิวแรก
            print(f"   Review {j}: {review[:200]}...")  # แสดง 200 ตัวอักษรแรกของแต่ละรีวิว
        
        print("\n" + "="*50 + "\n")

def get_match_level(probability):
    if probability > 0.8:
        return "Very High"
    elif probability > 0.6:
        return "High"
    elif probability > 0.4:
        return "Moderate"
    elif probability > 0.2:
        return "Low"
    else:
        return "Very Low"


# In[4]:


# 10. ตัวอย่างการใช้งาน
user_text = "ฉันกำลังมองหาร้านอาหารอิตาเลียนที่มีพาสต้าและพิซซ่าอร่อยๆ"
results = match_business_to_text(user_text)

print(f"User Text: {user_text}\n")
print("Top 5 Matching Businesses:")
display_results(results)


# In[5]:


# 10. ตัวอย่างการใช้งาน
user_text = "I'm looking for an Italian restaurant with delicious pasta and pizza."
results = match_business_to_text(user_text)

print(f"User Text: {user_text}\n")
print("Top 5 Matching Businesses:")
display_results(results)


# In[ ]:




