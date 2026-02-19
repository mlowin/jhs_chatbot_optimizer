#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import torch


from torch.utils.data.dataset import Dataset
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split

from transformers import TrainerCallback

class EpochEndCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = state.epoch
        _update_func(_uid, epoch)
        print(f"\n>>> Epoche {epoch} ist fertig!")        

     
class TextClassifierDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    
_update_func = None
_uid = None

def fine_tune(df, output_path, update_func, uid, test_split=0.1, tokenizer_model = 'bert-base-uncased', classification_model = "bert-base-multilingual-uncased", train_epochs = 8):

    global _update_func
    global _uid

    _update_func = update_func
    _uid = uid

    # Initial train and test split.
    train_df, test_df = train_test_split(
        df,
        test_size=test_split,
    )


    print(f"Number of rows in training set: {len(train_df)}")
    print(f"Number of rows in test set: {len(test_df)}")


    not_chosen_columns = ['user_query']

    # Select label columns that are not in the list of not chosen columns
    label_columns = [col for col in df.columns if col not in not_chosen_columns]

    # Create a new DataFrame containing only the selected label columns
    df_labels_train = train_df[label_columns]
    df_labels_test = test_df[label_columns]


    # Convert the label columns to lists for each row
    labels_list_train = df_labels_train.values.tolist()
    labels_list_test = df_labels_test.values.tolist()


    labels_list_train = [[float(label) for label in labels] for labels in labels_list_train]
    labels_list_test = [[float(label) for label in labels] for labels in labels_list_test]



    train_texts = train_df['user_query'].tolist()
    train_labels = labels_list_train

    eval_texts = test_df['user_query'].tolist()
    eval_labels = labels_list_test

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, do_lower_case=True)

    train_encodings = tokenizer(train_texts, padding="max_length", truncation=True, max_length=512)
    eval_encodings = tokenizer(eval_texts, padding="max_length", truncation=True, max_length=512)



    train_dataset = TextClassifierDataset(train_encodings, train_labels)
    eval_dataset = TextClassifierDataset(eval_encodings, eval_labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        classification_model,
        problem_type="multi_label_classification",
        num_labels=len(label_columns)
    )


    training_arguments = TrainingArguments(
        output_dir=".",
        eval_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=train_epochs,
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EpochEndCallback()] 
    )

    trainer.train()


    model.eval()

    print(output_path)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)











# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)


# possible_labels = df.Kategorie.unique()

# label_dict = {}
# for index, possible_label in enumerate(possible_labels):
#     label_dict[possible_label] = index
# label_dict

# with open('label_dict.pickle', 'wb') as f:
#     pickle.dump(label_dict, f)
# df['label'] = df.Kategorie.apply(lambda k: label_dict[k])

# from sklearn.model_selection import train_test_split

# X_train, X_val, y_train, y_val = train_test_split(df.index.values, 
#                                                   df.label.values, 
#                                                   test_size=0.15, 
#                                                   random_state=42, 
#                                                   stratify=df.label.values)


# # In[10]:


# df['data_type'] = ['not_set']*df.shape[0]

# df.loc[X_train, 'data_type'] = 'train'
# df.loc[X_val, 'data_type'] = 'val'

# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', 
#                                           do_lower_case=True)


# encoded_data_train = tokenizer.batch_encode_plus(
#     df[df.data_type=='train'].Beschreibung.values, 
#     add_special_tokens=True, 
#     return_attention_mask=True, 
#     pad_to_max_length=True, 
#     max_length=256, 
#     return_tensors='pt'
# )

# encoded_data_val = tokenizer.batch_encode_plus(
#     df[df.data_type=='val'].Beschreibung.values, 
#     add_special_tokens=True, 
#     return_attention_mask=True, 
#     pad_to_max_length=True, 
#     max_length=256, 
#     return_tensors='pt'
# )


# input_ids_train = encoded_data_train['input_ids']
# attention_masks_train = encoded_data_train['attention_mask']
# labels_train = torch.tensor(df[df.data_type=='train'].label.values)

# input_ids_val = encoded_data_val['input_ids']
# attention_masks_val = encoded_data_val['attention_mask']
# labels_val = torch.tensor(df[df.data_type=='val'].label.values)


# dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
# dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)


# model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-uncased",
#                                                       num_labels=df.label.nunique(),
#                                                       output_attentions=False,
#                                                       output_hidden_states=False)



# from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# batch_size = 10

# dataloader_train = DataLoader(dataset_train, 
#                               sampler=RandomSampler(dataset_train), 
#                               batch_size=batch_size)

# dataloader_validation = DataLoader(dataset_val, 
#                                    sampler=SequentialSampler(dataset_val), 
#                                    batch_size=batch_size)



# from transformers import AdamW, get_linear_schedule_with_warmup

# optimizer = AdamW(model.parameters(),
#                   lr=1e-5, 
#                   eps=1e-8)



# epochs = 20

# scheduler = get_linear_schedule_with_warmup(optimizer, 
#                                             num_warmup_steps=0,
#                                             num_training_steps=len(dataloader_train)*epochs)



# from sklearn.metrics import f1_score

# def f1_score_func(preds, labels):
#     preds_flat = np.argmax(preds, axis=1).flatten()
#     labels_flat = labels.flatten()
#     return f1_score(labels_flat, preds_flat, average='weighted')

# def accuracy_per_class(preds, labels):
#     label_dict_inverse = {v: k for k, v in label_dict.items()}
    
#     preds_flat = np.argmax(preds, axis=1).flatten()
#     labels_flat = labels.flatten()

#     for label in np.unique(labels_flat):
#         y_preds = preds_flat[labels_flat==label]
#         y_true = labels_flat[labels_flat==label]
#         print(f'Class: {label_dict_inverse[label]}')
#         print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')


# # In[21]:


# import random
# import numpy as np
# seed_val = 17
# random.seed(seed_val)
# np.random.seed(seed_val)
# torch.manual_seed(seed_val)
# torch.cuda.manual_seed_all(seed_val)


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)

# print(device)


# # In[23]:


# def evaluate(dataloader_val, model):

#     model.eval()
    
#     loss_val_total = 0
#     predictions, true_vals = [], []
    
#     for batch in dataloader_val:
        
#         batch = tuple(b.to(device) for b in batch)
        
#         inputs = {'input_ids':      batch[0],
#                   'attention_mask': batch[1],
#                   'labels':         batch[2],
#                  }

#         with torch.no_grad():        
#             outputs = model(**inputs)
            
#         loss = outputs[0]
#         logits = outputs[1]
#         loss_val_total += loss.item()

#         logits = logits.detach().cpu().numpy()
#         label_ids = inputs['labels'].cpu().numpy()
#         predictions.append(logits)
#         true_vals.append(label_ids)
    
#     loss_val_avg = loss_val_total/len(dataloader_val) 
    
#     predictions = np.concatenate(predictions, axis=0)
#     true_vals = np.concatenate(true_vals, axis=0)
            
#     return loss_val_avg, predictions, true_vals


# # In[24]:


# t_start = time.time()
# for epoch in tqdm(range(1, epochs+1)):
    
#     model.train()
    
#     loss_train_total = 0

#     progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
#     for batch in progress_bar:

#         model.zero_grad()
        
#         batch = tuple(b.to(device) for b in batch)
        
#         inputs = {'input_ids':      batch[0],
#                   'attention_mask': batch[1],
#                   'labels':         batch[2],
#                  }       

#         outputs = model(**inputs)
        
#         loss = outputs[0]
#         loss_train_total += loss.item()
#         loss.backward()

#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

#         optimizer.step()
#         scheduler.step()
        
#         progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
         
        
#     torch.save(model.state_dict(), f'finetuned_BERT_full_epoch_{epoch}.model')
        
#     tqdm.write(f'\nEpoch {epoch}')
    
#     loss_train_avg = loss_train_total/len(dataloader_train)            
#     tqdm.write(f'Training loss: {loss_train_avg}')
    
#     val_loss, predictions, true_vals = evaluate(dataloader_validation, model)
#     val_f1 = f1_score_func(predictions, true_vals)
#     tqdm.write(f'Validation loss: {val_loss}')
#     tqdm.write(f'F1 Score (Weighted): {val_f1}')
# t_end = time.time()

# # In[25]:


# model_test = BertForSequenceClassification.from_pretrained("bert-base-multilingual-uncased",
#                                                       num_labels=len(label_dict),
#                                                       output_attentions=False,
#                                                       output_hidden_states=False)

# model_test.to(device)


# # In[26]:


# model_test.load_state_dict(torch.load('finetuned_BERT_full_epoch_20.model', map_location=torch.device(device)))


# # In[27]:




# # df_test = pd.read_csv('df_log.csv')


# # #df_test = df_test.iloc[-100:]
# # encoded_data_test = tokenizer.batch_encode_plus(
# #     df_test.question.values, 
# #     add_special_tokens=True, 
# #     return_attention_mask=True, 
# #     pad_to_max_length=True, 
# #     max_length=256, 
# #     return_tensors='pt'
# # )

# # input_ids_test = encoded_data_test['input_ids']
# # attention_masks_test = encoded_data_test['attention_mask']
# # labels_test = torch.tensor(df_test.label.values)

# # dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)
# # dataloader_test = DataLoader(dataset_test, 
# #                                    sampler=SequentialSampler(dataset_test), 
# #                                    batch_size=batch_size)


# # _, predictions, true_vals = evaluate(dataloader_test, model_test)
# print("#################################################")
# print("Training time:",(t_end-t_start))
# # print("F1:",f1_score_func(predictions, true_vals)) # accuracy_per_class