
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import numpy as np

import csv

sex_lookup = {}
with open('/workspaces/ASN/data/parent_relation/sex.csv') as f:
    r = csv.reader(f, delimiter=',')
    header = next(r)
    for row in r:
        child,sex = row 
        sex_lookup[child] = sex


def create_data_splits(for_asn_training):
    
    # read in child_dict from csv
    child_df = pd.read_csv('/workspaces/ASN/data/parent_relation/celebtrity_dataset_split.csv')

    #read in the data
    df = pd.read_csv('/workspaces/ASN/data/parent_relation/celebrity_relations_parent_child_pairs.csv')

    
    train_data = []
    val_data = []
    
    #includes the pairs from which we have seen the other side during training
    parent_child_pairs_seen = []
    child_parent_pairs_seen = []
    
    #includes the pairs from which we have not seen the other side during training
    parent_child_pairs_unseen = []
    child_parent_pairs_unseen = []

    #go through all 1513 entries in the dataset
    for _, row in df.iterrows():
        #get the parent and child and the relation
        child = row['child']
        parent = row['parent']
        relation = row['parent_type']
        
        # obtain the reverse relation
        if sex_lookup[child] == 'Male':
            reverse_relation = 'son'
        elif sex_lookup[child] == 'Female':
            reverse_relation = 'daughter'
        
        
        if relation == 'father':
            sex_parent = 'Male'
        elif relation == 'mother':
            sex_parent = 'Female'
            
        
        
        # Add the (person1, person2, relation) tuple
        if child_df.loc[child_df['child']==child]['dataset'].item() == 'train': #for some add both to train
            
            #flip a coin to decide if we add the reverse relation to the train data. We add the other way to the test data
            if child_df.loc[child_df['child']==child]['direction'].item():
                train_data.append((parent, child, relation, sex_lookup[child]))
                child_parent_pairs_seen.append((child, parent, reverse_relation, sex_parent))

                if for_asn_training: #if we use SLASH training we obtain the other way around as well
                    train_data.append((child, parent, reverse_relation, sex_parent))
            else:
                train_data.append((child, parent, reverse_relation, sex_parent))
                parent_child_pairs_seen.append((parent, child, relation, sex_lookup[child]))
                if for_asn_training: #if we use SLASH training we obtain the other way around as well
                    train_data.append((parent, child, relation, sex_lookup[child]))
                    
        elif child_df.loc[child_df['child']==child]['dataset'].item() == 'val': #for some add both to train
                val_data.append((child, parent, reverse_relation, sex_parent))
                val_data.append((parent, child, relation, sex_lookup[child]))
                
        elif child_df.loc[child_df['child']==child]['dataset'].item() == 'test':#for some add only child parent to train and test the other way around
            parent_child_pairs_unseen.append((parent, child, relation, sex_lookup[child]))
            child_parent_pairs_unseen.append((child, parent, reverse_relation, sex_parent))

    train_data = np.array(train_data)
    val_data = np.array(val_data)
    child_parent_data_seen= np.array(child_parent_pairs_seen)
    parent_child_data_seen = np.array(parent_child_pairs_seen)
    child_parent_data_unseen = np.array(child_parent_pairs_unseen)
    parent_child_data_unseen = np.array(parent_child_pairs_unseen)

    return train_data, val_data, child_parent_data_seen, parent_child_data_seen, child_parent_data_unseen, parent_child_data_unseen


# Custom dataset
class FamilyRelationDataset(Dataset):
    """
    Custom dataset for the family relation classification task. It returns a dictionary with the input IDs, attention mask, and labels.
    """
    def __init__(self, data, tokenizer,split, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        person1, person2, relation,_ = self.data[idx]
        input_text = f"What is the relation between {person1} and {person2}? The answer is:"
        
        if self.split == 'train':
            # Append the relation to the input text for training
            full_text = f"{input_text} {relation}"
            
            # Tokenize the full text including the relation
            inputs = self.tokenizer(full_text, return_tensors="pt", max_length=self.max_length, padding="max_length", truncation=True)
            input_ids = inputs["input_ids"].squeeze()
            attention_mask = inputs["attention_mask"].squeeze()
            
            # Create labels by shifting the input_ids
            labels = input_ids.clone()
            
            # Set the labels of the input text (not including the relation) to -100
            input_length = len(self.tokenizer(input_text, add_special_tokens=False)['input_ids'])
            #print(tokenizer.decode(labels[:input_length+1]))
            labels[:input_length+1] = -100
            

        elif self.split == 'test':
            # For testing, we don't append the relation to the input
            inputs = self.tokenizer(input_text, return_tensors="pt", max_length=self.max_length, padding="max_length", truncation=True)
            input_ids = inputs["input_ids"].squeeze()
            attention_mask = inputs["attention_mask"].squeeze()
            labels = relation  # Keep the original label for evaluation

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        
# Custom dataset
class FamilyRelationASNDataset(Dataset):
    """
    Custom dataset for the family relation classification task. It returns a dictionary with the input IDs, attention mask, and labels.
    """
    def __init__(self, data, tokenizer,split, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.class_token_ids_relation = torch.tensor([tokenizer.encode(c, add_special_tokens=False)[0] for c in ['mother','father','daughter','son']])
        self.class_token_ids_sex = torch.tensor([tokenizer.encode(c, add_special_tokens=False)[0] for c in ['male','female']])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        person1, person2, relation, sex_p2 = self.data[idx]
        input_text_original = f"What is the relation between {person1} and {person2}? The answer is:"
        input_text_derived = f"What is the relation between {person2} and {person1}? The answer is:"
        input_text_sex_person1 = f"What is the sex of {person1}? The answer is:"
        input_text_sex_person2 = f"What is the sex of {person2}? The answer is:"
        
        # # one hot encode the labels using the following encoding: mother, father, daughter, son
        # if relation == 'father':
        #     labels = torch.tensor([1,0,0,0])
        # elif relation == 'mother':
        #     labels = torch.tensor([0,1,0,0])
        # elif relation == 'daughter':
        #     labels = torch.tensor([0,0,1,0])
        # elif relation == 'son':
        #     labels = torch.tensor([0,0,0,1])
        
        # Tokenize the full text including the relation
        inputs_original = self.tokenizer(input_text_original, return_tensors="pt", max_length=self.max_length, padding="max_length", truncation=True)
        input_ids_original = inputs_original["input_ids"].squeeze()
        attention_mask_original = inputs_original["attention_mask"].squeeze()
        
        inputs_derived = self.tokenizer(input_text_derived, return_tensors="pt", max_length=self.max_length, padding="max_length", truncation=True)
        input_ids_derived = inputs_derived["input_ids"].squeeze()
        attention_mask_derived = inputs_derived["attention_mask"].squeeze()
        
        inputs_sex_person1 = self.tokenizer(input_text_sex_person1, return_tensors="pt", max_length=self.max_length, padding="max_length", truncation=True)
        input_ids_sex_person1 = inputs_sex_person1["input_ids"].squeeze()
        attention_inputs_sex_person1 = inputs_sex_person1["attention_mask"].squeeze()
        
        inputs_sex_person2 = self.tokenizer(input_text_sex_person2, return_tensors="pt", max_length=self.max_length, padding="max_length", truncation=True)
        input_ids_sex_person2 = inputs_sex_person2["input_ids"].squeeze()
        attention_inputs_sex_person2 = inputs_sex_person2["attention_mask"].squeeze()
        
        # Create labels by shifting the input_ids
        #labels = input_ids.clone()
        
        # # Set the labels of the input text (not including the relation) to -100
        # input_length = len(self.tokenizer(input_text_child_parent, add_special_tokens=False)['input_ids'])
        # labels[:input_length+1] = -100
        
        
        return {
            "relation": relation,
            "sex_p2":sex_p2.lower(),
            "#npp(relation(p1,p2),[mother,father,daughter,son]) :- person(p1),person(p2),p1!=p2.": {
                "input_ids": input_ids_original,
                "attention_mask": attention_mask_original,
                "classes": self.class_token_ids_relation
                },
            '#npp(relation(p2,p1),[mother,father,daughter,son]) :- person(p2),person(p1),p2!=p1.': {
                "input_ids":input_ids_derived,
                "attention_mask":attention_mask_derived,
                "classes": self.class_token_ids_relation
                },
            '#npp(sex(p1),[male,female]) :- person(p1).': {
                'input_ids':input_ids_sex_person1,
                'attention_mask':attention_inputs_sex_person1,
                'classes': self.class_token_ids_sex
                },
            '#npp(sex(p2),[male,female]) :- person(p2).': {
                    'input_ids':input_ids_sex_person2,
                    'attention_mask':attention_inputs_sex_person2,
                    'classes': self.class_token_ids_sex
                } 
        }
        
# Custom dataset
class FamilyRelationASNDataset2(Dataset):
    """
    Custom dataset for the family relation classification task. It returns a dictionary with the input IDs, attention mask, and labels.
    """
    def __init__(self, data, tokenizer,split, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.class_token_ids_relation = torch.tensor([tokenizer.encode(c, add_special_tokens=False)[0] for c in ['mother','father','daughter','son']])
        self.class_token_ids_sex = torch.tensor([tokenizer.encode(c, add_special_tokens=False)[0] for c in ['male','female']])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        person1, person2, relation, sex_p2 = self.data[idx]
        input_text_original = f"What is the relation between {person1} and {person2}? The answer is:"
        input_text_derived = f"What is the relation between {person2} and {person1}? The answer is:"
        input_text_sex_person1 = f"What is the sex of {person1}? The answer is:"
        input_text_sex_person2 = f"What is the sex of {person2}? The answer is:"
        
        # # one hot encode the labels using the following encoding: mother, father, daughter, son
        # if relation == 'father':
        #     labels = torch.tensor([1,0,0,0])
        # elif relation == 'mother':
        #     labels = torch.tensor([0,1,0,0])
        # elif relation == 'daughter':
        #     labels = torch.tensor([0,0,1,0])
        # elif relation == 'son':
        #     labels = torch.tensor([0,0,0,1])
        
        # Tokenize the full text including the relation
        inputs_original = self.tokenizer(input_text_original, return_tensors="pt", max_length=self.max_length, padding="max_length", truncation=True)
        input_ids_original = inputs_original["input_ids"].squeeze()
        attention_mask_original = inputs_original["attention_mask"].squeeze()
        
        inputs_derived = self.tokenizer(input_text_derived, return_tensors="pt", max_length=self.max_length, padding="max_length", truncation=True)
        input_ids_derived = inputs_derived["input_ids"].squeeze()
        attention_mask_derived = inputs_derived["attention_mask"].squeeze()
        
        inputs_sex_person1 = self.tokenizer(input_text_sex_person1, return_tensors="pt", max_length=self.max_length, padding="max_length", truncation=True)
        input_ids_sex_person1 = inputs_sex_person1["input_ids"].squeeze()
        attention_inputs_sex_person1 = inputs_sex_person1["attention_mask"].squeeze()
        
        inputs_sex_person2 = self.tokenizer(input_text_sex_person2, return_tensors="pt", max_length=self.max_length, padding="max_length", truncation=True)
        input_ids_sex_person2 = inputs_sex_person2["input_ids"].squeeze()
        attention_inputs_sex_person2 = inputs_sex_person2["attention_mask"].squeeze()
        
        # Create labels by shifting the input_ids
        #labels = input_ids.clone()
        
        # # Set the labels of the input text (not including the relation) to -100
        # input_length = len(self.tokenizer(input_text_child_parent, add_special_tokens=False)['input_ids'])
        # labels[:input_length+1] = -100
        
        
        return {
            "relation": relation,
            "sex_p2":sex_p2.lower(),
            "#npp(relation(p1,p2),[mother,father,daughter,son]) :- person(p1),person(p2),p1!=p2.": {
                "input_ids": input_ids_original,
                "attention_mask": attention_mask_original,
                "classes": self.class_token_ids_relation
                },
            '#npp(relation(p2,p1),[mother,father,daughter,son]) :- person(p2),person(p1),p2!=p1.': {
                "input_ids":input_ids_derived,
                "attention_mask":attention_mask_derived,
                "classes": self.class_token_ids_relation
                },
            '#npp(sex(p1),[male,female]) :- person(p1).': {
                'input_ids':input_ids_sex_person1,
                'attention_mask':attention_inputs_sex_person1,
                'classes': self.class_token_ids_sex
                },
            '#npp(sex(p2),[male,female]) :- person(p2).': {
                    'input_ids':input_ids_sex_person2,
                    'attention_mask':attention_inputs_sex_person2,
                    'classes': self.class_token_ids_sex
                } 
        }
        
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
def evaluate_model(model, test_loader, tokenizer, device, batch_size=8, print_preds=False):
    """
    Evaluate the model on the test set and return the accuracy, precision, recall, and F1 score.
    """
    model.eval()
    
    relations = ["mother", "father", "daughter", "son"]
    # Define the relations we want to classify
    relation_token_ids = [tokenizer.encode(rel, add_special_tokens=False)[0] for rel in relations]

    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            next_token_pos = ((input_ids == 2).nonzero(as_tuple=True))[1][0]
            logits = outputs.logits[:, next_token_pos, :]  # Get logits for the last token -> get the logits at the end of the input text
            
            # Only consider logits for the specified relation tokens
            relation_logits = logits[:, relation_token_ids]
            predicted_relation_indices = torch.argmax(relation_logits, dim=1)
            predictions = [relations[i] for i in predicted_relation_indices.cpu().numpy()]
            all_preds.extend(predictions)
            all_labels.extend(batch["labels"])
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    
    if print_preds:
        print("ALL LABELS", all_labels)
        print("ALL PREDS ", all_preds)
    # Per-class metrics
    class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(all_labels, all_preds, zero_division=0, average=None)
    
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_class": {rel: {"precision": p, "recall": r, "f1": f} 
                      for rel, p, r, f in zip(relations, class_precision, class_recall, class_f1)}
    }
    
    return results