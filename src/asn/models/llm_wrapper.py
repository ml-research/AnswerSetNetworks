

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import torch
from transformers import LlamaForCausalLM
import torch
from peft import get_peft_model, LoraConfig, TaskType
        
    
    
#wrapper class for AutoModelForSequenceClassification
class LLMWrapper():
    
    def __init__(self, model):
        self.model = model

        
    def forward(self, kwargs):
        #forward pass to obtain logits for the specified tokens
        
        return self.model(kwargs).logits.softmax(dim=1)
    
    



# LLM wrapper class that inherits from nn.Module
class LLMWithClassProbs(torch.nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, *args, **kwargs):        
        input_ids = args[0]
        attention_mask = args[1]
        class_token_ids = args[2]
        
        seq_lengths = attention_mask.sum(dim=1) 
        
        output = self.model(input_ids)
        # Get the logits for the last token
        logits = output.logits[0, seq_lengths, :]


        # Get the probabilities for the class tokens
        probs = torch.nn.functional.softmax(logits[:,class_token_ids[0]], dim=1)
        return probs
        
