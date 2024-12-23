import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class T5ChatDataset(Dataset):
    def __init__(self, contexts: pd.Series, replies: pd.Series, tokenizer, max_input_len: int, max_output_len: int):
        super().__init__()
        self.contexts = contexts
        self.replies = replies
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

        assert len(contexts) == len(replies), f"The replies (={len(replies)}) should of the same size as the contexts (={len(contexts)})."

    def __len__(self):
        return len(self.contexts)
    
    def __getitem__(self, index):
        if index >= len(self.contexts) or index < 0:
            raise IndexError(f"Index {index} is out of range for the dataset of length {len(self.contexts)}")

        context = self.contexts[index]
        reply = self.replies[index]

        # Tokenize
        context_tokenized = self.tokenizer.encode_plus(
            str(context),
            truncation=True,
            padding='max_length',
            max_length=self.max_input_len,
            return_tensors='pt'
        )
        
        reply_tokenized = self.tokenizer.encode_plus(
            str(reply),
            truncation=True,
            padding='max_length',
            max_length=self.max_output_len,
            return_tensors='pt'
        )

        return {
            'intput_tensor': context_tokenized['input_ids'].flatten(),
            'attention_mask': context_tokenized['attention_mask'].flatten(),
            'output_tensors': reply_tokenized['input_ids'].flatten()
        }
