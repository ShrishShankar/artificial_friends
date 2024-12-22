import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
import math
import os

tqdm.pandas()

SENDER_TOKEN_LEN = 3 # Token length of the string "sender:"
CONTEXT_TOKEN_LEN = 3 # Token length of the string "context:"
MAX_TOKEN_LENGTH = 450
CONTEXT_WINDOW_DURATION = 3600

def get_tokenized_length(tokenizer: AutoTokenizer):
    def tokenize_str(string: str):
        return tokenizer.encode_plus(string, return_tensors='pt')['input_ids'].size()[1]
    return tokenize_str

def create_chat_dataset(chat_df: pd.DataFrame) -> pd.DataFrame:
    for idx, row in tqdm(chat_df.iterrows(), total=len(chat_df)):
        # print(row)
        # Set
        sender = row['sender']
        message_time = row['timestamp']
        chat_df.loc[idx, 'input'] = "sender: " + sender + " | " + "context: \n"
        chat_df.loc[idx, 'context_length'] = 0
        total_token_length = SENDER_TOKEN_LEN + CONTEXT_TOKEN_LEN
        chat_df.loc[idx, 'total_tokens'] = total_token_length

        # If no messages before it in last 3600 seconds then context = start conversation
        if idx==0:
            duration_from_previous_message = math.inf
        else:
            duration_from_previous_message = (message_time - chat_df.loc[idx-1, 'timestamp']).seconds
            total_token_length += chat_df.loc[idx-1, 'tokenized_message_with_sender_length']

        if duration_from_previous_message > 3600 or total_token_length > MAX_TOKEN_LENGTH:
            chat_df.loc[idx, 'input'] +=  "start conversation"
        else:
            # If there are messages in last 3600 seconds then add sender+message to context till context is 500 tokens large or time is greater than 3600
            prev_idx = 1
            context = ""
            while(duration_from_previous_message <= CONTEXT_WINDOW_DURATION and 
                  total_token_length <= MAX_TOKEN_LENGTH and
                  prev_idx <= idx):

                context = chat_df.loc[idx-prev_idx, 'sender'] + ": " + \
                          chat_df.loc[idx-prev_idx, 'message'] + \
                          ("\n" if prev_idx!=1 else "") + \
                          context
                
                chat_df.loc[idx, 'context_length'] += 1
                chat_df.loc[idx, 'total_tokens'] += chat_df.loc[idx-prev_idx, 'tokenized_message_with_sender_length']

                prev_idx += 1
                # Handle first row logic
                if prev_idx<=idx:
                    duration_from_previous_message = (message_time - chat_df.loc[idx-prev_idx, 'timestamp']).seconds
                    total_token_length += chat_df.loc[idx-prev_idx, 'tokenized_message_with_sender_length']
                else:
                    duration_from_previous_message = math.inf
                    total_token_length = MAX_TOKEN_LENGTH + 1
            chat_df.loc[idx, 'input'] += context

    return chat_df


if __name__=='__main__':
    file_path = '/home/shrish/artificial_friends/datasets/whatsapp/_chat_29092024_manually_verified.tsv'
    chat_df = pd.read_csv(file_path, sep='\t')
    chat_df['timestamp'] = pd.to_datetime(chat_df['timestamp'])

    # Remove messages with greater than 200 characters
    print(chat_df['message'].str.len().mean())
    chat_df = chat_df.loc[chat_df['message'].str.len() <= 200]
    chat_df = chat_df.reset_index(drop=True)
    print(chat_df['message'].str.len().mean())

    model_name = 'google-t5/t5-small'
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Get tokenized length for context creation
    chat_df['tokenized_message_with_sender_length'] = chat_df.progress_apply(lambda x: get_tokenized_length(tokenizer)(x['sender'] + ': ' + x['message']), axis=1)
    print(chat_df['tokenized_message_with_sender_length'].mean())
    print(chat_df['tokenized_message_with_sender_length'].max())

    # Create a context based dataset
    chat_df = create_chat_dataset(chat_df)

    destination_file = str(os.path.splitext(file_path)[0]) + '_with_context' + '.tsv'
    chat_df.to_csv(destination_file, sep='\t', index=False)

    print(chat_df)
