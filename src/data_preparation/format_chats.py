import pandas as pd
import datetime
import re
import os
import json
from tqdm import tqdm

def convert_whatsapp_to_csv(file_path: str) -> pd.DataFrame:
    chat_data = []
    timestamp_pattern = r'^\[\d{2}/\d{2}/\d{2},\s\d{1,2}:\d{2}:\d{2}\s(AM|PM)\]'
    with open(file_path, 'r') as file:
        for line in tqdm(file.readlines()):
            line = line.replace('‎', '')
            if re.match(timestamp_pattern, line):
                line_split = line.split('] ', 1)
                timestamp = line_split[0].replace('[', '').replace(',', '')
                sender_and_msg = line_split[1]

                msg = ''
                sender_and_msg_split = sender_and_msg.split(': ', 1)
                sender = sender_and_msg_split[0]
                if len(sender_and_msg_split) == 2:
                    msg = sender_and_msg_split[1]

                chat_data.append({
                    'timestamp': pd.to_datetime(timestamp, dayfirst=True),
                    'sender': sender,
                    'message': msg[:-1]
                })
            else:
                chat_data[-1]['message'] += '\n' + line[:-1]
    
    # Convert to pandas DataFrame
    chat_df = pd.DataFrame(chat_data)

    return chat_df


def convert_discord_to_csv(file_path: str) -> pd.DataFrame:
    chat_data = []
    timestamp_pattern = r'^\[\d{2}/\d{2}/\d{4}\s\d{2}:\d{2}\]'

    with open(file_path, 'r') as file:
        lines = file.readlines()
        total_lines = len(lines)

        for idx, line in enumerate(tqdm(lines)):
            if re.match(timestamp_pattern, line):

                if chat_data:
                    chat_data[-1]['message'] = chat_data[-1]['message'][:-3]

                line_split = line.split('] ', 1)
                timestamp = line_split[0].replace('[', '').replace(',', '')
                sender = line_split[1][:-1]

                chat_data.append({
                    'timestamp': pd.to_datetime(timestamp, dayfirst=True),
                    'sender': sender,
                    'message': ''
                })
            else:
                chat_data[-1]['message'] += re.sub(r'\n+', '\n', line)

            if idx == total_lines-1:
                chat_data[-1]['message'] = chat_data[-1]['message'][:-1]
    
    # Convert to pandas DataFrame
    chat_df = pd.DataFrame(chat_data)

    return chat_df


def calculate_duration_between_replies(chat_df: pd.DataFrame) -> pd.DataFrame:
    chat_df['timestamp'] = pd.to_datetime(chat_df['timestamp'])
    chat_df = chat_df.sort_values('timestamp', ascending=True)
    chat_df['duration'] = 0

    set_first_duration = False
    for idx, row in tqdm(chat_df.iterrows(), total=len(chat_df)):
        if not set_first_duration:
            chat_df.loc[idx, 'duration'] = 0
            set_first_duration = True
        else:
            chat_df.loc[idx, 'duration'] = (row['timestamp'] - prev_time).seconds

        prev_time = row['timestamp']

    return chat_df


def replace_names(string: str, name_dict: dict) -> str:

    for key in name_dict.keys():
        string = string.replace(key, name_dict[key])

    return string


def handle_actions(sender:str, string: str, actions:dict) -> str:
    if string in actions.keys():
        return str(sender) + " " + str(actions[string])
    else:
        return string


if __name__=='__main__':
    file_path = '/home/shrish/artificial_friends/datasets/whatsapp/_chat_29092024.txt'
    chat_df = convert_whatsapp_to_csv(file_path)

    chat_df = chat_df.loc[chat_df['sender']!='✨CGs over 9000✨']

    chat_df = calculate_duration_between_replies(chat_df)

    name_dict_file_path = '/home/shrish/artificial_friends/datasets/discord_user_name_map.json'
    with open(name_dict_file_path, 'r') as file:
        name_dict = json.load(file)
    chat_df['sender'] = chat_df['sender'].apply(lambda x: replace_names(x, name_dict))

    chat_df['message'] = chat_df['message'].apply(lambda x: replace_names(x, name_dict))

    chat_df['message'] = chat_df['message'].apply(lambda x: "document omitted" if "document omitted" in x else x)
    action_dict_file_path = '/home/shrish/artificial_friends/datasets/action_map.json'
    with open(action_dict_file_path, 'r') as file:
        action_dict = json.load(file)
    chat_df['message'] = chat_df.apply(lambda x: handle_actions(x['sender'], x['message'], action_dict), axis=1)

    destination_file = str(os.path.splitext(file_path)[0]) + '.tsv'
    chat_df.to_csv(destination_file, sep='\t', index=False)
