#保存其他特征到数据中
import json
import os
from collections import defaultdict
import csv

def extract_users_with_pictures(json_file_path):
    """
    从JSON文件中提取用户信息，包括昵称、推文内容和其他特征。

    :param json_file_path: JSON文件路径
    :return: 包含用户信息的列表和所有图片URL的列表
    """
    selected_users_info = []
    all_picture_urls = []

    with open(json_file_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
        for index, data in enumerate(data_list):
            user_info = defaultdict(list)
            if 'nickname' in data:
                user_info['nickname'] = data['nickname']
            if 'tweets' in data:
                for sub_index, tweet in enumerate(data['tweets']):
                    tweet_info = {}
                    if 'tweet_content' in tweet:
                        tweet_info['tweet_content'] = tweet['tweet_content']
                    if 'num_of_comments' in tweet:
                        tweet_info['num_of_comments'] = tweet['num_of_comments']
                    if 'num_of_forwards' in tweet:
                        tweet_info['num_of_forwards'] = tweet['num_of_forwards']
                    if 'num_of_likes' in tweet:
                        tweet_info['num_of_likes'] = tweet['num_of_likes']
                    if 'posted_picture_url' in tweet:
                        picture_urls = tweet['posted_picture_url']
                        if isinstance(picture_urls, list):
                            all_picture_urls.extend(picture_urls)
                            tweet_info['picture_urls'] = picture_urls
                        else:
                            all_picture_urls.append(picture_urls)
                            tweet_info['picture_urls'] = [picture_urls]
                    if 'posting_time' in tweet:
                        tweet_info['posting_time'] = tweet['posting_time']
                    if 'tweet_is_original' in tweet:
                        tweet_info['tweet_is_original'] = tweet['tweet_is_original']

                    user_info['tweets'].append(tweet_info)
            selected_users_info.append(user_info)

    return selected_users_info, all_picture_urls


def save_tweets_to_csv(selected_users_info, csv_file_path):
    """
    将提取的推文内容和其他特征保存为CSV文件。

    :param selected_users_info: 包含用户信息的列表
    :param csv_file_path: CSV文件路径
    """
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['user_index', 'tweet_sub_index', 'tweet_content', 'num_of_comments', 'num_of_forwards',
                      'num_of_likes', 'posted_picture_url', 'posting_time', 'tweet_is_original']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for user_index, user in enumerate(selected_users_info):
            for tweet_sub_index, tweet in enumerate(user['tweets']):
                writer.writerow({
                    'user_index': user_index,
                    'tweet_sub_index': tweet_sub_index,
                    'tweet_content': tweet.get('tweet_content', ''),
                    'num_of_comments': tweet.get('num_of_comments', 0),
                    'num_of_forwards': tweet.get('num_of_forwards', 0),
                    'num_of_likes': tweet.get('num_of_likes', 0),
                    'posted_picture_url': ','.join(tweet.get('picture_urls', ['无'])),
                    'posting_time': tweet.get('posting_time', ''),
                    'tweet_is_original': tweet.get('tweet_is_original', 'False')
                })


# 示例用法
json_file_path = 'depressed.json'
selected_users_info, all_picture_urls = extract_users_with_pictures(json_file_path)
save_csv_path = 'depressed.csv'
save_tweets_to_csv(selected_users_info, save_csv_path)


json_file_path_second = 'normal.json'
selected_users_info, all_picture_urls = extract_users_with_pictures(json_file_path_second)
save_csv_path_second = 'normal.csv'
save_tweets_to_csv(selected_users_info, save_csv_path_second)


#匹配用户行为
import pandas as pd
import numpy as np
# 读取 depressed.csv
depressed_df = pd.read_csv('depressed.csv')
print((depressed_df['posted_picture_url']!='无').sum())

# 读取 depressed_behavior.csv
behavior_df = pd.read_csv('depressed_behavior.csv')
# 检查 depressed_df 的列
print(depressed_df.columns)
# 检查 behavior_df 的列
print(behavior_df.columns)
# 基于 user_index 列合并数据
merged_df = pd.merge(depressed_df, behavior_df, on='user_index', how='left')
# 保存合并后的数据
merged_df.to_csv('merged_depressed.csv', index=False)

# 读取 depressed.csv
depressed_df = pd.read_csv('normal.csv')
print((depressed_df['posted_picture_url']!='无').sum())
# 读取 depressed_behavior.csv
behavior_df = pd.read_csv('normal_behavior.csv')
# 检查 depressed_df 的列
print(depressed_df.columns)
# 检查 behavior_df 的列
print(behavior_df.columns)
# 基于 user_index 列合并数据
merged_df = pd.merge(depressed_df, behavior_df, on='user_index', how='left')
# 保存合并后的数据
merged_df.to_csv('merged_normal.csv', index=False)
