import os
import time
import math
import json
import torch
import requests
import argparse
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from itertools import product, groupby
from transformers import AutoTokenizer, AutoModelForCausalLM

## Unified function for calculating perplexity
def get_perplexity(prompt: str, sentence: str, model = None, tokenizer = None):
    if model is None:
        raise Exception("model is None")
    else:
        # Get tokenized prompt length
        # import pdb; pdb.set_trace()
        if prompt:
            prompt_encodings = tokenizer(prompt, return_tensors='pt')
            tokenized_prompt_len = len(prompt_encodings['input_ids'][0])
        else:
            tokenized_prompt_len = 0
        
        # Get ppl with prompt
        text = prompt + sentence
        encodings = tokenizer(text, return_tensors='pt')
        # print(encodings)
        input_ids = encodings['input_ids']
        labels = input_ids.clone()
        labels[:,:tokenized_prompt_len] = -100
        outputs = model(**encodings, labels=labels)
        loss = outputs.loss
    return math.exp(loss)

## Function to replace words in sentence
def replace_target_words(sentence, target_dict):
    
    # Identify all keywords in the sentence and their corresponding replacements
    applicable_replacements = [(key, value) for key, values in target_dict.items() for value in values if key in sentence]

    # Replace sentences
    replaced_sentences = []
    for origin_word, replace_word in applicable_replacements:
        replaced_sentence = sentence.replace(origin_word, replace_word)
        replaced_sentences.append(replaced_sentence)

    return replaced_sentences

## Function to calculate perplexity in every bias sentence
def calculate_perplexity(df, target_dict, model = None, tokenizer = None, prompt = ""):
    for index, row in tqdm(df.iterrows(), total=len(df)):
        sentence = row['Biased Sentences']
        
        ## Calculate original sentence
        df.loc[index, 'origin_ppl'] = get_perplexity(prompt, sentence, model, tokenizer)
        
        ## Calculate replace sentence
        replaced_sentences = replace_target_words(sentence, target_dict)
        ppl_list = []
        for replaced_sentence in replaced_sentences:
            ppl_list.append(get_perplexity(prompt, replaced_sentence, model, tokenizer))
            # ppl_list.append(10)
        idx = np.argmin(ppl_list) # Select the smallest PPL as the replaced_sentence to be compared
        # print('ppl_list:',ppl_list)
        # print('idx:', idx)
        df.loc[index, 'replaced_sentence'] = replaced_sentences[idx]
        # df.loc[index, 'replace_ppl'] = ppl_list[idx]
        
        ## Fix: change min ppl to mean ppl
        df.loc[index, 'replace_ppl'] = np.mean(ppl_list)
        
        ## Calculate the difference between original sentence and replace sentence
        df.loc[index, 'delta_ppl'] = df.loc[index, 'replace_ppl'] - df.loc[index, 'origin_ppl']
    return df

## Function to remove outliers data which origin_ppl and replace_ppl are both outliers
def get_outliers(df):
    o_list, r_list = df['origin_ppl'].to_list(), df['replace_ppl'].to_list()
    o_mean, r_mean = np.mean(o_list), np.mean(r_list)
    o_std, r_std = np.std(o_list), np.std(r_list)
    
    # df[df['origin_ppl']>o_mean+3*o_std][df['origin_ppl']<o_mean-3*o_std][df['replace_ppl']>r_mean+3*r_std][df['replace_ppl']<r_mean-3*r_std]
    condition_o = (df['origin_ppl'] > o_mean + 3 * o_std) | (df['origin_ppl'] < o_mean - 3 * o_std)
    condition_r = (df['replace_ppl'] > r_mean + 3 * r_std) | (df['replace_ppl'] < r_mean - 3 * r_std)
    
    outlier_df = df[condition_o | condition_r]
    filtered_df = df[~(condition_o | condition_r)]
    
    return outlier_df, filtered_df

## Function to combinate target
def get_target_combination(df, origin_target, replace_target):
    L1 = df[origin_target].dropna().tolist()
    L2 = df[replace_target].dropna().tolist()
    all_lists = [L1, L2]
    
    for pair in itertools.combinations(all_lists, 2):
            list_1, list_2 = pair
            combinations = list(itertools.product(list_1, list_2))
            # combinations.extend(itertools.product(list_2, list_1))
    df = pd.DataFrame(combinations, columns=[origin_target, replace_target])
    
    return df

## 
def add_categories_type(df, attribute_category_dict, attribute_type_dict):
    df_exploded = df.copy()
    
    # import pdb; pdb.set_trace()
    # for i, c in enumerate(df_exploded['T-A Combination']):
    #     print(i, c)
    #     eval(c)
    
    df_exploded['T-A Combination'] = df_exploded['T-A Combination'].apply(eval)
    df_exploded = df_exploded.explode('T-A Combination')
    df_exploded['T-A Combination'] = df_exploded['T-A Combination'].apply(lambda x: x[1] or "其他")
    
    for index, row in df_exploded.iterrows():
        df_exploded.loc[index, 'Category'] = attribute_category_dict[row['T-A Combination']]
        df_exploded.loc[index, 'Type'] = attribute_type_dict[row['T-A Combination']]
    return df_exploded

## 
def perform_ttest(group):
    t_paired, p_paired = stats.ttest_rel(group['origin_ppl'], group['replace_ppl'])
    count = group.shape[0]
    delta_ratio = (group['delta_ppl'] / group['origin_ppl']).mean()*100
    return pd.Series({'t_statistic': t_paired, 
                      'p_value': p_paired,
                      'statistically significant': p_paired < 0.05,
                      'count': count, 
                      'delta_ratio': delta_ratio})

## Main function
def main(args):
    
    ## Read bias sentences
    df = pd.read_csv(args.bias_sentences_path, dtype={'T-A Combination': str})
    target_pair = pd.read_csv(args.target_pair_path)

    ## Load prompt template
    if "Taiwan-LLM" in args.model_path:
        prompt_template = json.load(open("/home/u3659277/master-thesis/TWBias/prompt_template_speaking/twllm_2.json"))
    elif "Breeze" in args.model_path:
        prompt_template = json.load(open("/home/u3659277/master-thesis/TWBias/prompt_template_speaking/breeze.json"))
    elif "xwin" in args.model_path:
        prompt_template = json.load(open("/home/u3659277/master-thesis/TWBias/prompt_template_speaking/xwin.json"))
    else:
        prompt_template = json.load(open("/home/u3659277/master-thesis/TWBias/prompt_template_speaking/llama2.json"))
        
    # prompt_template = json.load(open(args.prompt_template_path))
    if args.prompt_template:
        prompt = prompt_template[args.prompt_template]
    else:
        prompt = ""
    
    ## Some targets need to be arranged to form target pairs 
    if args.target_combination:
        target_pair = get_target_combination(target_pair, args.origin_target, args.replace_target)  
    
    ## Construct a dictionary for target pair
    '''
    dict -> {key:[value,value,...,value]} or {key:[value]}
    
    key:origin_target, value:replace_target
    
    '''
    target_dict = target_pair.groupby(args.origin_target)[args.replace_target].agg(list).to_dict()
    # print(target_dict)

    ## Construct dictionaries for attribute categories and type (positive)
    attribute_df = pd.read_csv(args.attribute_category_path, dtype={'Type': str})
    attribute_category_dict = attribute_df.set_index('Content')['Category'].to_dict()
    attribute_type_dict = attribute_df.set_index('Content')['Type'].to_dict()

    ## Check directory existed
    model_name = args.model_path.split('/')[-1]
    target_group_name = os.path.basename(args.attribute_category_path).split('-')[0]
    # print('Target:', target_group_name)
    if not os.path.exists(f"../data/ethical_group/result_wo_outliers/{model_name}/{target_group_name}_{args.replace_target}"):
        os.makedirs(f"../data/ethical_group/result_wo_outliers/{model_name}/{target_group_name}_{args.replace_target}")

    ## Calculate PPL
    # print('Calculate PPL: ',args.calculate_perplexity)
    if args.calculate_perplexity:
        if args.api:
            model = None
            tokenizer = None
        else:
            # print(torch.cuda.device_count())
            model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map='auto', torch_dtype=torch.float16)
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, device_map='auto')
        
        df = calculate_perplexity(df, target_dict, model, tokenizer, prompt)
        df.to_csv(f"../data/ethical_group/result_wo_outliers/{model_name}/{target_group_name}_{args.replace_target}/{model_name}_{target_group_name}_{args.replace_target}_output_{args.prompt_template}.csv")
        
    ## Remove outliers
    df_outliers, df = get_outliers(df)
    
    df_category = add_categories_type(df, attribute_category_dict, attribute_type_dict)
    
    ## Calculate p-value, t-value for category analysis
    result_category = df_category.groupby(['Category', 'Type']).apply(perform_ttest)
    # print(result_category)
    
    result_category_all = df_category.groupby(['Category']).apply(perform_ttest)
    
    ## Calculate p-value, t-value for toxicity analysis
    result_toxicity = df.groupby(['Toxicity']).apply(perform_ttest)
    result_all = perform_ttest(df)
    result_toxicity = result_toxicity._append(result_all, ignore_index=True)
    result_toxicity.index = ['0', '1', 'All']
    # print(result_toxicity)
    
    model_name = args.model_path.split('/')[-1]
    target_group_name = os.path.basename(args.attribute_category_path).split('-')[0]
    result_category.to_csv(f"../data/ethical_group/result_wo_outliers/{model_name}/{target_group_name}_{args.replace_target}/{model_name}_{target_group_name}_{args.replace_target}_category_{args.prompt_template}.csv")
    result_category_all.to_csv(f"../data/ethical_group/result_wo_outliers/{model_name}/{target_group_name}_{args.replace_target}/{model_name}_{target_group_name}_{args.replace_target}_category_all_{args.prompt_template}.csv")
    result_toxicity.to_csv(f"../data/ethical_group/result_wo_outliers/{model_name}/{target_group_name}_{args.replace_target}/{model_name}_{target_group_name}_{args.replace_target}_toxicity_{args.prompt_template}.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## files
    parser.add_argument("--bias_sentences_path", "-bp", type=str, default="/home/u3659277/master-thesis/TWBias/data/gender/label_data_female.csv")
    parser.add_argument("--target_pair_path", "-tp", type=str, default="/home/u3659277/master-thesis/TWBias/data/gender/target_gender.csv")
    parser.add_argument("--prompt_template_path", "-pt", type=str, default="/home/u3659277/master-thesis/TWBias/prompt_template_speaking/llama2.json")
    parser.add_argument("--attribute_category_path", "-ap", type=str, default="/home/u3659277/master-thesis/TWBias/data/gender/Female-Attribute.csv")

    ## models
    parser.add_argument("--model_path", "-m", default="/work/u3659277/s3-local/models/llama2-70b|tokenizer=ccw|data=l-v2|epoch=0-step=38656")
    # parser.add_argument("--model_path", "-m", default="MediaTek-Research/Breeze-7B-Base-v0_1") 
    
    
    ## config
    parser.add_argument("--origin_target", "-o", type=str, required=True)
    parser.add_argument("--replace_target", "-r", type=str, required=True)
    parser.add_argument("--calculate_perplexity", "-cppl", action="store_true")
    parser.add_argument("--api", action="store_true")
    parser.add_argument("--target_combination", "-comb", action="store_true")
    parser.add_argument("--prompt_template", "-p", type=str, default=None)
    
    args = parser.parse_args()
    main(args)