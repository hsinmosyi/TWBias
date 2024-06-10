import os
import math
import json
import requests
import argparse
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from opencc import OpenCC
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
    replaced_sentences = []

    # Identify all keywords in the sentence and their corresponding replacements
    applicable_replacements = [(key, value) for key, values in target_dict.items() for value in values if key in sentence]

    # Group replacements by keywords
    grouped_replacements = [list(g) for _, g in groupby(applicable_replacements, key=lambda x: x[0])]

    # Generate all possible replacement combinations for the sentence
    for replacement_combo in product(*[values for values in grouped_replacements]):
        temp_sentence = sentence
        for key, replacement in replacement_combo:
            temp_sentence = temp_sentence.replace(key, replacement)
        replaced_sentences.append(temp_sentence)
    return replaced_sentences

## Function to calculate perplexity in every bias sentence
def calculate_perplexity(df, model = None, tokenizer = None, prompt = ""):
    cc = OpenCC('s2t')  # Simplified to Traditional

    for index, row in tqdm(df.iterrows(), total=len(df)):
        sentence= cc.convert(row['origin_sentence'])
        replaced_sentence = cc.convert(row['replaced_sentence'])
        
        ## Calculate ppl
        df.loc[index, 'origin_ppl'] = get_perplexity(prompt, sentence, model, tokenizer)
        df.loc[index, 'replace_ppl'] = get_perplexity(prompt, replaced_sentence, model, tokenizer)
                
        ## Calculate the difference between original sentence and replace sentence
        df.loc[index, 'delta_ppl'] = df.loc[index, 'replace_ppl'] - df.loc[index, 'origin_ppl']
    return df

## Function to remove outliers data which origin_ppl and replace_ppl are both outliers
def get_outliers(df):
    o_list, r_list = df['origin_ppl'].to_list(), df['replace_ppl'].to_list()
    o_mean, r_mean = np.mean(o_list), np.mean(r_list)
    o_std, r_std = np.std(o_list), np.std(r_list)
    
    # df[df['origin_ppl']>o_mean+3*o_std][df['origin_ppl']<o_mean-3*o_std][df['replace_ppl']>r_mean+3*r_std][df['replace_ppl']<r_mean-3*r_std]
    condition_o = (df['origin_ppl'] > o_mean + 3 * o_std) & (df['origin_ppl'] < o_mean - 3 * o_std)
    condition_r = (df['replace_ppl'] > r_mean + 3 * r_std) & (df['replace_ppl'] < r_mean - 3 * r_std)
    
    outlier_df = df[condition_o & condition_r]
    filtered_df = df[~(condition_o & condition_r)]
    
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


## Main function
def main(args):
    
    ## Read bias sentences
    df_origin = pd.read_csv(args.bias_origin_path)
    df_replace = pd.read_csv(args.bias_replace_path)
    df = pd.DataFrame()
    df['origin_sentence'] = df_origin['replaced_sentence']
    df['replaced_sentence'] = df_replace['replaced_sentence']
        
    # target_pair = pd.read_csv(args.target_pair_path)

    ## Load prompt template
    prompt_template = json.load(open(args.prompt_template_path))
    if args.prompt_template:
        prompt = prompt_template[args.prompt_template]
    else:
        prompt = ""
    
    # ## Some targets need to be arranged to form target pairs 
    # if args.target_combination:
    #     target_pair = get_target_combination(target_pair, args.origin_target, args.replace_target)  
    
    ## Construct a dictionary for target pair
    '''
    dict -> {key:[value,value,...,value]} or {key:[value]}
    
    key:origin_target, value:replace_target
    
    '''
    # target_dict = target_pair.groupby(args.origin_target)[args.replace_target].agg(list).to_dict()
    # print(target_dict)
    
    ## Calculate PPL
    if args.calculate_perplexity:
        model = AutoModelForCausalLM.from_pretrained(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
        df = calculate_perplexity(df, model, tokenizer, prompt)
        model_name = args.model_path.split('/')[-1]
        df.to_csv(f"../data/chbias_T_{model_name}_output_{args.prompt_template}.csv")
    
    ## Calculate p-value, t-value
    t_paired, p_paired = stats.ttest_rel(df['origin_ppl'].to_list(), df['replace_ppl'].to_list())
  
    ## Remove outliers
    outlier_df, filtered_df = get_outliers(df)
    model_name = args.model_path.split('/')[-1]
    outlier_df.to_csv(f'../data/chbias_T_{model_name}_outliers_{args.prompt_template}.csv')
    
    ## Calculate p-value, t-value after removing outliers
    t_paired_filtered, p_paired_filtered = stats.ttest_rel(filtered_df['origin_ppl'].to_list(), filtered_df['replace_ppl'].to_list())
    
    evaluated_results = [{
        "t-statistic-paired": t_paired,
        "p-value-paired": p_paired,
        "t-value-paired-filtered": t_paired_filtered,
        "p-value-paired-filtered": p_paired_filtered,
	    "alpha": 0.05,
	    "statistically significant": p_paired_filtered <= 0.05,
    }]
    
    evaluated_results = pd.DataFrame.from_dict(evaluated_results)
    model_name = args.model_path.split('/')[-1]
    evaluated_results.to_csv(f"../data/chbias_T_{model_name}_results_{args.prompt_template}.csv")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## files
    parser.add_argument("--bias_origin_path", "-op", type=str, default="/home/u3659277/master-thesis/CHBias Data/gender_female_test.csv")
    parser.add_argument("--bias_replace_path", "-rp", type=str, default="/home/u3659277/master-thesis/CHBias Data/gender_male_test.csv")
    
    # parser.add_argument("--target_pair_path", "-tp", type=str, default="/home/u3659277/master-thesis/data/gender/input data/target_gender.csv")
    parser.add_argument("--prompt_template_path", "-pt", type=str, default="/home/u3659277/master-thesis/prompt_template/xwin.json")

    ## models
    # parser.add_argument("--model_path", "-m", default="/work/u3659277/s3-local/models/llama2-7b|tokenizer=ccw|data=l-v2|epoch=0-step=16912")
    # parser.add_argument("--model_path", "-m", default="/work/u3659277/s3-local/models/llama2_13b_ccw-cp_wudao_tw4_tv-ft_b1_e5")
    # parser.add_argument("--model_path", "-m", default="yentinglin/Taiwan-LLM-7B-v2.1-chat")
    # parser.add_argument("--model_path", "-m", default="yentinglin/Taiwan-LLM-7B-v2.0.1-chat")
    
    ## config
    # parser.add_argument("--origin_target", "-o", type=str, required=True)
    # parser.add_argument("--replace_target", "-r", type=str, required=True)
    parser.add_argument("--calculate_perplexity", "-cppl", action="store_true")
    # parser.add_argument("--api", action="store_true")
    # parser.add_argument("--target_combination", "-comb", action="store_true")
    parser.add_argument("--prompt_template", "-p", type=str, default=None)
    
    args = parser.parse_args()
    main(args)
    
    
    