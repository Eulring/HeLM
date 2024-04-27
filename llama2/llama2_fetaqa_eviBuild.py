import argparse
import torch
import os
import pandas as pd
import evaluate
from datasets import load_dataset
import pickle
import warnings
from tqdm import tqdm
import ipdb
import json
import random
import numpy as np

from llama_patch import unplace_flash_attn_with_attn
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

from prompts import INFERENCE_SUMMARIZATION_PROMPT_v2

from utils import EvaluateTool
from utils_data import get_data, table_convert
warnings.filterwarnings("ignore")

evaluator = EvaluateTool(args=None)
evaluator.matrics = ['sacrebleu']


def s2i(state, table, query, info):
    ntable = [tb for st, tb in zip(state, table) if st == 1]
    table = table_convert(ntable, info)
    return "### Table:{}  ### Query:{}  ### Output:".format(table, query)


def reward(instruct, answer, model, tokenizer):
    input_ids = tokenizer(instruct, return_tensors="pt", truncation=True ).input_ids.cuda()

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids, max_new_tokens=100,
            do_sample=True, top_p=0.9, temperature=1e-2,)
        result = tokenizer.batch_decode(
            outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
        result = result[len(instruct) :]
        ans = evaluator.evaluate([result], [answer])
        return ans['sacrebleu']


def state_add_drop(state, table, query, info1, answer, model, tokenizer, add_step, drop_step):
    n = len(state)
    baseline_rd = reward(s2i(state, table, query, info1), answer, model, tokenizer)
    # print(baseline_rd)

    rw_cnt = 1
    for i_ in range(add_step):
        update = False
        for i in range(1, n):
            if state[i] == 1: continue
            state[i] = 1
            new_rd = reward(s2i(state, table, query, info1), answer, model, tokenizer)
            rw_cnt += 1

            if new_rd <= baseline_rd: 
                state[i] = 0
            else: 
                baseline_rd = new_rd
                update = True
                break
        print(baseline_rd)
        if update == False:
            break
    print(rw_cnt)
    print("add phrase end")
        
    for i_ in range(drop_step):
        update = False
        for i in range(1, n):
            if state[i] == 0: continue
            state[i] = 0
            new_rd = reward(s2i(state, table, query, info1), answer, model, tokenizer)
            rw_cnt += 1

            if new_rd < baseline_rd: 
                state[i] = 1
            else: 
                baseline_rd = new_rd
                update = True
                break
        print(baseline_rd)
        if update == False:
            break

    print("drop phrase end")
    print(rw_cnt)

    return state, baseline_rd


def state_init_1gram(table, answer):
    state = [0 for i in range(len(answer))]
    state[0] = 1

    for i in range(1, len(table)):
        for ele in table[i]:
            if ele in answer:
                state[i] = 1

    return state


def state_init_by1(state, table, query, info1, answer, model, tokenizer):
    rw_list = []
    for i in range(1, len(state)):
        state[i] = 1
        rd = reward(s2i(state, table, query, info1), answer, model, tokenizer)
        rw_list.append(-rd)
        state[i] = 0
    
    idxs = np.argsort(np.array(rw_list))
    nstate = [0 for i in range(len(state))]
    nstate[0] = 1
    nstate[idxs[0] + 1] = 1

    baseline_rd = reward(s2i(nstate, table, query, info1), answer, model, tokenizer)
    # print(baseline_rd)
    for i in range(1, len(idxs)):
        nstate[idxs[i] + 1] = 1
        rd = reward(s2i(nstate, table, query, info1), answer, model, tokenizer)
        if rd > baseline_rd: baseline_rd = rd
        else: nstate[idxs[i] + 1] = 0
    # print(baseline_rd)

    return nstate, baseline_rd


def main(args):
    # tables, queries, answers, info1s 
    dataset_ = get_data(split = 'test', usage = 'build_evi')
    output_file = './data/fetaqa/evi/{}/test{}_{}.json'.format(args.stage, args.start, args.end)
    print(output_file)

    experiment = args.experiment_dir
    peft_model_id = f"{experiment}/{args.adapter_dir}"

    # unpatch flash attention
    unplace_flash_attn_with_attn()
    model = AutoPeftModelForCausalLM.from_pretrained(
        peft_model_id,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(peft_model_id)

    # evaluator = EvaluateTool(args=None)
    results = []
    for i in tqdm(range(len(dataset_))):
        if i < args.start: continue
        if i >= args.end: continue

        sample = dataset_[i]
        table = sample['table']
        query = sample["instruction"]
        answer = sample['output']
        info1 = sample['info1']
        
        if args.evi_method == 'drop':
            state = [1 for i in range(len(table))]
            state, rwd = state_add_drop(state, table, query, info1, answer, model, tokenizer, 0, len(table)//2 + 3)
        
        if args.evi_method == 'add':
            state = [0 for i in range(len(table))]
            state[0] = 1
            state, rwd = state_add_drop(state, table, query, info1, answer, model, tokenizer, 10, 0)
            
        if args.evi_method == '1-gram':
            state = state_init_1gram(table, answer)
            print(len(state))
            print(sum(state))
            state, rwd = state_add_drop(state, table, query, info1, answer, model, tokenizer, 2,  2)

        if args.evi_method == 'n2':
            state = [0 for i in range(len(table))]
            state[0] = 1
            state, rwd = state_init_by1(state, table, query, info1, answer, model, tokenizer)

        results.append(rwd)
        if 'evi' not in dataset_[i]: dataset_[i]['evi'] = {}
        if args.evi_method not in dataset_[i]['evi']:
            dataset_[i]['evi'][args.evi_method] = {'state': state, 'reward':rwd}


    # print(results)
    with open(output_file, 'w') as of: json.dump(dataset_, of)
    print(sum(results)/len(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_dir", default="",)
    parser.add_argument("--adapter_dir", default="checkpoint-2000",)
    parser.add_argument("--evi_method", default="1-gram",)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=7000, type=int)
    parser.add_argument("--stage", default='p0',)

    args = parser.parse_args()
    main(args)
