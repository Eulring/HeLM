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

def reward(instruct, answer, model, tokenizer):
    input_ids = tokenizer(instruct, return_tensors="pt", truncation=True ).input_ids.cuda()

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids, max_new_tokens=400,
            do_sample=True, top_p=0.9, temperature=1e-2,)
        result = tokenizer.batch_decode(
            outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
        result = result[len(instruct) :]
        ans = evaluator.evaluate([result], [answer])
        # ipdb.set_trace()
        return ans['sacrebleu']


def main(args):
    # tables, queries, answers, info1s 
    split = 'test'
    if split == 'train':
        dataset = get_data(split=split, usage = 'build_evi_s2', info = 'p0')[:8300]
        states2 = get_data(split=split, usage = 'build_evi_s2_gpt', info = 'p0')
        states1 = []
        for i, ele in enumerate(dataset):
            if i < 8300: states1.append(ele['evi']['n2']['state'])

        prompts2, summaries = get_data(split=split, usage = 'test_ggpthighlight_evi', info = 'p0')
        prompts1_ = get_data(split=split, usage = 'train_highlight_evi', info = 'p0')
        prompts1 = []
        for prompt in prompts1_:
            # ipdb.set_trace()
            units = prompt['instructions'].split("### Output:")
            prompts1.append("{} ### Output: ".format(units[0]))
    
    if split == 'test':
        dataset = get_data(split=split, usage = 'build_evi_s2', info = 'p0')
        states2 = get_data(split=split, usage = 'build_evi_s2_gpt', info = 'p0')
        states1 = []
        for i, ele in enumerate(dataset):
            states1.append(ele['evi']['n2']['state'])

        prompts2, summaries = get_data(split=split, usage = 'test_ggpthighlight_evi', info = 'p0')
        prompts1_ = get_data(split=split, usage = 'train_highlight_evi', info = 'p0')
        prompts1 = []
        for prompt in prompts1_:
            units = prompt['instructions'].split("### Output:")
            prompts1.append("{} ### Output: ".format(units[0]))

    # ipdb.set_trace()
    output_file = './data/fetaqa/evi/{}/{}_{}_{}.json'.format(args.stage, split, args.start, args.end)
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
    # scores 
    for i in tqdm(range(len(dataset))):
        if i < args.start: continue
        if i >= args.end: continue

        answer = summaries[i]

        rwd1 = reward(prompts1[i], answer, model, tokenizer)
        rwd2 = reward(prompts2[i], answer, model, tokenizer)

        # nstate = states1[i]
        if rwd1 > rwd2: nstate = states1[i]
        if rwd1 < rwd2: nstate = states2[i]
        if rwd1 == rwd2: 
            if sum(states1[i]) < sum(states2[i]): nstate = states1[i]
            else: nstate = states2[i]
        dataset[i]['evi']['merge'] = {'state': nstate, 'reward':[rwd1, rwd2]}

        # print(rwd1)
        # print(rwd2)
        # results.append(rwd)


    # print(results)
    with open(output_file, 'w') as of: json.dump(dataset, of)
    # print(sum(results)/len(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_dir", default="",)
    parser.add_argument("--adapter_dir", default="checkpoint-2000")
    parser.add_argument("--evi_method", default="s2")
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=2003, type=int)
    parser.add_argument("--stage", default='p1',)

    args = parser.parse_args()
    main(args)
