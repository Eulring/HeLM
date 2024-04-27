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

from llama_patch import unplace_flash_attn_with_attn
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from prompts import INFERENCE_SUMMARIZATION_PROMPT_v2

from utils import EvaluateTool
from utils_data import get_data
warnings.filterwarnings("ignore")


def main(args):

    prompts, evidataset = get_data(split = 'test', usage = 'evi_infer')
    # output_file_name = './data/fetaqa/infered_evi/llama2/test.json'
    output_file_name = './data/fetaqa/infered_evi/{}/test.json'.format(args.stage)
    print(output_file_name)
    print(len(prompts))


    experiment = args.experiment_dir
    peft_model_id = f"{experiment}/{args.adapter_dir}"

    # unpatch flash attention
    unplace_flash_attn_with_attn()

    if 'evi' not in peft_model_id:
        icl_prompt = """ Below is an Example: 
        ### Table: Table caption: University of Oregon Admissions
        col: - | 2014 | 2013 | 2012 | 2011 | 2010
        row 1: Applicants | 21,359 | 21,938 | 21,263 | 23,012 | 18,515
        row 2: Admits | 15,997 | 16,206 | 15,770 | 16,790 | 14,588
        row 3: % Admitted | 74.9 | 73.9 | 74.2 | 73.0 | 78.8
        row 4: Avg GPA | 3.58 | 3.60 | 3.57 | 3.59 | 3.52
        row 5: Enrolled | 3,961 | 3,966 | 4,031 | 4,167 | 3,978
        row 6: SAT range* | 990–1230 | 990–1240 | 991–1224 | 993–1223 | 991–1218
        ### Query: How many students were accepted from the 21,359 people who applied for the University of  Oregon in 2014 and how many students enrolled?
        ### Output: 1,2,3,5 \n now given input: """
        for i in range(len(prompts)):
            lines = prompts[i].split("\n")
            lines[0] = lines[0] + icl_prompt
            nprompt = '\n'.join(lines)
            prompts[i] = nprompt
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.experiment_dir,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            use_cache=False,
            device_map="auto",
        )
        model.config.pretraining_tp = 1
        # if use_flash_attention:
        #     from llama_patch import forward
        #     assert (
        #         model.model.layers[0].self_attn.forward.__doc__ == forward.__doc__
        #     ), "Model is not using flash attention"

        tokenizer = AutoTokenizer.from_pretrained(args.experiment_dir)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    else:
        # load base LLM model and tokenizer
        model = AutoPeftModelForCausalLM.from_pretrained(
            peft_model_id,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            load_in_4bit=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(peft_model_id)

    results = []
    for i, prompt in tqdm(enumerate(prompts)):

        input_ids = tokenizer(
            prompt, return_tensors="pt", truncation=True
        ).input_ids.cuda()
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=200,
                do_sample=True,
                top_p=0.9,
                temperature=1e-2,
            )
            result = tokenizer.batch_decode(
                outputs.detach().cpu().numpy(), skip_special_tokens=True
            )[0]
            result = result[len(prompt) :]
            results.append(result)
            # ipdb.set_trace()

            n = len(evidataset[i]['table'])
            state = [0 for st_ in range(n)]
            state[0] = 1

            result = result.split(",")
            for ele in result:
                try: 
                    idx = int(ele)
                    state[idx] = 1
                except: idx = 0

            evidataset[i]['evi'] = {}  
            evidataset[i]['evi']['model'] = {'state': state, 'reward': 0.0}

    with open(output_file_name, 'w') as of:
        json.dump(evidataset, of)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", default="",)
    parser.add_argument("--adapter_dir", default="checkpoint-2000",)
    parser.add_argument("--stage", default='p0',)

    args = parser.parse_args()
    main(args)
