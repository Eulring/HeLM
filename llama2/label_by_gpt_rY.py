import random
import json
import requests
import time
import codecs
from hashlib import md5
import traceback
from utils_data import get_data, table_convert
import ipdb
from tqdm import tqdm



split = 'train'
resume = False
save_interval = 50
infer_model = "gpt-3.5-turbo"
debug_mode = False


def gpt_request_cr(table, query, answer, info1, info2, mode = 'full', max_tokens = 2048):
    # headers = signed_headers()
    if mode == 'full': table_text = table_convert(table, info1, info2, '3')
    else: table_text = table_convert(table, info1, info2, '3', limit_row = 10)
    prompt_input = """###Input:{} \n ##Question: {}\n ##Answer: {}\n ###Sub-table Row: """.format(table_text, query, answer)
    
    prompt = """You are a linguistic expert, and you need to determine which parts of the row data in table are relevant to the question and answer.
        I will first give you two examples of input and Sub-table Row output. Note that your output format can only be 'row(*)' !!!."""

    prompt1 = """
        ###Input:
        Table caption: University of Oregon Admissions
        col: - | 2014 | 2013 | 2012 | 2011 | 2010
        row 1: Applicants | 21,359 | 21,938 | 21,263 | 23,012 | 18,515
        row 2: Admits | 15,997 | 16,206 | 15,770 | 16,790 | 14,588
        row 3: % Admitted | 74.9 | 73.9 | 74.2 | 73.0 | 78.8
        row 4: Avg GPA | 3.58 | 3.60 | 3.57 | 3.59 | 3.52
        row 5: Enrolled | 3,961 | 3,966 | 4,031 | 4,167 | 3,978
        row 6: SAT range* | 990–1230 | 990–1240 | 991–1224 | 993–1223 | 991–1218

        ##Question: How many students were accepted from the 21,359 people who applied for the University of  Oregon in 2014 and how many students enrolled?
        ##Answer: For students entering University of Oregon 2014, 15,997 freshmen were accepted out of 21,359 applicants, a 74.9% acceptance rate, and 3,961 enrolled.
        ###Sub-table Row: row(1, 2, 3, 5)
    """

    prompt2 = """
        ###Input:
        Table caption: Annette Taddeo Early elections, 2008–2016
        col: Party | Party | Candidate | Votes | %
        row 1: - | Republican | Rick Scott/Carlos López-Cantera | 2,865,343 | 48.1%
        row 2: - | Democratic | Charlie Crist/Annette Taddeo | 2,801,198 | 47.1%
        row 3: - | Libertarian | Adrian Wyllie/Greg Roe | 223,356 | 3.8%
        row 4: - | No Party Affiliation | Glenn Burkett/Jose Augusto Matos | 41,341 | 0.7%
        row 5: - | No Party Affiliation | Farid Khavari/Lateresa A. Jones | 20,186 | 0.3%
        row 6: Total votes | Total votes | Total votes | 5,951,561 | -

        ##Question: What duo finished second in the election, what duo won the election, what percentage of vote did each duo receive, and what party was victorious?
        ##Answer: The Crist-Taddeo lost the election to Republican Rick Scott and Carlos López-Cantera, 48.1 to 47.1%.
        ###Sub-table Row: row(1, 2)
    """

    if debug_mode:
        print(prompt)
        
        ipdb.set_trace()

    if mode == 'full':
        messages = [
            { "role":"system", "content":prompt },
            { "role":"user", "content":prompt1 },
            { "role":"user", "content":prompt2 },
            { "role":"user", "content":prompt_input },
        ]
    if mode == 'short':
        messages = [
            { "role":"system", "content":prompt },
            { "role":"user", "content":prompt1 },
            { "role":"user", "content":prompt_input },
        ]
    body = {
        "messages": messages,
        "model": infer_model,
        "maxTokens": max_tokens,
        "temperature": 1,
        "topP": 1,
        "stop": None,
        "presencePenalty": 0,
        "frequencyPenalty": 0
    }
    # r = requests.post(url, json=body, headers=headers)
    try:
        raw = json.loads(r.text)
        r = raw['detail']['choices'][0]['message']['content']
    except:
        r = 'Fail'
        print(raw)

    return r


if __name__=="__main__":
    tar_path = '../datasets/FeTaQA/data_r/' + split + '.json'
    print(tar_path)

    if resume: data = json.load(open(tar_path))
    else: data = get_data(split = split, usage = 'raw', data_name='fetaqa')

    
    for cnt in tqdm(range(len(data))):
        if infer_model in data[cnt] and data[cnt][infer_model] != 'Fail': continue

        d = data[cnt]
        table, query, answer, info1, info2 = d['table'], d['instruction'], d['output'], d['info1'], d['info2']

        if infer_model in data[cnt] and data[cnt][infer_model] == 'Fail':
            res = gpt_request_cr(table, query, answer, info1, info2, mode = 'short')
        else:
            res = gpt_request_cr(table, query, answer, info1, info2)
            if res == 'Fail':
                res = gpt_request_cr(table, query, answer, info1, info2, mode = 'short')
                if res == 'Fail':
                    res = 'row()'
        
        data[cnt][infer_model] = res

        if (cnt + 1) % save_interval == 0:
            with open(tar_path, 'w') as of:
                json.dump(data, of)

        if debug_mode:
            print(res)
            ipdb.set_trace()

    with open(tar_path, 'w') as of:
        json.dump(data, of)
