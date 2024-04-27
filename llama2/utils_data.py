from datasets import load_metric
import nltk
from datasets import load_dataset
import datasets
import ipdb
import json
import pandas as pd
import os
from tqdm import tqdm

def get_prompt2(table, query):
    # prompt1 = 
    pm = "You are required to Select relevant sub-columns,rows in the given table that support or oppose the Query. First i will show you an example: "
    pm += " ##Table_caption : south wales derby [SEP]"
    pm += " col: competition | total matches | cardiff win | draw | swansea win [SEP]"
    pm += " row 1: league | 55 | 6/19 | 16 | 20 [SEP]"
    pm += " row 2: fa cup | 2 | 6/0 | 27 | 2 [SEP]"
    pm += " row 3: league cup | 5 | 6/2 | 0 | 3 [SEP]"
    pm += " row 4: eu cup | 12 | 11/3 | 1 | 1 [SEP]"
    pm += " ##Query: there are no cardiff wins that have a draw greater than 26 or smaller than 1. [SEP]"
    pm += " ##relevant Sub-table Column,Row: col(cardiff win, draw), row(2, 3)"
    pm_ict = pm
    # pm += "## Above is an example [SEP]"
    pm = " \n\n Now, given new input:"
    pm += (" ##" + table) 
    pm += (" ##Query: " + query + " [SEP]")
    pm += " ##relevant Sub-table Column,Row: "
    pm_input = pm
    # pm += " ##Sub-table Column,Row: "
    pm = pm_ict + pm_input
    # ipdb.set_trace()
    return pm


def get_prompt(table, query):
    # prompt1 = 
    pm = "###Instruction: Select relevant columns / rows in the given table that support or oppose the statement. \n\n"
    pm += " ##Table_caption : south wales derby [SEP]"
    pm += " col: competition | total matches | cardiff win | draw | swansea win [SEP]"
    pm += " row 1: league | 55 | 6/19 | 16 | 20 [SEP]"
    pm += " row 2: fa cup | 2 | 6/0 | 27 | 2 [SEP]"
    pm += " row 3: league cup | 5 | 6/2 | 0 | 3 [SEP]"
    pm += " ##Question: there are no cardiff wins that have a draw greater than 26 or smaller than 1. [SEP]"
    pm += " ##Sub-table Column,Row: col(cardiff win, draw), row(2, 3) \n\n"
    
    # pm += "## Above is an example [SEP]"
    # pm += "##Instruction: Select relevant columns / rows in the given table that support or oppose the statement. [SEP]"
    pm += (" ##" + table) 
    pm += ("##Question: " + query + " [SEP] ")
    pm += " ##Sub-table Column,Row: "

    # ipdb.set_trace()
    return pm


def prepare_inferevi_prompt_fetaqa(tables, infos1, infos2, queries):
    prompts = [] 
    texts = []
    for table, info1, info2, query in zip(tables, infos1, infos2, queries):
        table_text = table_convert(table, info1, info2, mode = '3')
        prompts.append(get_prompt2(table_text, query))
        texts.append(table_text)
    return prompts, texts

# def prepare_evicr_data_fetaqa(tables, infos1, infos2, queries, gpt_res):
#     sub_tables = [] 
#     for table, info1, info2, query, res in tqdm(zip(tables, infos1, infos2, queries, gpt_res)):
#         try: sub_table = parse_gpt_res(table, res)
#         except: sub_table = table
#         # table_text = table_convert(sub_table, info1, info2, mode = '3')
#         sub_tables.append(sub_table)

#     return sub_tables

# def prepare_evic_data_fetaqa(tables, infos1, infos2, queries, gpt_res):
#     sub_tables = [] 
#     for table, info1, info2, query, res in tqdm(zip(tables, infos1, infos2, queries, gpt_res)):
#         try: sub_table = parse_gpt_res(table, res)
#         except: sub_table = table
#         sub_tables.append(sub_table)

#     return sub_tables


def prepare_test_prompt(tables, queries, texts, task='featqa'):
    instructions = []
    if task == 'fetaqa': prompt =  """Your task is to output the answer given Table and Query.
### Table:{table}  ### Query: {query}  ### Output:"""
    if task == 'tabfact': prompt =  """Your task is to output whether the Query is True or False given the Table.
### Table:{table}  ### Query: {query}  ### Output:"""
    if task == 'wtq': prompt =  """Your task is to output the answer given Table and Query.
### Table: {table}  ### Query: {query}  ### Output:"""
    if task == 'qtsumm': prompt =  """Your task is to summary the Table given Query
### Table: {table}  ### Query: {query}  ### Output:"""
    for table, query, text in zip(tables, queries, texts):
        example = prompt.format(table=table, query=query)
        instructions.append(example)
    return instructions

def prepare_train_prompt(tables, queries, texts, task='featqa'):
    instructions = []
    if task == 'fetaqa': prompt =  """Your task is to output the answer given Table and Query.
### Table:{table}  ### Query: {query}  ### Output: {text}"""
    if task == 'tabfact': prompt =  """Your task is to output whether the Query is True or False given the Table.
### Table:{table}  ### Query: {query}  ### Output: {text}"""
    if task == 'wtq': prompt =  """Your task is to output the answer given Table and Query.
### Table: {table}  ### Query: {query}  ### Output: {text}"""
    if task == 'qtsumm': prompt =  """Your task is to summary the Table given Query
### Table: {table}  ### Query: {query}  ### Output: {text}"""
    for table, query, text in zip(tables, queries, texts):
        example = prompt.format(table=table, query=query, text=text)
        instructions.append(example)
    return instructions

def prepare_train_he_prompt_fetaqa(tables, states, queries, infos, answers, task = 'fetaqa'):
    prompts = []
    if task == 'fetaqa': prompt =  """Your task is to output the answer given Table and Query. Relative table units to query are surrounded by '*'.
### Table:{table}  ### Query: {query}  ### Output: {text}"""
    if task == 'qtsumm':  prompt =  """Your task is to summary the Table given Query. Relative table units to query are surrounded by '*'.
### Table: {table}  ### Query:{query}  ### Output: {text}"""
    for table, query, answer, info, state in zip(tables, queries, answers, infos, states):
        table_text = table_convert(table, info, info2=state, mode='4')
        example = prompt.format(table=table_text,query=query,text=answer)
        prompts.append(example)
    # ipdb.set_trace()
    return prompts

def prepare_test_he_prompt_fetaqa(tables, states, queries, infos, answers, task = 'fetaqa'):
    prompts = []
    if task == 'fetaqa': prompt =  """Your task is to output the answer given Table and Query. Relative table units to query are surrounded by '*'.
### Table:{table}  ### Query:{query}  ### Output:"""
    if task == 'qtsumm':  prompt =  """Your task is to summary the Table given Query. Relative table units to query are surrounded by '*'.
### Table: {table}  ### Query:{query}  ### Output:"""
    for table, query, answer, info, state in zip(tables, queries, answers, infos, states):
        table_text = table_convert(table, info, info2=state, mode='4')
        example = prompt.format(table=table_text,query=query)
        prompts.append(example)
    return prompts

def prepare_test_st_prompt_fetaqa(tables, states, queries, infos, answers, task = 'fetaqa'):
    prompts = []
    if task == 'fetaqa': prompt =  """Your task is to output the answer given Table and Query.
### Table:{table}  ### Query:{query}  ### Output:"""
    if task == 'qtsumm':  prompt =  """Your task is to summary the Table given Query.
### Table: {table}  ### Query:{query}  ### Output:"""
    for table, query, answer, info, state in zip(tables, queries, answers, infos, states):
        table_text = table_convert(table, info, info2=state, mode='5')
        example = prompt.format(table=table_text,query=query)
        prompts.append(example)
    return prompts

def prepare_evitrain_prompt_fetaqa(tables, states, queries, infos):
    def state_convert(st):
        idxs = [str(i) for i in range(1, len(st)) if st[i] == 1]
        return ",".join(idxs)

    prompts = []
    prompt =  """You are an expert table reasoner, your task is to output the relative rows ids which might be helpful for answering the query.
### Table:{table}  ### Query:{query}  ### Output:{output}"""
    for table, state, query, info in zip(tables, states, queries, infos):
        # ntable = [table[i] for i in range(len(table)) if state[i] == 1]
        table_text = table_convert(table, info, mode='3')
        state_text = state_convert(state)
        prompts.append(prompt.format(table=table_text, query=query, output=state_text))
        # ipdb.set_trace()
    return prompts

def prepare_eviinfer_prompt_fetaqa(tables, queries, infos):
    prompts = []
    prompt =  """You are an expert table reasoner, your task is to output the relative rows ids which might be helpful for answering the query.
### Table:{table}  ### Query:{query}  ### Output:"""
    for table, query, info in zip(tables, queries, infos):
        table_text = table_convert(table, info, mode='3')
        prompts.append(prompt.format(table=table_text, query=query))
    return prompts


def table_convert(tb, info1, info2 = None, mode = '2', limit_row = -1):
    # tb = data['table_array']
    if limit_row != -1: tb = tb[:limit_row]
    if mode == '1':
        st = "Table caption: " + info1
        st += '\n'
        for row in tb:
            for i, ele in enumerate(row):
                if i > 0: st += ' | '
                st += ele
            st += '\n'

    if mode == '2':
        st = "Table caption: " + info1
        st += ' [SEP] '
        for row in tb:
            for i, ele in enumerate(row):
                if i > 0: st += ' | '
                st += ele
            st += ' [SEP] '

    if mode == '3':
        st = "Table caption: " + info1
        for j, row in enumerate(tb):
            if j == 0: st += "\n  col: "
            else: st += "   row {}: ".format(str(j))
            for i, ele in enumerate(row):
                if i > 0: st += ' | '
                st += ele
            st += '\n'
            
    if mode == '4':
        st = "Table caption: " + info1
        st += ' [HEADER] '
        info2[0] = 0
        for hst, row in zip(info2, tb):
            for i, ele in enumerate(row):
                if i > 0:
                    if hst == 1: st += "*|*"
                    else: st += ' | '
                st += ele
            st += ' [SEP] '

    if mode == '5':
        st = "Table caption: " + info1
        st += ' [HEADER] '
        info2[0] = 1
        for hst, row in zip(info2, tb):
            if hst == 0: continue
            for i, ele in enumerate(row):
                if i > 0: st += ' | '
                st += ele
            st += ' [SEP] '

    return st


def get_data(data_name = 'fetaqa', split = 'train', usage = 'train', info = 'None', ds = False):

    assert data_name in ['fetaqa', 'tabfact', 'wtq', 'qtsumm']
    
    file_path = '../datasets/&&&/data_hf/hf_###.json'
    if data_name == 'fetaqa': file_path = file_path.replace('&&&', 'FeTaQA')
    if data_name == 'tabfact': file_path = file_path.replace('&&&', 'TabFact')
    if data_name == 'wtq': file_path = file_path.replace('&&&', 'WTQ')
    if data_name == 'qtsumm': file_path = file_path.replace('&&&', 'QTSumm')
    
    if usage == 'test_gpthighlight_evi': file_path = '../datasets/FeTaQA/data_r/###.json'
    if usage in ['test_ggpthighlight_evi', 'train_ggpthighlight_evi']: 
        if data_name == 'fetaqa': file_path = '../datasets/FeTaQA/data_cr/###.json'
        if data_name == 'qtsumm': file_path = '../datasets/QTSumm/data_r/###.json'
    if usage == 'build_evi_s2_gpt':
        if data_name == 'fetaqa': file_path = '../datasets/FeTaQA/data_cr/###.json'
        if data_name == 'qtsumm': file_path = '../datasets/QTSumm/data_r/###.json'
    if usage == 'build_evi_s2_gold':
        file_path = '../datasets/QTSumm/data_hf/hf_###.json'

    # loading dataset
    if usage in ['train_evi', 'train_highlight_evi', 'test_highlight_evi', 'build_evi_s2', 'test_pgoldhighlight_evi', 'test_subtable_evi', 'train_mergehighlight_evi', 'train_mergeevi', 'test_mergehighlight_evi', 'test_gmergehighlight_evi']:
        # ipdb.set_trace()
        if usage in ['train_evi', 'train_highlight_evi', 'build_evi_s2', 'test_pgoldhighlight_evi', 'test_gmergehighlight_evi', 'train_mergehighlight_evi', 'train_mergeevi']:
            if 'merge' in usage: info = 'p1'
            if data_name == 'fetaqa': file_root = 'data/fetaqa/evi/{}'.format(info)
            if data_name == 'qtsumm': file_root = 'data/qtsumm/evi/{}'.format(info)

        if usage in ['test_highlight_evi', 'test_subtable_evi', 'test_mergehighlight_evi']:
            if data_name == 'fetaqa': file_root = 'data/fetaqa/infered_evi/p0'
            if data_name == 'qtsumm': file_root = 'data/qtsumm/infered_evi/p2'
            if 'merge' in usage: file_root = 'data/fetaqa/infered_evi/p2'

        ipdb.set_trace()
        file_names = os.listdir(file_root)
        dataset = []
        for fname in file_names:
            if split in fname and '.json' in fname:
                fpath = os.path.join(file_root, fname)
                sub_dataset_ = load_dataset("json", data_files=fpath)
                sub_dataset = sub_dataset_["train"]
                # ipdb.set_trace()
                if len(dataset) == 0: 
                    dataset = [ele.copy() for ele in sub_dataset]
                else: 
                    assert len(dataset) == len(sub_dataset)
                    for i in range(len(sub_dataset)):
                        
                        if sub_dataset[i]['evi'] == None: 
                            continue
                        if dataset[i]['evi'] == None: 
                            dataset[i]['evi'] = sub_dataset[i]['evi'].copy()
                        else:
                            # ipdb.set_trace()
                            for key in sub_dataset[i]['evi'].keys():
                                if key not in dataset[i]['evi']:
                                    dataset[i]['evi'][key] = sub_dataset[i]['evi'][key]
        
        if usage == 'build_evi_s2': return dataset
        sidx = [i for i in range(len(dataset)) if dataset[i]['evi'] != None]
        state, table, query, info, answer = [], [], [], [], []
        # ipdb.set_trace()
        for idx in sidx:
            try:
                if 'merge' in usage:
                    state.append(dataset[idx]['evi']['merge']['state'])
                else: state.append(dataset[idx]['evi']['n2']['state'])
            except:
                # ipdb.set_trace()
                state.append(dataset[idx]['evi']['model']['state'])
            table.append(dataset[idx]['table'])
            query.append(dataset[idx]['instruction'])
            info.append(dataset[idx]['info1'])
            answer.append(dataset[idx]['output'])
    else:
        file_path = file_path.replace("###", split)
        dataset_ = load_dataset("json", data_files=file_path)
        dataset = dataset_["train"]
    
        table_text = dataset["input"]
        table = dataset['table']
        query = dataset["instruction"]
        answer = dataset['output']
        info1 = dataset['info1']
        info2 = dataset['info2']

        if usage in ['train_goldhighlight_evi', 'test_goldhighlight_evi', 'build_evi_s2_gold']:
            state = [ele['gold']['state'] for ele in dataset['evi']]
            info = info1
        if usage in ['test_gpthighlight_evi', 'test_subtable_evi', 'train_ggpthighlight_evi', 'build_evi_s2_gpt', 'test_ggpthighlight_evi']:
            response = dataset['gpt-3.5-turbo']
            state = parse_gpt(table, response)
            info = info1
        # ipdb.set_trace()
    if usage in ['build_evi_s2_gold']:
        return dataset, state
    if usage in ['build_evi_s2_gpt', 'build_evi_s2_gold']:
        return state

    if usage == 'train':
        prompts = prepare_train_prompt(table_text, query, answer, data_name)
        train_dataset = datasets.Dataset.from_pandas(
            pd.DataFrame(data={"instructions": prompts}))
        return train_dataset
    
    if usage in ['train_highlight_evi', 'train_goldhighlight_evi', 'train_mergehighlight_evi', 'train_ggpthighlight_evi']:
        prompts = prepare_train_he_prompt_fetaqa(table, state, query, info, answer, data_name)
        train_dataset = datasets.Dataset.from_pandas(
            pd.DataFrame(data={"instructions": prompts}))
        return train_dataset
    
    if usage in ['test_highlight_evi', 'test_goldhighlight_evi', 'test_gpthighlight_evi', 'test_mergehighlight_evi', 'test_pgoldhighlight_evi', 'test_gmergehighlight_evi', 'test_ggpthighlight_evi']:
        prompts = prepare_test_he_prompt_fetaqa(table, state, query, info, answer, data_name)
        if ds: return prompts[:200], answer[:200]
        return prompts, answer
    
    if usage in ['test_subtable_evi']:
        prompts = prepare_test_st_prompt_fetaqa(table, state, query, info, answer, data_name)
        if ds: return prompts[:200], answer[:200]
        return prompts, answer

    if usage in ['train_evi', 'train_mergeevi']:
        prompts = prepare_evitrain_prompt_fetaqa(table, state, query, info)
        train_dataset = datasets.Dataset.from_pandas(
            pd.DataFrame(data={"instructions": prompts}))
        return train_dataset
        
    if usage == 'evi_infer':
        dataset = json.load(open(file_path, 'r'))
        prompts = prepare_eviinfer_prompt_fetaqa(table, query, info1)
        return prompts, dataset

    if usage == 'build_evi':
        dataset = json.load(open(file_path, 'r'))
        return dataset
    
    if usage in ['test', 'test_evi_f']:
        prompts = prepare_test_prompt(table_text, query, answer, data_name)
        if ds: return prompts[:100], answer[:100]
        return prompts, answer
    
    if usage == 'raw':
        dataset = json.load(open(file_path, 'r'))
        if ds: return dataset[:200]
        return dataset

    if ds: return train_instructions[:200], text[:200]
    return train_instructions, text







































def parse_gpt(tbs, ress):
    states = []
    for res, tb in zip(ress, tbs):
        state = [0 for i in range(len(tb))]
        state[0] = 1
        i = 0
        while i < len(res):
            sres = res[i:]
            if sres.startswith('row('):
                right = 0
                for j in range(len(sres)):
                    if sres[j] == ')': 
                        right = j
                        break
                if right == 0:
                    right = len(sres) - 1 
                    # ipdb.set_trace()
                try:
                    ssres = sres[4:right].replace(" ", "").replace("-",',').split(',')
                    row = [int(e_) for e_ in ssres]
                except: 
                    row = []
                for rid in row: 
                    if rid < len(tb): state[rid] = 1

                i += max(right, 4)
            else: i+=1
        
        # ipdb.set_trace()
        states.append(state)
    return states




def parse_gpt_res(tb, res):
    row, col, col_name = [], [], []
    i = 0
    while i < len(res):
        sres = res[i:]
        if sres.startswith('col['):
            right = 0
            for j in range(len(sres)):
                if sres[j] == ']': 
                    right = j
                    break
            # if right == 0: ipdb.set_trace()
            ssres = sres[4:right].split(', ')
            col_name.extend(ssres)
            i += max(right, 4)
        elif sres.startswith('row['):
            right = 0
            for j in range(len(sres)):
                if sres[j] == ']': 
                    right = j
                    break
            # if right == 0: ipdb.set_trace()
            ssres = sres[4:right].replace(" ", "").replace("-",',').split(',')
            row.extend([int(e_) for e_ in ssres])
            i += max(right, 4)
        else: 
            i += 1

    n = len(tb[0])
    m = len(tb)

    if len(col_name) == 0: col_name = tb[0]
    if len(row) == 0: row = [i+1 for i in range(m-1)]

    ncol_name = []
    for i in range(n):
        if tb[0][i] in col_name: 
            col.append(i)
            ncol_name.append(tb[0][i])
    
    ntb = [ncol_name] 
    for i in range(m):
        if i in row:
            tb_row = []
            for j in range(n):
                if j in col: tb_row.append(tb[i][j])
            ntb.append(tb_row)

    return ntb