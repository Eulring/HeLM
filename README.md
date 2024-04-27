Before conducting finetuning, you need to downlaod the llama2-XXB-hf model checkpoints from huggingface

This project is modified on https://github.com/georgian-io/LLM-Finetuning-Hub, requirements can be found in this project.



## Training

### Training LLaMA2-13B-Lora:
```
python llama2_train.py \
    --pretrained_ckpt  $PATH_OF_LLAMA2$ \
    --lora_r 16 --epochs 10 --dropout 0.1 --save_step 1000 --table_mode raw
```

### Traing HeLM 

Step1: get Egpt and train a feedback model
```
python llama2_train.py \
    --pretrained_ckpt  $PATH_OF_LLAMA2$ \
    --lora_r 16 --epochs 10 --dropout 0.1 --save_step 1000 --table_mode train_ggpthighlight_evi
```
If you want to call ChatGPT API by yourself to get Egpt, we also provide the scrip: label_by_gpt_rY.py 




Step2: Get Emerge
```
Build searching evidence: 
python llama2_fetaqa_eviBuild.py \
--experiment_dir $PATH_OF_FEEDBACK_MODEL$ --evi_method n2

Build merged evidence:
python llama2_fetaqa_eviBuild-merge.py \
--experiment_dir $PATH_OF_FEEDBACK_MODEL$
```



Step3: train reasoner ans summarizer
```
python llama2_fetaqa_evi_train.py \
    --pretrained_ckpt $PATH_OF_LLAMA2$ \
    --lora_r 16 --dropout 0.1 --save_step 1000

python llama2_train.py \
    --pretrained_ckpt $PATH_OF_LLAMA2$ \
    --lora_r 16 --dropout 0.1 --save_step 1000 --table_mode train_mergehighlight_evi
```

## Evaluation

### HeLM Evaluation

Step1: reasoning by reasoner
```
python llama2_fetaqa_evi_infer.py --stage p2 --experiment_dir $PATH_OF_REASONER$
```

Step2: use summarizer generate output by highlighted evidence:
```
python llama2_fetaqa_test.py \
    --experiment_dir $PATH_OF_SUMMARIZER$
    --adapter_dir checkpoint-2000 --data_mode test_mergehighlight_evi
```

### LLaMA2-13B-Lora Evaluation
```
python llama2_fetaqa_test.py \
    --experiment_dir $PATH_OF_SUMMARIZER$
    --adapter_dir checkpoint-2000 --data_mode raw
```