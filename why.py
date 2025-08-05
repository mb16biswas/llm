import subprocess

# List of commands
commands = [
    "pip install datasets==2.21.0",
    "pip install transformers",
    "pip install sentence_transformers",
    "pip install evaluate",
    "pip install nltk rouge_score",
    "pip install sacrebleu",
    "pip install sacremoses",
    "pip install bert_score",

]

# Execute each command
for cmd in commands:
    subprocess.run(cmd, shell=True)


import torch
from datasets import load_metric
import evaluate
from evaluate import load
import pandas as pd
import os
import nltk
import numpy as np
import json
nltk.download('punkt_tab')

meteor = evaluate.load("meteor")
sari = load("sari")
rouge = load_metric("rouge",trust_remote_code=True)
bleu = load_metric("bleu",trust_remote_code=True)
sacrebleu = load_metric("sacrebleu",trust_remote_code=True)
bertscore = load("bertscore")

Path_Pred = [
    "/workspace/data/Momojit/Contract-QA2/rag-results2/cuad-vanila-saul-2-revised", 
    # "/workspace/data/Momojit/Contract-QA2/rag-results2/cuad-vanila-mis-2-revised", 
    # "/workspace/data/Momojit/Contract-QA2/rag-results2/cuad-vanila-llama-2-revised"
]

# ["/workspace/data/Momojit/Contract-QA2/rag-results2/cuad-llama-2" ,
#              "/workspace/data/Momojit/Contract-QA2/rag-results2/cuad-mis-2" ,
#              "/workspace/data/Momojit/Contract-QA2/rag-results2/cuad-saul-2" ,
#              "/workspace/data/Momojit/Contract-QA2/rag-results2/cuad-vanila-saul-2", 
#              "/workspace/data/Momojit/Contract-QA2/rag-results2/cuad-vanila-mis-2", 
#              "/workspace/data/Momojit/Contract-QA2/rag-results2/cuad-vanila-llama-2", 
#              "/workspace/data/Momojit/Contract-QA2/rag-results2/cuad-saul-col-bert-2-revised", 
#              "/workspace/data/Momojit/Contract-QA2/rag-results2/cuad-mis-col-bert-2-revised", 
#              "/workspace/data/Momojit/Contract-QA2/rag-results2/cuad-llama-col-bert-2-revised",  
#             "/workspace/data/Momojit/Contract-QA2/rag-results2/cuad-saul-ablation1-v-2" ,
#             "/workspace/data/Momojit/Contract-QA2/rag-results2/cuad-mis-ablation1-2" ,
#             "/workspace/data/Momojit/Contract-QA2/rag-results2/cuad-llama-ablation1-2" ,
#             "/workspace/data/Momojit/Contract-QA2/rag-results2/cuad-saul-ablation2-v-2" ,
#             "/workspace/data/Momojit/Contract-QA2/rag-results2/cuad-mis-ablation2-2" ,
#             "/workspace/data/Momojit/Contract-QA2/rag-results2/cuad-llama-ablation2-2" ,
#             "/workspace/data/Momojit/Contract-QA2/rag-results2/cuad-saul-ablation3-v-2" ,
#             "/workspace/data/Momojit/Contract-QA2/rag-results2/cuad-mis-ablation3-2" ,
#             "/workspace/data/Momojit/Contract-QA2/rag-results2/cuad-llama-ablation3-2" ,
#             "/workspace/data/Momojit/Contract-QA2/rag-results2/cuad-saul-ablation4-v-2" ,
#             "/workspace/data/Momojit/Contract-QA2/rag-results2/cuad-mis-ablation4-2" ,
#             "/workspace/data/Momojit/Contract-QA2/rag-results2/cuad-llama-ablation4-2" ,


# ]



# "/workspace/data/Momojit/Contract-QA2/rag-results2/cuad-saul-ablation1-2" ,
# "/workspace/data/Momojit/Contract-QA2/rag-results2/cuad-mis-ablation1-2" ,
# "/workspace/data/Momojit/Contract-QA2/rag-results2/cuad-llama-ablation1-2" ,
# "/workspace/data/Momojit/Contract-QA2/rag-results2/cuad-saul-ablation2-2" ,
# "/workspace/data/Momojit/Contract-QA2/rag-results2/cuad-mis-ablation2-2" ,
# "/workspace/data/Momojit/Contract-QA2/rag-results2/cuad-llama-ablation2-2" ,
# "/workspace/data/Momojit/Contract-QA2/rag-results2/cuad-saul-ablation3-2" ,
# "/workspace/data/Momojit/Contract-QA2/rag-results2/cuad-mis-ablation3-2" ,
# "/workspace/data/Momojit/Contract-QA2/rag-results2/cuad-llama-ablation3-2" ,
# "/workspace/data/Momojit/Contract-QA2/rag-results2/cuad-saul-ablation4-2" ,
# "/workspace/data/Momojit/Contract-QA2/rag-results2/cuad-mis-ablation4-2" ,
# "/workspace/data/Momojit/Contract-QA2/rag-results2/cuad-llama-ablation4-2" ,

Path_GT = "/workspace/data/Momojit/Contract-QA2/ground-truth-cuad-final"


def rouge_score(pred,truth):

    Precision1 = []
    Recall1 = []
    Fmeasure1 = []

    Precision2 = []
    Recall2 = []
    Fmeasure2 = []

    PrecisionL = []
    RecallL = []
    FmeasureL = []

    PrecisionLs = []
    RecallLs = []
    FmeasureLs = []

    for i,j in zip(pred,truth):

        res = rouge.compute(predictions=[i], references=[j])

        Precision1.append(res["rouge1"].mid.precision)
        Recall1.append(res["rouge1"].mid.recall)
        Fmeasure1.append(res["rouge1"].mid.fmeasure)

        Precision2.append(res["rouge2"].mid.precision)
        Recall2.append(res["rouge2"].mid.recall)
        Fmeasure2.append(res["rouge2"].mid.fmeasure)


        PrecisionL.append(res["rougeL"].mid.precision)
        RecallL.append(res["rougeL"].mid.recall)
        FmeasureL.append(res["rougeL"].mid.fmeasure)


        PrecisionLs.append(res["rougeLsum"].mid.precision)
        RecallLs.append(res["rougeLsum"].mid.recall)
        FmeasureLs.append(res["rougeLsum"].mid.fmeasure)

    return (Precision1, Recall1, Fmeasure1), (Precision2, Recall2, Fmeasure2), (PrecisionL, RecallL, FmeasureL), (PrecisionLs, RecallLs, FmeasureLs)


def bleu_score(pred,truth):

    Blue = []

    for i,j in zip(pred,truth):


        i = [i.split(" ")]
        j = [[j.split(" ")]]

        res = bleu.compute(predictions=i, references=j)['bleu']

        Blue.append(res)

    return Blue


def sacrebleu_score(pred,truth):

    Blue = []

    for i,j in zip(pred,truth):

        i = [i]
        j = [[j]]

        res = sacrebleu.compute(predictions=i, references=j)['score']
        Blue.append(res)

    return Blue


def meteor_score(pred,truth):

    Meteor = []

    for i,j in zip(pred,truth):

        i = [i]
        j = [j]

        res = meteor.compute(predictions=i, references=j)['meteor']
        Meteor.append(res)

    return Meteor


def sari_score(pred,truth):

    Sari = []

    for i,j in zip(pred,truth):

        i = [i]
        j = [j]

        res = sari.compute(sources = i , predictions=j, references=[j])['sari']
        Sari.append(res)

    return Sari


def bert_score(pred,truth):

    Bert_pre = []
    Bert_recall = []
    Bert_f1 = []

    for i,j in zip(pred,truth):

        i = [i]
        j = [j]

        res = bertscore.compute(predictions = i , references=j, model_type = "distilbert-base-uncased")

        Bert_pre.append(res['precision'][0])
        Bert_recall.append(res['recall'][0])
        Bert_f1.append(res['f1'][0])

    return Bert_pre, Bert_recall, Bert_f1




def create_folder(f_name):

    folder_path = "/workspace/data/Momojit/Contract-QA2/rag-results2/eval-res2/" + f_name

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("Folder  created successfully.")
    else:
        print(f"Folder already exists.")

    return folder_path


def find_csv_files(directory):
    csv_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return sorted(csv_files)





for p_ in Path_Pred:

    try:

        PREDS = os.listdir(p_)
        GT = os.listdir(Path_GT)
        files_names  = []

        for p in PREDS:

            if(p in GT and p.endswith(".csv")):

                files_names.append(p)
        

        name = p_.split("/")[-1]

        for f_ in files_names:

            pred_csv_path = os.path.join(p_,f_)
            target_csv_path= os.path.join(Path_GT,f_)

            print()
            print()
            print("pred_csv_path: ", pred_csv_path)
            print("target_csv_path: ", target_csv_path)
            print()
            print()

            df = pd.read_csv(pred_csv_path)
            df2 = pd.read_csv(target_csv_path)

            print()
            print(df)
            print()
        
            print()
            print(df2)
            print()
            
            
            questions = list(df2["Question"])

            key = df.keys()[-1]
            key2 = df2.keys()[-1]

            answers = list(df[key])
            truths = list(df2[key2])

            truths = [i if type(i) == str else "The Answer is not mentioned in the context" for i in truths] 

            (Precision1, Recall1, Fmeasure1), (Precision2, Recall2, Fmeasure2), (PrecisionL, RecallL, FmeasureL), (PrecisionLs, RecallLs, FmeasureLs) = rouge_score(answers,truths)

            Blue = bleu_score(answers,truths)

            Sac_Blue = sacrebleu_score(answers,truths)

            Meteor = meteor_score(answers,truths)

            Sari = sari_score(answers,truths)

            Bert_pre, Bert_recall, Bert_f1 = bert_score(answers,truths)


            df_ = {"Question" : questions,
            "Precision_Rouge1" : Precision1,
            "Recall_Rouge1" : Recall1 ,
            "Fmeasure_Rouge1" : Fmeasure1 ,
            "Precision_Rouge2" : Precision2,
            "Recall_Rouge2" : Recall2 ,
            "Fmeasure_Rouge2" : Fmeasure2 ,
            "Precision_RougeL" : PrecisionL,
            "Recall_RougeL": RecallL ,
            "Fmeasure_RougeL" : FmeasureL ,
            "Precision_RougeLsum" : PrecisionLs,
            "RecallLs_RougeLsum" : RecallLs ,
            "Fmeasure_RougeLsum" : FmeasureLs,
            "Blue" : Blue ,
            "Sac_Blue" : Sac_Blue ,
            "Meteor" : Meteor, 
            "Sari" : Sari , 
            "Bert_pre" : Bert_pre , 
            "Bert_recall" :  Bert_recall , 
            "Bert_f1" : Bert_f1 
            
                }

            
            print()
            print()
            print()
            print()
            
            final_path = create_folder(name)
            df_ = pd.DataFrame(df_)

            df_.to_csv(final_path + "/" + f_, index = False)
        
    except Exception as e:

        print(e)






base_folder_infer = "/workspace/data/Momojit/ensemble-llm/pre-train-model/infer-res/"
base_folder_met = "/workspace/data/Momojit/ensemble-llm/pre-train-model/infer-res-met/"


paths = ["t5-small-pred-Con_Model_All-covidqa.csv",
"t5-small-pred-Con_Model_All-delucionqa.csv",
"t5-small-pred-Con_Model_All-finqa.csv" , 
"t5-small-pred-Con_Model_All-hagrid.csv",
"t5-small-pred-Con_Model_All-hotpotqa.csv"]



def rouge_score(pred,truth):
    
    print()
    print("rouge_score")
    print()


    FmeasureL = []


    for i,j in zip(pred,truth):

        res = rouge.compute(predictions=[i], references=[j])


        FmeasureL.append(res["rougeL"].mid.fmeasure)


    return np.mean(FmeasureL)


def bleu_score(pred,truth):
    
    print()
    print("bleu_score")
    print()


    Blue = []

    for i,j in zip(pred,truth):


        i = [i.split(" ")]
        j = [[j.split(" ")]]

        res = bleu.compute(predictions=i, references=j)['bleu']

        Blue.append(res)

    return np.mean(Blue)


def sacrebleu_score(pred,truth):

    print()
    print("sacrebleu_score")
    print()
    

    Blue = []

    for i,j in zip(pred,truth):

        i = [i]
        j = [[j]]

        res = sacrebleu.compute(predictions=i, references=j)['score']
        Blue.append(res)

    return np.mean(Blue)


def meteor_score(pred,truth):
    
    
    print()
    print("meteor_score")
    print()

    Meteor = []

    for i,j in zip(pred,truth):

        i = [i]
        j = [j]

        res = meteor.compute(predictions=i, references=j)['meteor']
        Meteor.append(res)

    return np.mean(Meteor)


def sari_score(pred,truth):
    
    print()
    print("sari_score")
    print()

    Sari = []

    for i,j in zip(pred,truth):

        i = [i]
        j = [j]

        res = sari.compute(sources = i , predictions=j, references=[j])['sari']
        Sari.append(res)

    return np.mean(Sari)


def bert_score(pred,truth):

    print()
    print("bert_score")
    print()
    
    Bert_f1 = []

    for i,j in zip(pred,truth):

        i = [i]
        j = [j]

        res = bertscore.compute(predictions = i , references=j, model_type = "distilbert-base-uncased")


        Bert_f1.append(res['f1'][0])

    return np.mean(Bert_f1)


for f in paths:
    
    print()
    print("*"*100)
    print("*"*100)
    print()
    print(f)
    print()
    print("*"*100)
    print("*"*100)
    print()

    d = {}
    
    d_path = os.path.join(base_folder_infer,f)
    
    df = pd.read_csv(d_path)
    
    mis2 = list(df["mis2"])
    llama3 = list(df["llama3"])
    mis3 = list(df["mis3"])
    pred = list(df["Pred"])
    gt = list(df["GT"])

    rl = rouge_score(mis2,gt)

    Blue = bleu_score(mis2,gt)

    Sac_Blue = sacrebleu_score(mis2,gt)

    Meteor = meteor_score(mis2,gt)

    Sari = sari_score(mis2,gt)

    Bert_f1 = bert_score(mis2,gt)

    print()
    print()
    print("*"*100)
    
    print("Mistral-2")
    print(f"Rouge: {rl}")
    print(f"Blue: {Blue}")
    print(f"Sac_Blue: {Sac_Blue}")
    print(f"Meteor: {Meteor}")
    print(f"Sari: {Sari}")
    print(f"Bert_f1: {Bert_f1}")
    print()
    print("*"*100)
    print()
    print()
    

    d1 = {
        "Rouge-l" : rl,
        "Blue" : Blue , 
        "Sac_Blue" : Sac_Blue, 
        "Sari" : Sari, 
        "Bert_f1" : Bert_f1
    }

    d["mistral-2"] = d1 
    
    rl = rouge_score(mis3,gt)

    Blue = bleu_score(mis3,gt)

    Sac_Blue = sacrebleu_score(mis3,gt)

    Meteor = meteor_score(mis3,gt)

    Sari = sari_score(mis3,gt)

    Bert_f1 = bert_score(mis3,gt)

    print()
    print()
    print("*"*100)
    
    print("Mistral-3")
    print(f"Rouge: {rl}")
    print(f"Blue: {Blue}")
    print(f"Sac_Blue: {Sac_Blue}")
    print(f"Meteor: {Meteor}")
    print(f"Sari: {Sari}")
    print(f"Bert_f1: {Bert_f1}")
    print()
    print("*"*100)
    print()
    print()
    

    d1 = {
        "Rouge-l" : rl,
        "Blue" : Blue , 
        "Sac_Blue" : Sac_Blue, 
        "Sari" : Sari, 
        "Bert_f1" : Bert_f1
    }


    d["mistral-3"] = d1     

    rl = rouge_score(llama3,gt)    

    Blue = bleu_score(llama3,gt)

    Sac_Blue = sacrebleu_score(llama3,gt)

    Meteor = meteor_score(llama3,gt)

    Sari = sari_score(llama3,gt)

    Bert_f1 = bert_score(llama3,gt)

    print()
    print()
    print("*"*100)
    
    print("Llama-3")
    print(f"Rouge: {rl}")
    print(f"Blue: {Blue}")
    print(f"Sac_Blue: {Sac_Blue}")
    print(f"Meteor: {Meteor}")
    print(f"Sari: {Sari}")
    print(f"Bert_f1: {Bert_f1}")
    print()
    print("*"*100)
    print()
    print()
    
    d1 = {
        "Rouge-l" : rl,
        "Blue" : Blue , 
        "Sac_Blue" : Sac_Blue, 
        "Sari" : Sari, 
        "Bert_f1" : Bert_f1
    }

    d["mistral-3"] = d1     
    
    rl = rouge_score(pred,gt)    

    Blue = bleu_score(pred,gt)

    Sac_Blue = sacrebleu_score(pred,gt)

    Meteor = meteor_score(pred,gt)

    Sari = sari_score(pred,gt)

    Bert_f1 = bert_score(pred,gt)

    print()
    print()
    print("*"*100)
    
    print("propossed")
    print(f"Rouge: {rl}")
    print(f"Blue: {Blue}")
    print(f"Sac_Blue: {Sac_Blue}")
    print(f"Meteor: {Meteor}")
    print(f"Sari: {Sari}")
    print(f"Bert_f1: {Bert_f1}")
    print()
    print("*"*100)
    print()
    print()

    d1 = {
        "Rouge-l" : rl,
        "Blue" : Blue , 
        "Sac_Blue" : Sac_Blue, 
        "Sari" : Sari, 
        "Bert_f1" : Bert_f1
    }


    d["proposed"] = d1     


    print()
    print()
    print(d)
    print()
    print()
    
    
    with open(os.path.join(base_folder_met,f[:-4] + ".json"), "w") as f_:
        json.dump(d, f_, indent=4)  
    



    
