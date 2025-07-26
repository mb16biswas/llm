import subprocess

# List of commands
commands = [
    "pip install --upgrade pip",
    "pip install torch==2.0.1",
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
nltk.download('punkt_tab')

meteor = evaluate.load("meteor")
sari = load("sari")
rouge = load_metric("rouge",trust_remote_code=True)
bleu = load_metric("bleu",trust_remote_code=True)
sacrebleu = load_metric("sacrebleu",trust_remote_code=True)
bertscore = load("bertscore")



paths = [

]


files = os.listdir(paths[1])

df = pd.read_csv(os.path.join(paths[1],files[1]))

res = list(df["Response"])
ans = list(df["Answer"])

# res,ans



def rouge_score(pred,truth):


    FmeasureL = []
    FmeasureLs = []

    for i,j in zip(pred,truth):

        res = rouge.compute(predictions=[i], references=[j])


        FmeasureL.append(res["rougeL"].mid.fmeasure)

        FmeasureLs.append(res["rougeLsum"].mid.fmeasure)

    return (FmeasureL,FmeasureLs)


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

    Bert_f1 = []

    for i,j in zip(pred,truth):

        i = [i]
        j = [j]

        res = bertscore.compute(predictions = i , references=j, model_type = "distilbert-base-uncased")

        Bert_pre.append(res['precision'][0])
        Bert_recall.append(res['recall'][0])
        Bert_f1.append(res['f1'][0])

    return Bert_f1

answers = ans 
truths = res



FmeasureL,FmeasureLs = rouge_score(answers,truths)




Blue = bleu_score(answers,truths)  

Sac_Blue = sacrebleu_score(answers,truths)

Meteor = meteor_score(answers,truths)

Sari = sari_score(answers,truths)

Bert_f1 = bert_score(answers,truths)


print()
print()
print("FmeasureL")
print(FmeasureL)
print()
print()
print("*"*100)
print()
print()
print("Blue")
print(Blue)
print()
print()
print("*"*100)
print()
print()
print("Sac_Blue")
print(Sac_Blue)
print()
print()
print("*"*100)
print()
print()
print("Meteor")
print(Meteor)
print()
print()
print("*"*100)
print("*"*100)
print()
print()
print("Sari")
print(Sari)
print()
print()
print("*"*100)
print("*"*100)
print()
print()
print("Bert_f1 ")
print(Bert_f1 )
print()
print()
print("*"*100)
