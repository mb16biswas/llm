from datasets import load_dataset
import ragas
#pip install ragas==0.1.7 , any versions above this gives error.
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from langchain_community.chat_models import ChatOllama
from ragas import evaluate
from langchain_community.embeddings import OllamaEmbeddings


print("ragas version: " ,ragas.__version__)


# loading the V2 dataset
amnesty_qa = load_dataset("explodinggradients/amnesty_qa", "english_v2",trust_remote_code=True)
amnesty_subset = amnesty_qa["eval"].select(range(2))




df = amnesty_subset.to_pandas()

print(df.head(2))


langchain_llm = ChatOllama(model="moondream")
langchain_embeddings = OllamaEmbeddings(model="moondream")

print("langchain_llm: ", langchain_llm)
print("langchain_embeddings: ", langchain_embeddings)

result = evaluate(amnesty_subset,
                  metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall], llm=langchain_llm,embeddings=langchain_embeddings)



###########################################################################################################################################################################################
###########################################################################################################################################################################################
###########################################################################################################################################################################################
###########################################################################################################################################################################################
###########################################################################################################################################################################################
###########################################################################################################################################################################################
###########################################################################################################################################################################################
###########################################################################################################################################################################################
###########################################################################################################################################################################################

#pip install ragas==0.1.7 
from datasets import Dataset 
import os
from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness

data_samples = {
    'question': [
        'When was the first super bowl?', 
        'Who won the most super bowls?'
    ],
    'answer': [
        'The first superbowl was held on Jan 15, 1967', 
        'The most super bowls have been won by The New England Patriots'
    ],
    'contexts': [
        [
            'The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles,'
        ], 
        [
            'The Green Bay Packers...Green Bay, Wisconsin.',
            'The Packers compete...Football Conference'
        ]
    ],
    'ground_truth': [
        'The first superbowl was held on January 15, 1967', 
        'The New England Patriots have won the Super Bowl a record six times'
    ]
}

dataset = Dataset.from_dict(data_samples)

score = evaluate(dataset, metrics=[faithfulness, answer_correctness])
df = score.to_pandas()
df.to_csv('score.csv', index=False)
