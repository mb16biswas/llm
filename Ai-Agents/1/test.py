from dotenv import load_dotenv
import os
import pandas as pd
from llama_index.query_engine import PandasQueryEngine
from prompts import new_prompt, instruction_str, context
from note_engine import note_engine
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.agent import ReActAgent
from llama_index.llms import OpenAI
from pdf import canada_engine
# from key import OPENAI_API_KEY

# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


load_dotenv()


population_path = os.path.join("data", "population.csv")
population_df = pd.read_csv(population_path)




population_query_engine = PandasQueryEngine(
    df=population_df,verbose=True, instruction_str=instruction_str
)



population_query_engine.update_prompts({"pandas_prompt": new_prompt})





s = population_query_engine.query(
    "what is the population of india?",
)

print()
print("*"*100)
print()
print()
print("what is the population of india?")
print(str(s))
print()
print("*"*100)
print()

s = population_query_engine.query(
    "Who is Walter White?",
)


print()
print("*"*100)
print()
print("Who is Walter White?")
print(str(s))
print()
print("*"*100)
print()



