import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langsmith import wrappers, Client
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
# Load environment variables
load_dotenv()

LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = "evaluation_demo"




# llm = ChatGroq(model="mixtral-8x7b-32768")

# res = llm.invoke("who is the pm of india")
# print(res)



client = Client()
# openai_client = wrappers.wrap_openai(OpenAI())

# For other dataset creation methods, see:
# https://docs.smith.langchain.com/evaluation/how_to_guides/manage_datasets_programmatically
# https://docs.smith.langchain.com/evaluation/how_to_guides/manage_datasets_in_application

# Create inputs and reference outputs
examples = [
  (
      "Which country is Mount Kilimanjaro located in?",
      "Mount Kilimanjaro is located in Tanzania.",
  ),
  (
      "What is Earth's lowest point?",
      "Earth's lowest point is The Dead Sea.",
  ),
]

inputs = [{"question": input_prompt} for input_prompt, _ in examples]
outputs = [{"answer": output_answer} for _, output_answer in examples]

# Programmatically create a dataset in LangSmith
dataset = client.create_dataset(
  dataset_name="Sample dataset", description="A sample dataset in LangSmith."
)

# Add examples to the dataset
client.create_examples(inputs=inputs, outputs=outputs, dataset_id=dataset.id)



# Define the application logic you want to evaluate inside a target function
# The SDK will automatically send the inputs from the dataset to your target function
def target(inputs):
  
    llm = ChatGroq(model="mixtral-8x7b-32768")
    res = llm.invoke(inputs)

    return res.content


# Define output schema for the LLM judge
class Grade(BaseModel):
    score: bool = Field(
        description="Boolean that indicates whether the response is accurate relative to the reference answer"
    )

def accuracy(outputs: dict, reference_outputs: dict) -> bool:
    # Set up a parser + inject instructions into the prompt template.
    parser = JsonOutputParser(pydantic_object=Grade)
    model = ChatGroq(model="mixtral-8x7b-32768")
    prompt = PromptTemplate(
        template=f"""Evaluate Student Answer against Ground Truth for conceptual similarity and classify true or false: 
        - False: No conceptual match and similarity
        - True: Most or full conceptual match and similarity
        - Key criteria: Concept should match, not exact wording.

        Ground Truth answer: {reference_outputs["answer"]}
        Student's Answer: {outputs["response"]}
        """,
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | model | parser

    return chain.invoke()


# After running the evaluation, a link will be provided to view the results in langsmith
# experiment_results = client.evaluate_run(
#   target,
#   data="Sample dataset",
#   evaluators=[
#       accuracy,
#       # can add multiple evaluators here
#   ],
#   experiment_prefix="first-eval-in-langsmith",
#   max_concurrency=2,
# )


# print(experiment_results)