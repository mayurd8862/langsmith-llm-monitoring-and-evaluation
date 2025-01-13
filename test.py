from langsmith import Client, traceable

client = Client()

# Define dataset: these are your test cases
dataset = client.create_dataset(
    "Sample Dataset",
    description="A sample dataset in LangSmith.",
)

client.create_examples(
    inputs=[
        {"postfix": "to LangSmith"},
        {"postfix": "to Evaluations in LangSmith"},
    ],
    outputs=[
        {"response": "Welcome to LangSmith"},
        {"response": "Welcome to Evaluations in LangSmith"},
    ],
    dataset_id=dataset.id,
)

# Define an interface to your application (tracing optional)
@traceable
def dummy_app(inputs: dict) -> dict:
    return {"response": "Welcome " + inputs["postfix"]}

# Define your evaluator(s)
def exact_match(outputs: dict, reference_outputs: dict) -> bool:
    return outputs["response"] == reference_outputs["response"]

# Run the evaluation
experiment_results = client.evaluate(
    dummy_app, # Your AI system goes here
    data=dataset, # The data to predict and grade over
    evaluators=[exact_match], # The evaluators to score the results
    experiment_prefix="sample-experiment", # The name of the experiment
    metadata={"version": "1.0.0", "revision_id": "beta"}, # Metadata about the experiment
    max_concurrency=4,  # Add concurrency.
)

# Analyze the results via the UI or programmatically
# If you have 'pandas' installed you can view the results as a
# pandas DataFrame by uncommenting below:

# experiment_results.to_pandas()