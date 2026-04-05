from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

# Initialize Hugging Face LLM
hf_pipeline = pipeline(
    "text-generation",
    model="distilgpt2",
    max_new_tokens=150,
    truncation=True
)

model = HuggingFacePipeline(pipeline=hf_pipeline)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic': 'Unemployment in India'})

print(result)

# Uncomment below to visualize the chain (requires grandalf)
chain.get_graph().print_ascii()