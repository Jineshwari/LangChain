# from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# llm=HuggingFacePipeline.from_model_id(
#     model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
#     task='text-generation',
#     pipeline_kwargs=dict(temperature=0.7, max_new_tokens=100),
# )
# model=ChatHuggingFace(llm=llm)
# result=model.invoke("What is the capital of India?")
# print(result.content)



print("STARTING SCRIPT")

from langchain_huggingface import HuggingFacePipeline
print("IMPORT DONE")

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(temperature=0.7, max_new_tokens=100),
)
print("MODEL LOADED, INVOKING...")

prompt = "<s>### Instruction:\nAnswer briefly: What is the capital of India?\n\n### Response:\n"
result = llm.invoke(prompt)
print(result)