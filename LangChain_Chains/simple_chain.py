from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from transformers import pipeline

# ===== Step 1: Create a Prompt Template =====
prompt_template = PromptTemplate(
    input_variables=["topic"],
    template="Write a short and informative paragraph about {topic}. Keep it concise and clear."
)

# ===== Step 2: Initialize Hugging Face LLM =====
# Using a smaller model for faster inference
hf_pipeline = pipeline(
    "text-generation",
    model="distilgpt2",  # Smaller, faster model
    max_new_tokens=100,
    truncation=True
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

# ===== Step 3: Create a Simple Chain =====
chain = LLMChain(llm=llm, prompt=prompt_template)

# ===== Step 4: Run the Chain and Display Response =====
def run_chain(topic):
    print(f"\n{'='*60}")
    print(f"Topic: {topic}")
    print(f"{'='*60}")
    
    try:
        response = chain.run(topic=topic)
        print(f"\nResponse:\n{response}")
        print(f"{'='*60}\n")
        return response
    except Exception as e:
        print(f"Error: {e}")
        return None

# ===== Main Execution =====
if __name__ == "__main__":
    # Example topics
    topics = [
        "Artificial Intelligence",
        "Climate Change",
        "Machine Learning"
    ]
    
    # Run chain for each topic
    for topic in topics:
        run_chain(topic)
