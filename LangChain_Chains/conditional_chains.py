from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel, Field

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

# 🔹 Hugging Face FREE LLM
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    temperature=0.2,
    max_new_tokens=200
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

# 🔹 Pydantic schema
class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Give the sentiment of the feedback"
    )

parser2 = PydanticOutputParser(pydantic_object=Feedback)

# 🔹 Classification prompt
prompt1 = PromptTemplate(
    template=(
        "You are a strict JSON generator.\n"
        "Classify the sentiment of the following feedback as either "
        "`positive` or `negative`.\n\n"
        "Rules:\n"
        "- Output ONLY valid JSON\n"
        "- Do NOT add any text before or after JSON\n"
        "- Do NOT explain\n\n"
        "Feedback:\n{feedback}\n\n"
        "{format_instruction}"
    ),
    input_variables=["feedback"],
    partial_variables={
        "format_instruction": parser2.get_format_instructions()
    }
)


classifier_chain = prompt1 | model | RunnableLambda(safe_parse)

# 🔹 Response prompts
prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback:\n{feedback}",
    input_variables=["feedback"]
)

prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback:\n{feedback}",
    input_variables=["feedback"]
)

# 🔹 Branching logic
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | model | parser),
    (lambda x: x.sentiment == "negative", prompt3 | model | parser),
    RunnableLambda(lambda _: "Could not determine sentiment")
)

# 🔹 Final chain
chain = classifier_chain | branch_chain

# 🔹 Test
print(chain.invoke({"feedback": "This is a beautiful phone"}))

import re
from langchain_core.exceptions import OutputParserException

def safe_parse(output: str):
    try:
        return parser2.parse(output)
    except Exception:
        # Extract JSON manually
        match = re.search(r"\{.*\}", output, re.DOTALL)
        if not match:
            raise OutputParserException("No JSON found", llm_output=output)
        return parser2.parse(match.group())


