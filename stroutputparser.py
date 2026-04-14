from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",
    task='text-generation',
    max_new_tokens=512
)

model = ChatHuggingFace(llm=llm)

# 1st Prompt : Detailed Report
template1 = PromptTemplate(
    template = 'write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd Prompt : Summary
template2 = PromptTemplate(
    template = 'write a 5 line summary on the following text. \n {text}',
    input_variables=['text']
)

prompt1 = template1.invoke({'topic': 'black hole'})

result = model.invoke(prompt1)

print("=== RESULT 1 ===")
print(result.content)  # ← add this
print("================")

prompt2 = template2.invoke({'text':result.content})

result1 = model.invoke(prompt2)

print(result1.content)