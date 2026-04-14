from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="crumb/nano-mistral",
    task='text-generation'
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

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'black hole'})

print(result)