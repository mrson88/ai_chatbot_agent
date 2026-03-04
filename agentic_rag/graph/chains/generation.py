from langchain_ollama import ChatOllama

# The LangChain Hub SDK no longer exports a `hub` object; instead we
# interact with `Client` which returns raw JSON manifests.  Rather than
# pulling prompts at runtime we can define the prompt directly so that
# the package dependency is short-circuited and imports succeed.
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# instantiate the LLM we'll use for generation
llm = ChatOllama(model="qwen3:1.7b", temperature=0)

# manually built prompt matching the definition previously stored at
# rlm/rag-prompt on LangChain Hub.  This avoids the need to pull from
# the (deprecated) langchainhub SDK.
prompt = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template(
        "You are an assistant for question-answering tasks. Use the following "
        "pieces of retrieved context to answer the question. If you don't "
        "know the answer, just say that you don't know. Use three sentences "
        "maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"
    )
])

# wire together the prompt, the LLM, and a simple string output parser
# so that invoking the chain returns the generated text only.
generation_chain = prompt | llm | StrOutputParser()
