"""Chat and RAG functionality."""
import os
import logging
import markdown
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    openai_api_key=os.environ["OPENAI_API_KEY"]
)

# HR Analysis prompt template
HR_PROMPT = PromptTemplate(
    template="""You are an expert HR analyst. Analyze the following question using the provided context.

Context: {context}

Question: {question}

Format your response in markdown with:
## Summary
[Executive summary]

## Key Insights
- [Key finding 1]
- [Key finding 2]

## Recommendations
1. [Recommendation 1]
2. [Recommendation 2]

## Skills Assessment
[Skills alignment if relevant]

Focus on HR factors like skills, experience, qualifications, and cultural fit.
""",
    input_variables=["context", "question"]
)


def process_chat_question(question: str, vectorstore) -> Dict[str, Any]:
    """Process chat question using RAG chain."""
    if not vectorstore:
        raise ValueError("Vector store not available")
    
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # Create RAG chain
    chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough()
        }
        | HR_PROMPT
        | llm
    )
    
    # Execute chain
    result = chain.invoke(question)
    markdown_answer = result.content if hasattr(result, 'content') else str(result)
    
    # Convert to HTML
    html_answer = markdown.markdown(markdown_answer)
    
    # Get source documents
    source_docs = retriever.get_relevant_documents(question)
    sources = [
        {
            "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            "metadata": doc.metadata
        }
        for doc in source_docs
    ]
    
    return {"answer": html_answer, "sources": sources}


def _format_docs(docs) -> str:
    """Format retrieved documents for context."""
    return "\n\n".join(doc.page_content for doc in docs)
