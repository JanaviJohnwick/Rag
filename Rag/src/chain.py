"""
QA Chain Module.
Builds a LangChain RetrievalQA chain with a strict prompt template
that forces the model to answer ONLY from retrieved context.
"""

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# ── Strict Prompt Template ──────────────────────────────────────────────────
# Forces the LLM to answer strictly from the provided context,
# eliminating hallucinations by design.
PROMPT_TEMPLATE = """You are a precise and helpful assistant. Use ONLY the 
following context extracted from the user's documents to answer the question. 
If the answer is not contained within the context, say exactly: 
"I don't have enough information in the provided documents to answer this question."

Do NOT make up or infer any information beyond what is explicitly stated in the context.

Context:
{context}

Question: {question}

Answer:"""


def build_qa_chain(llm, retriever) -> RetrievalQA:
    """
    Build a RetrievalQA chain with strict grounding.

    Args:
        llm: The language model instance.
        retriever: The document retriever instance.

    Returns:
        A configured RetrievalQA chain.
    """
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    return qa_chain
