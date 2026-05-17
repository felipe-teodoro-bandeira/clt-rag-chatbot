from __future__ import annotations

import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from src.retrieval.relevance import evaluate_chunks
from src.retrieval.retriever import get_retriever

load_dotenv()

MAX_ATTEMPTS = 3
RELEVANCE_THRESHOLD = 0.45  # combined score (0–1) abaixo do qual reformula a query

_CONDENSE_PROMPT = ChatPromptTemplate.from_template(
    """Dado o histórico da conversa e uma pergunta de acompanhamento, reformule a pergunta
para ser independente e autocontida, preservando o contexto necessário. Responda apenas com a
pergunta reformulada, sem explicações adicionais.

Histórico da conversa:
{chat_history}

Pergunta de acompanhamento: {question}

Pergunta reformulada:"""
)

_QA_PROMPT = ChatPromptTemplate.from_template(
    """Você é um assistente jurídico especialista na Consolidação das Leis do Trabalho (CLT) brasileira.

Diretrizes:
- Responda com base exclusivamente nos trechos da CLT fornecidos abaixo.
- Cite sempre o número do artigo (ex: "Conforme o Art. 134 da CLT, ...") quando disponível.
- Use linguagem clara e acessível; explique termos técnicos sempre que necessário.
- Se os trechos não contiverem a informação solicitada, informe claramente: \
"Não encontrei essa informação nos trechos da CLT disponíveis."
- Nunca invente, presuma ou extrapole informações além do que está nos trechos.
- Quando a resposta envolver prazos, valores ou percentuais, destaque-os explicitamente.

Trechos da CLT (ordenados por relevância):
{context}

Pergunta: {question}

Resposta:"""
)

_REFORMULATE_PROMPT = ChatPromptTemplate.from_template(
    """Você está refinando uma busca em um banco de dados da CLT brasileira.

Pergunta original do usuário: {original_question}
Query de busca atual: {current_query}
Artigos recuperados com baixa relevância: {low_quality_articles}

Reformule a query de busca para encontrar trechos mais relevantes.
Use termos jurídicos específicos da CLT e seja mais direto sobre o tema legal buscado.
Responda APENAS com a nova query reformulada, sem explicações adicionais.

Nova query:"""
)


class AgentState(TypedDict):
    original_question: str
    current_query: str
    chat_history: list[tuple[str, str]]
    retrieved_docs: list[Document]
    chunk_scores: list[dict]
    attempts: int
    answer: str


_llm: ChatGoogleGenerativeAI | None = None
_retriever = None


def _get_components():
    global _llm, _retriever
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.2,
        )
    if _retriever is None:
        _retriever = get_retriever(k=8)
    return _llm, _retriever


# --- Nós do grafo ---

def _node_retrieve(state: AgentState) -> dict:
    _, retriever = _get_components()
    docs = retriever.invoke(state["current_query"])
    return {"retrieved_docs": docs}


def _node_evaluate(state: AgentState) -> dict:
    scores = evaluate_chunks(state["current_query"], state["retrieved_docs"])
    return {"chunk_scores": scores}


def _node_reformulate(state: AgentState) -> dict:
    llm, _ = _get_components()
    low = [s for s in state["chunk_scores"] if s["combined"] < RELEVANCE_THRESHOLD]
    articles = ", ".join(s["doc"].metadata.get("artigo", "?") for s in low[:3])
    new_query = (_REFORMULATE_PROMPT | llm | StrOutputParser()).invoke({
        "original_question": state["original_question"],
        "current_query": state["current_query"],
        "low_quality_articles": articles or "nenhum artigo relevante encontrado",
    })
    return {"current_query": new_query.strip(), "attempts": state["attempts"] + 1}


def _node_generate(state: AgentState) -> dict:
    llm, _ = _get_components()
    good = [s for s in state["chunk_scores"] if s["combined"] >= RELEVANCE_THRESHOLD]
    docs_to_use = good if good else state["chunk_scores"][:4]
    context = "\n\n".join(
        f"[{s['doc'].metadata.get('artigo', '')} | relevância: {s['combined']}]\n"
        f"{s['doc'].page_content}"
        for s in docs_to_use
    )
    answer = (_QA_PROMPT | llm | StrOutputParser()).invoke({
        "context": context,
        "question": state["original_question"],
    })
    return {"answer": answer}


# --- Condição de roteamento ---

def _should_retry(state: AgentState) -> str:
    if not state["chunk_scores"]:
        return "generate"
    avg = sum(s["combined"] for s in state["chunk_scores"]) / len(state["chunk_scores"])
    if avg < RELEVANCE_THRESHOLD and state["attempts"] < MAX_ATTEMPTS:
        return "reformulate"
    return "generate"


# --- Construção do grafo ---

def _build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", _node_retrieve)
    graph.add_node("evaluate", _node_evaluate)
    graph.add_node("reformulate", _node_reformulate)
    graph.add_node("generate", _node_generate)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "evaluate")
    graph.add_conditional_edges(
        "evaluate",
        _should_retry,
        {"reformulate": "reformulate", "generate": "generate"},
    )
    graph.add_edge("reformulate", "retrieve")
    graph.add_edge("generate", END)

    return graph.compile()


_graph = None


def get_agent_answer(question: str, chat_history: list[tuple[str, str]]) -> str:
    """Executa o pipeline RAG agêntico com avaliação de relevância e retry de query.

    Fluxo:
        1. Recupera k=8 chunks do vectorstore
        2. Avalia cada chunk com cosine similarity + LLM-as-judge
        3. Se relevância média < threshold e tentativas < MAX_ATTEMPTS,
           reformula a query e repete
        4. Gera resposta usando apenas chunks de alta relevância
    """
    global _graph
    if _graph is None:
        _graph = _build_graph()

    llm, _ = _get_components()
    query = question
    if chat_history:
        formatted = "\n".join(
            f"Humano: {h}\nAssistente: {a}" for h, a in chat_history
        )
        query = (_CONDENSE_PROMPT | llm | StrOutputParser()).invoke({
            "chat_history": formatted,
            "question": question,
        })

    result = _graph.invoke({
        "original_question": question,
        "current_query": query,
        "chat_history": chat_history,
        "retrieved_docs": [],
        "chunk_scores": [],
        "attempts": 0,
        "answer": "",
    })
    return result["answer"]


def get_agent_answer_traced(
    question: str,
    chat_history: list[tuple[str, str]],
) -> tuple[str, list[str], list[str]]:
    """Returns (answer, used_articles, all_retrieved_articles).

    used_articles: artigos dos chunks com combined >= RELEVANCE_THRESHOLD,
                   ou top 4 como fallback se nenhum passar o threshold.
    all_retrieved_articles: todos os artigos retornados pelo retriever na última tentativa.
    """
    global _graph
    if _graph is None:
        _graph = _build_graph()

    llm, _ = _get_components()
    query = question
    if chat_history:
        formatted = "\n".join(
            f"Humano: {h}\nAssistente: {a}" for h, a in chat_history
        )
        query = (_CONDENSE_PROMPT | llm | StrOutputParser()).invoke({
            "chat_history": formatted,
            "question": question,
        })

    result = _graph.invoke({
        "original_question": question,
        "current_query": query,
        "chat_history": chat_history,
        "retrieved_docs": [],
        "chunk_scores": [],
        "attempts": 0,
        "answer": "",
    })

    all_retrieved = [s["doc"].metadata.get("artigo", "—") for s in result["chunk_scores"]]
    good = [s for s in result["chunk_scores"] if s["combined"] >= RELEVANCE_THRESHOLD]
    docs_to_use = good if good else result["chunk_scores"][:4]
    used = [s["doc"].metadata.get("artigo", "—") for s in docs_to_use]

    return result["answer"], used, all_retrieved
