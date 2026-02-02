"""
TechGear Electronics - LangGraph Workflow Agent
This module implements the customer support workflow using LangGraph.
Nodes: Classifier -> RAG Responder / Escalation
"""

import os
from typing import TypedDict, Literal
from enum import Enum
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rag_chain import get_rag_chain

# Load environment variables
load_dotenv()


class QueryCategory(str, Enum):
    """Categories for customer queries."""
    PRODUCTS = "PRODUCTS"
    RETURNS = "RETURNS"
    GENERAL = "GENERAL"
    ESCALATE = "ESCALATE"


class AgentState(TypedDict):
    """State object passed through the LangGraph workflow."""
    input: str                    # Original user query
    category: str                 # Classified category
    response: str                 # Final response to user
    needs_escalation: bool        # Flag for human escalation


# Classification prompt
CLASSIFIER_PROMPT = """You are a query classifier for TechGear Electronics customer support.
Analyze the customer query and classify it into ONE of these categories:

- PRODUCTS: Questions about product features, specifications, prices, availability, compatibility, troubleshooting, or technical support
- RETURNS: Questions about returns, refunds, exchanges, order cancellations, or warranty claims
- GENERAL: General inquiries about store hours, locations, payment methods, shipping, or company information
- ESCALATE: Complaints, issues requiring human intervention, legal matters, or requests to speak with a manager

Customer Query: {query}

Respond with ONLY the category name (PRODUCTS, RETURNS, GENERAL, or ESCALATE), nothing else."""


# Escalation message template
ESCALATION_MESSAGE = """I understand your concern, and I want to make sure you receive the best possible assistance.

I'm connecting you with one of our customer support specialists who can help you further. A team member will be with you shortly.

**What you can expect:**
- A support specialist will contact you within 24 hours
- For urgent matters, please call our toll-free number: 1800-102-TECH (1800-102-8324)
- You can also email us at support@techgear.com

Thank you for your patience. Is there anything else I can help you with in the meantime?"""


def get_classifier_llm():
    """Get the LLM for classification (lightweight, fast responses)."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
        max_tokens=20,
    )


def classifier_node(state: AgentState) -> AgentState:
    """
    Node 1: Classify the user query into a category.
    Categories: PRODUCTS, RETURNS, GENERAL, ESCALATE
    """
    query = state["input"]
    
    llm = get_classifier_llm()
    prompt = ChatPromptTemplate.from_template(CLASSIFIER_PROMPT)
    chain = prompt | llm | StrOutputParser()
    
    # Get classification
    result = chain.invoke({"query": query}).strip().upper()
    
    # Validate and normalize category
    valid_categories = {cat.value for cat in QueryCategory}
    if result not in valid_categories:
        # Default to GENERAL if classification is unclear
        result = QueryCategory.GENERAL.value
    
    return {
        **state,
        "category": result,
        "needs_escalation": result == QueryCategory.ESCALATE.value
    }


def rag_responder_node(state: AgentState) -> AgentState:
    """
    Node 2: Use RAG to generate a response based on the knowledge base.
    Handles PRODUCTS, RETURNS (policy info), and GENERAL queries.
    """
    query = state["input"]
    
    # Get RAG chain and generate response
    rag = get_rag_chain()
    response = rag.invoke(query)
    
    return {
        **state,
        "response": response
    }


def escalation_node(state: AgentState) -> AgentState:
    """
    Node 3: Handle escalation for complex issues or complaints.
    Returns a message indicating human support will be provided.
    """
    return {
        **state,
        "response": ESCALATION_MESSAGE,
        "needs_escalation": True
    }


def route_after_classification(state: AgentState) -> Literal["rag_responder", "escalation"]:
    """
    Conditional routing based on classification result.
    Routes to RAG for most queries, escalation for complaints.
    """
    category = state.get("category", "")
    
    if category == QueryCategory.ESCALATE.value:
        return "escalation"
    else:
        # PRODUCTS, RETURNS, GENERAL all go through RAG
        return "rag_responder"


def build_workflow() -> StateGraph:
    """
    Build the LangGraph workflow for the customer support chatbot.
    
    Flow:
    START -> Classifier -> [PRODUCTS/RETURNS/GENERAL] -> RAG Responder -> END
                        -> [ESCALATE] -> Escalation Node -> END
    """
    # Create the state graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("classifier", classifier_node)
    workflow.add_node("rag_responder", rag_responder_node)
    workflow.add_node("escalation", escalation_node)
    
    # Set entry point
    workflow.set_entry_point("classifier")
    
    # Add conditional edges from classifier
    workflow.add_conditional_edges(
        "classifier",
        route_after_classification,
        {
            "rag_responder": "rag_responder",
            "escalation": "escalation"
        }
    )
    
    # Add edges to END
    workflow.add_edge("rag_responder", END)
    workflow.add_edge("escalation", END)
    
    return workflow


def compile_workflow():
    """Compile and return the workflow graph."""
    workflow = build_workflow()
    return workflow.compile()


# Singleton compiled graph
_compiled_graph = None


def get_compiled_graph():
    """Get the compiled LangGraph workflow (singleton)."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = compile_workflow()
    return _compiled_graph


def process_query(query: str) -> dict:
    """
    Process a customer query through the workflow.
    
    Args:
        query: The customer's question or message
        
    Returns:
        dict with keys: input, category, response, needs_escalation
    """
    graph = get_compiled_graph()
    
    # Initialize state
    initial_state: AgentState = {
        "input": query,
        "category": "",
        "response": "",
        "needs_escalation": False
    }
    
    # Run the workflow
    result = graph.invoke(initial_state)
    
    return result


# For testing
if __name__ == "__main__":
    print("Testing LangGraph Workflow...")
    print("=" * 60)
    
    test_queries = [
        "What is the price of SmartWatch Pro X?",                    # PRODUCTS
        "How do I return a defective product?",                       # RETURNS
        "What are your store hours?",                                 # GENERAL
        "This is unacceptable! I want to speak to a manager!",       # ESCALATE
        "My earbuds are not connecting to my phone",                 # PRODUCTS (troubleshooting)
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        try:
            result = process_query(query)
            print(f"Category: {result['category']}")
            print(f"Escalated: {result['needs_escalation']}")
            print(f"Response: {result['response'][:200]}...")
        except Exception as e:
            print(f"Error: {e}")
        print()
