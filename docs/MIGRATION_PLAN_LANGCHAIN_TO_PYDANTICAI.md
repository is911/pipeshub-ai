# Migration Plan: Langchain to PydanticAI

## Executive Summary

This document outlines a comprehensive migration strategy to replace Langchain with PydanticAI in the PipesHub AI codebase. The migration involves **20 files**, **72+ Langchain imports**, and **16 Langchain packages**.

### Why PydanticAI?

| Aspect | Langchain | PydanticAI |
|--------|-----------|------------|
| **Type Safety** | Limited | First-class with Pydantic |
| **Complexity** | High abstraction layers | Pythonic, minimal abstraction |
| **Validation** | External parsers needed | Built-in Pydantic validation |
| **Debugging** | Complex tracing | Logfire integration |
| **Dependencies** | 16+ packages | Single package |
| **Learning Curve** | Steep | FastAPI-like simplicity |

---

## Current State Analysis

### Langchain Dependencies (16 packages)

```
langchain==0.3.19
langgraph==0.3.34
langchain-anthropic==0.3.17
langchain-aws==0.2.18
langchain-cohere==0.4.5
langchain-community==0.3.18
langchain-experimental==0.3.4
langchain-fireworks==0.3.0
langchain-google-genai==2.1.8
langchain-google-vertexai==2.0.18
langchain-groq==0.3.6
langchain-huggingface==0.3.0
langchain-mistralai==0.2.11
langchain-ollama==0.3.0
langchain-openai==0.3.28
langchain-qdrant==0.2.0
langchain-together==0.3.0
langchain-voyageai==0.1.6
langchain-xai==0.2.4
```

### Files Requiring Migration

| File | Complexity | Components Used |
|------|------------|-----------------|
| `app/utils/aimodels.py` | 4/5 | 15+ LLM providers, 14+ embedding providers |
| `app/utils/streaming.py` | 5/5 | Async streaming, tool binding, message types |
| `app/modules/transformers/vectorstore.py` | 5/5 | QdrantVectorStore, embeddings, hybrid search |
| `app/modules/retrieval/retrieval_service.py` | 4/5 | Vector search, embeddings |
| `app/modules/agents/qna/graph.py` | 4/5 | Langgraph StateGraph |
| `app/modules/agents/qna/nodes.py` | 4/5 | Agent nodes, message types |
| `app/modules/indexing/run.py` | 4/5 | SemanticChunker |
| `app/utils/query_decompose.py` | 4/5 | LCEL chains, structured output |
| `app/modules/transformers/document_extraction.py` | 4/5 | PydanticOutputParser, multimodal |
| `app/utils/custom_embeddings.py` | 3/5 | Custom Voyage embeddings |
| `app/utils/indexing_helpers.py` | 3/5 | Output parsers |
| `app/modules/extraction/domain_extraction.py` | 3/5 | Classification parser |
| `app/utils/query_transform.py` | 2/5 | LCEL chains |
| `app/modules/agents/qna/tools.py` | 2/5 | BaseTool |
| `app/modules/agents/qna/tool_registry.py` | 2/5 | Tool wrapper |
| `app/utils/fetch_full_record.py` | 2/5 | Tool decorator |
| `app/modules/qna/agent_prompt.py` | 2/5 | Message types |
| `app/utils/llm.py` | 2/5 | Model wrapper |
| `app/modules/parsers/excel/prompt_template.py` | 1/5 | ChatPromptTemplate |
| `app/modules/parsers/csv/csv_parser.py` | 2/5 | LLM calls |

---

## PydanticAI Component Mapping

### 1. LLM/Chat Models

| Langchain | PydanticAI Equivalent |
|-----------|----------------------|
| `ChatAnthropic` | `Agent('anthropic:claude-sonnet-4-0')` |
| `ChatOpenAI` | `Agent('openai:gpt-4o')` |
| `AzureChatOpenAI` | `Agent('azure:gpt-4')` |
| `ChatBedrock` | `Agent('bedrock:...')` |
| `ChatGoogleGenerativeAI` | `Agent('gemini-1.5-pro')` |
| `ChatGroq` | `Agent('groq:llama-3.1-70b')` |
| `ChatMistralAI` | `Agent('mistral:mistral-large')` |
| `ChatOllama` | `Agent('ollama:llama3.2')` |
| `ChatCohere` | `Agent('cohere:command-r-plus')` |
| `ChatFireworks` | Custom model implementation |
| `ChatTogether` | Custom model implementation |
| `ChatXAI` | Custom model implementation |

### 2. Message Types

| Langchain | PydanticAI Equivalent |
|-----------|----------------------|
| `HumanMessage` | `ModelRequest` with `UserPromptPart` |
| `AIMessage` | `ModelResponse` with `TextPart` |
| `SystemMessage` | `instructions` parameter or `SystemPromptPart` |
| `ToolMessage` | `ToolReturnPart` |
| `BaseMessage` | `ModelMessage` |

### 3. Tools

| Langchain | PydanticAI Equivalent |
|-----------|----------------------|
| `BaseTool` class | `@agent.tool` decorator |
| `@tool` decorator | `@agent.tool` decorator |
| `args_schema` | Function type hints + docstring |
| `_run()` method | Function body |
| Tool binding | Automatic via decorator |

### 4. Output Parsing

| Langchain | PydanticAI Equivalent |
|-----------|----------------------|
| `PydanticOutputParser` | `output_type=MyModel` |
| `StrOutputParser` | `output_type=str` (default) |
| Manual JSON parsing | Automatic Pydantic validation |

### 5. Prompts & Chains

| Langchain | PydanticAI Equivalent |
|-----------|----------------------|
| `ChatPromptTemplate` | f-strings or `@agent.instructions` |
| LCEL `\|` chains | Standard Python functions |
| `RunnablePassthrough` | Function parameters |

### 6. Vector Store & Embeddings

| Langchain | PydanticAI Equivalent |
|-----------|----------------------|
| `QdrantVectorStore` | Direct `qdrant-client` |
| `FastEmbedSparse` | Direct `fastembed` library |
| `*Embeddings` classes | Direct API calls or custom functions |

### 7. Agent Orchestration

| Langchain | PydanticAI Equivalent |
|-----------|----------------------|
| Langgraph `StateGraph` | `pydantic-graph` or explicit Python flow |
| `add_node()` | `@dataclass` nodes with `run()` |
| `add_edge()` | Return type hints |
| `compile()` | `Graph.run()` |

---

## Migration Phases

### Phase 1: Foundation (Week 1-2)

**Goal**: Create core abstractions and utilities

#### 1.1 Create Custom Model Factory

Replace `aimodels.py` with a PydanticAI-compatible model factory.

**New file**: `app/utils/pydantic_models.py`

```python
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.models.mistral import MistralModel
from pydantic_ai.models.ollama import OllamaModel
from typing import Optional
import os

class ModelFactory:
    """Factory for creating PydanticAI models from configuration."""

    @staticmethod
    def create_model(config: dict) -> Model:
        provider = config.get("provider", "").lower()
        model_name = config.get("modelName", "")
        api_key = config.get("apiKey", "")

        match provider:
            case "anthropic":
                return AnthropicModel(model_name, api_key=api_key)
            case "openai":
                return OpenAIModel(model_name, api_key=api_key)
            case "azure_openai":
                return OpenAIModel(
                    model_name,
                    api_key=api_key,
                    base_url=config.get("endpoint"),
                )
            case "gemini" | "google":
                return GeminiModel(model_name, api_key=api_key)
            case "groq":
                return GroqModel(model_name, api_key=api_key)
            case "mistral":
                return MistralModel(model_name, api_key=api_key)
            case "ollama":
                return OllamaModel(model_name, base_url=config.get("baseUrl"))
            case "bedrock":
                # Use bedrock via custom implementation
                return create_bedrock_model(config)
            case _:
                raise ValueError(f"Unsupported provider: {provider}")
```

#### 1.2 Create Custom Embedding Service

Replace Langchain embedding classes with direct API implementations.

**New file**: `app/utils/embedding_service.py`

```python
from abc import ABC, abstractmethod
from typing import List
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

class EmbeddingService(ABC):
    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        pass

class OpenAIEmbeddings(EmbeddingService):
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.api_key = api_key
        self.model = model
        self.client = httpx.AsyncClient()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = await self.client.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"input": texts, "model": self.model}
        )
        data = response.json()
        return [item["embedding"] for item in data["data"]]

    async def embed_query(self, text: str) -> List[float]:
        embeddings = await self.embed_documents([text])
        return embeddings[0]

# Similar classes for: Anthropic (via Voyage), Cohere, Google, etc.
```

#### 1.3 Create Message Types

**New file**: `app/utils/messages.py`

```python
from dataclasses import dataclass
from typing import List, Optional, Any, Union
from pydantic import BaseModel

@dataclass
class Message:
    role: str  # "user", "assistant", "system", "tool"
    content: str
    tool_calls: Optional[List[dict]] = None
    tool_call_id: Optional[str] = None

def to_pydantic_messages(messages: List[Message]) -> List[dict]:
    """Convert internal messages to PydanticAI format."""
    result = []
    for msg in messages:
        if msg.role == "user":
            result.append({"role": "user", "content": msg.content})
        elif msg.role == "assistant":
            result.append({"role": "assistant", "content": msg.content})
        elif msg.role == "system":
            result.append({"role": "system", "content": msg.content})
    return result
```

---

### Phase 2: Vector Store Migration (Week 2-3)

**Goal**: Replace langchain-qdrant with direct qdrant-client

#### 2.1 Update vectorstore.py

```python
# Before (Langchain)
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode

vector_store = QdrantVectorStore.from_documents(
    documents,
    embedding=embeddings,
    collection_name=collection_name,
    sparse_embedding=FastEmbedSparse("Qdrant/bm25"),
    retrieval_mode=RetrievalMode.HYBRID,
)

# After (Direct Qdrant Client)
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding

class VectorStoreService:
    def __init__(self, url: str, api_key: str):
        self.client = QdrantClient(url=url, api_key=api_key)
        self.sparse_encoder = SparseTextEmbedding("Qdrant/bm25")

    async def add_documents(
        self,
        documents: List[Document],
        embeddings: List[List[float]],
        collection_name: str
    ):
        points = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            sparse_embedding = list(self.sparse_encoder.embed([doc.content]))[0]
            points.append(models.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": embedding,
                    "sparse": models.SparseVector(
                        indices=sparse_embedding.indices.tolist(),
                        values=sparse_embedding.values.tolist(),
                    )
                },
                payload=doc.metadata
            ))

        await self.client.upsert(collection_name, points)

    async def search(
        self,
        query_embedding: List[float],
        query_text: str,
        collection_name: str,
        limit: int = 10
    ) -> List[Document]:
        sparse_embedding = list(self.sparse_encoder.embed([query_text]))[0]

        results = await self.client.query_points(
            collection_name,
            prefetch=[
                models.Prefetch(query=query_embedding, using="dense", limit=limit),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_embedding.indices.tolist(),
                        values=sparse_embedding.values.tolist(),
                    ),
                    using="sparse",
                    limit=limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
        )
        return [Document(content=r.payload["content"], metadata=r.payload) for r in results.points]
```

#### 2.2 Update retrieval_service.py

```python
# Before
from langchain_qdrant import QdrantVectorStore

# After
from app.utils.vector_store_service import VectorStoreService
from app.utils.embedding_service import EmbeddingFactory

class RetrievalService:
    def __init__(self, config: dict):
        self.vector_store = VectorStoreService(
            url=config["qdrant_url"],
            api_key=config["qdrant_api_key"]
        )
        self.embeddings = EmbeddingFactory.create(config["embedding_config"])

    async def retrieve(self, query: str, collection_name: str, limit: int = 10):
        query_embedding = await self.embeddings.embed_query(query)
        return await self.vector_store.search(
            query_embedding=query_embedding,
            query_text=query,
            collection_name=collection_name,
            limit=limit
        )
```

---

### Phase 3: Agent Migration (Week 3-4)

**Goal**: Replace Langgraph with PydanticAI agents

#### 3.1 Create Main QnA Agent

**New file**: `app/modules/agents/qna/pydantic_agent.py`

```python
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field
from typing import List, Optional
from dataclasses import dataclass

# Dependencies injected into agent
@dataclass
class QnADependencies:
    org_id: str
    user_id: str
    retrieval_service: RetrievalService
    conversation_history: List[dict]
    available_tools: List[str]

# Structured output
class QnAResponse(BaseModel):
    answer: str = Field(description="The answer to the user's question")
    sources: List[str] = Field(default_factory=list, description="Source document IDs")
    confidence: float = Field(ge=0, le=1, description="Confidence score")
    follow_up_questions: List[str] = Field(default_factory=list)

# Create the agent
qna_agent = Agent(
    'anthropic:claude-sonnet-4-0',  # or dynamically configured
    deps_type=QnADependencies,
    output_type=QnAResponse,
    instructions="""You are a helpful assistant that answers questions
    based on retrieved documents. Always cite your sources.""",
)

@qna_agent.tool
async def retrieve_documents(
    ctx: RunContext[QnADependencies],
    query: str,
    limit: int = 10
) -> str:
    """Retrieve relevant documents from the knowledge base."""
    docs = await ctx.deps.retrieval_service.retrieve(
        query=query,
        collection_name=f"org_{ctx.deps.org_id}",
        limit=limit
    )
    return "\n\n".join([f"[{doc.id}]: {doc.content}" for doc in docs])

@qna_agent.tool
async def web_search(
    ctx: RunContext[QnADependencies],
    query: str,
    num_results: int = 5
) -> str:
    """Search the web for current information."""
    from duckduckgo_search import DDGS
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=num_results))
    return "\n".join([f"- {r['title']}: {r['body']}" for r in results])

# Dynamic instructions based on context
@qna_agent.instructions
async def dynamic_instructions(ctx: RunContext[QnADependencies]) -> str:
    base = """You are a helpful assistant for organization queries."""
    if ctx.deps.available_tools:
        base += f"\n\nAvailable tools: {', '.join(ctx.deps.available_tools)}"
    return base
```

#### 3.2 Streaming Implementation

```python
async def stream_qna_response(
    query: str,
    deps: QnADependencies,
    model_config: dict
) -> AsyncGenerator[dict, None]:
    """Stream QnA responses with tool execution."""

    # Create agent with dynamic model
    model = ModelFactory.create_model(model_config)
    agent = qna_agent.override(model=model)

    async with agent.run_stream(query, deps=deps) as result:
        async for event in result.stream_events():
            if event.kind == "text":
                yield {"type": "text", "content": event.data}
            elif event.kind == "tool_call":
                yield {"type": "tool_call", "tool": event.tool_name, "args": event.args}
            elif event.kind == "tool_result":
                yield {"type": "tool_result", "result": event.result}

        # Final structured output
        final = await result.get_output()
        yield {"type": "final", "output": final.model_dump()}
```

#### 3.3 Graph-Based Workflow (Complex Flows)

For the existing Langgraph workflow, use `pydantic-graph`:

```python
from pydantic_graph import BaseNode, Graph, End
from dataclasses import dataclass

@dataclass
class WorkflowState:
    query: str
    retrieved_docs: List[Document] = field(default_factory=list)
    plan: Optional[str] = None
    tool_results: List[dict] = field(default_factory=list)
    final_response: Optional[str] = None

@dataclass
class AnalyzeNode(BaseNode[WorkflowState]):
    async def run(self, state: WorkflowState) -> RetrieveNode | PrepareNode:
        # Analyze if retrieval is needed
        needs_retrieval = await analyze_query(state.query)
        if needs_retrieval:
            return RetrieveNode()
        return PrepareNode()

@dataclass
class RetrieveNode(BaseNode[WorkflowState]):
    async def run(self, state: WorkflowState) -> PrepareNode:
        state.retrieved_docs = await retrieve_documents(state.query)
        return PrepareNode()

@dataclass
class PrepareNode(BaseNode[WorkflowState]):
    async def run(self, state: WorkflowState) -> AgentNode:
        # Prepare context for agent
        return AgentNode()

@dataclass
class AgentNode(BaseNode[WorkflowState]):
    async def run(self, state: WorkflowState) -> ExecuteToolsNode | FinalNode:
        result = await run_agent(state)
        if result.needs_tools:
            return ExecuteToolsNode()
        return FinalNode()

@dataclass
class ExecuteToolsNode(BaseNode[WorkflowState]):
    async def run(self, state: WorkflowState) -> AgentNode:
        state.tool_results = await execute_tools(state)
        return AgentNode()

@dataclass
class FinalNode(BaseNode[WorkflowState]):
    async def run(self, state: WorkflowState) -> End:
        state.final_response = await generate_final_response(state)
        return End()

# Create and run graph
graph = Graph(nodes=[AnalyzeNode, RetrieveNode, PrepareNode, AgentNode, ExecuteToolsNode, FinalNode])
result = await graph.run(AnalyzeNode(), state=WorkflowState(query=user_query))
```

---

### Phase 4: Output Parsing Migration (Week 4)

**Goal**: Replace PydanticOutputParser with native PydanticAI validation

#### 4.1 Document Extraction

```python
# Before (Langchain)
from langchain.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=DocumentClassification)
prompt = PromptTemplate(
    template="Classify this document:\n{document}\n{format_instructions}",
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
result = parser.parse(llm_response)

# After (PydanticAI)
from pydantic_ai import Agent
from pydantic import BaseModel

class DocumentClassification(BaseModel):
    departments: List[str]
    categories: List[str]
    topics: List[str]
    sentiment: str

extraction_agent = Agent(
    'anthropic:claude-sonnet-4-0',
    output_type=DocumentClassification,
    instructions="Classify the provided document into departments, categories, topics, and sentiment."
)

async def classify_document(content: str) -> DocumentClassification:
    result = await extraction_agent.run(f"Classify this document:\n\n{content}")
    return result.output  # Already validated Pydantic model
```

#### 4.2 Query Decomposition

```python
# Before (Langchain LCEL)
chain = (
    {"query": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)
result = chain.invoke(query)

# After (PydanticAI)
class DecomposedQuery(BaseModel):
    action: str  # "decompose", "expand", "use_as_is"
    queries: List[str]
    reasoning: str

decompose_agent = Agent(
    'anthropic:claude-sonnet-4-0',
    output_type=DecomposedQuery,
    instructions="""Analyze the query and decide whether to:
    - decompose: Break into sub-queries
    - expand: Add related terms
    - use_as_is: Keep original query"""
)

async def transform_query(query: str) -> DecomposedQuery:
    result = await decompose_agent.run(query)
    return result.output
```

---

### Phase 5: Tool Migration (Week 4-5)

**Goal**: Replace BaseTool with PydanticAI tool decorators

#### 5.1 Fetch Full Record Tool

```python
# Before (Langchain)
from langchain.tools import BaseTool

class FetchFullRecordTool(BaseTool):
    name = "fetch_full_record"
    description = "Fetch full record content"
    args_schema = FetchFullRecordArgs

    def _run(self, record_ids: List[str]) -> str:
        # implementation
        pass

# After (PydanticAI)
@qna_agent.tool
async def fetch_full_record(
    ctx: RunContext[QnADependencies],
    record_ids: List[str]
) -> str:
    """Fetch full record content from the database.

    Args:
        record_ids: List of record IDs to fetch

    Returns:
        Concatenated content of all records
    """
    records = await fetch_records_from_db(record_ids, ctx.deps.org_id)
    return "\n\n".join([r.content for r in records])
```

#### 5.2 Web Search Tool

```python
# Before (Langchain BaseTool)
class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the web"
    args_schema: Type[BaseModel] = WebSearchInput

    def _run(self, query: str, num_results: int = 5) -> str:
        pass

# After (PydanticAI)
@qna_agent.tool
async def web_search(
    ctx: RunContext[QnADependencies],
    query: str,
    num_results: int = 5
) -> str:
    """Search the web for current information.

    Args:
        query: Search query string
        num_results: Number of results to return (default: 5)

    Returns:
        Formatted search results
    """
    from duckduckgo_search import DDGS
    async with DDGS() as ddgs:
        results = [r async for r in ddgs.atext(query, max_results=num_results)]
    return "\n".join([f"- {r['title']}: {r['body']}" for r in results])
```

---

### Phase 6: Streaming Migration (Week 5-6)

**Goal**: Replace Langchain streaming with PydanticAI streaming

#### 6.1 Core Streaming Function

```python
# Before (Langchain)
async def stream_llm_response(llm, messages, tools):
    llm_with_tools = llm.bind_tools(tools)
    async for chunk in llm_with_tools.astream(messages):
        if chunk.content:
            yield {"type": "text", "content": chunk.content}
        if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
            yield {"type": "tool_call", "calls": chunk.tool_calls}

# After (PydanticAI)
async def stream_response(
    agent: Agent,
    query: str,
    deps: Any
) -> AsyncGenerator[dict, None]:
    async with agent.run_stream(query, deps=deps) as result:
        async for text in result.stream_text(delta=True):
            yield {"type": "text", "content": text}

        # Get final result with validation
        output = await result.get_output()
        yield {"type": "complete", "output": output}
```

#### 6.2 Tool Execution Streaming

```python
async def stream_with_tools(
    agent: Agent,
    query: str,
    deps: Any
) -> AsyncGenerator[dict, None]:
    """Stream with tool call visibility."""

    async with agent.iter(query, deps=deps) as run:
        async for node in run:
            if node.kind == "model_request":
                yield {"type": "model_request", "messages": node.messages}
            elif node.kind == "model_response":
                for part in node.parts:
                    if hasattr(part, "content"):
                        yield {"type": "text", "content": part.content}
                    elif hasattr(part, "tool_name"):
                        yield {"type": "tool_call", "tool": part.tool_name, "args": part.args}
            elif node.kind == "tool_return":
                yield {"type": "tool_result", "result": node.return_value}

    yield {"type": "complete", "output": run.output}
```

---

### Phase 7: Cleanup & Testing (Week 6-7)

#### 7.1 Remove Langchain Dependencies

Update `pyproject.toml`:

```toml
# Remove these dependencies
# langchain==0.3.19
# langgraph==0.3.34
# langchain-anthropic==0.3.17
# ... (all 16 langchain packages)

# Add these dependencies
pydantic-ai = "^1.25.0"
pydantic-graph = "^0.1.0"  # If using graph workflows
qdrant-client = "^1.7.0"
fastembed = "^0.2.0"
httpx = "^0.27.0"
tenacity = "^8.2.0"
```

#### 7.2 Delete Deprecated Files

Files to delete after migration:
- Any Langchain-specific adapters
- Unused LCEL chain utilities

#### 7.3 Update Imports Across Codebase

```bash
# Find and replace patterns
# from langchain* -> from app.utils.* (new implementations)
# from langgraph* -> from pydantic_graph (or custom)
```

---

## Detailed File Migration Guide

### High Priority Files

#### 1. `app/utils/aimodels.py` (Lines 1-453)

**Current**: Factory for 15+ LLM providers and 14+ embedding providers
**Migration**:
1. Create `ModelFactory` class using PydanticAI models
2. Create `EmbeddingFactory` using direct API calls
3. Maintain same interface for backward compatibility

```python
# New structure
app/utils/
├── model_factory.py      # PydanticAI model creation
├── embedding_factory.py  # Direct embedding API calls
└── aimodels.py          # Backward-compatible wrapper (temporary)
```

#### 2. `app/utils/streaming.py` (Lines 1-1374)

**Current**: Complex async streaming with tool binding
**Migration**:
1. Replace `llm.bind_tools()` with PydanticAI `@agent.tool`
2. Replace `aiter` streaming with `agent.run_stream()`
3. Update tool result handling

#### 3. `app/modules/transformers/vectorstore.py` (Lines 1-1173)

**Current**: QdrantVectorStore with hybrid search
**Migration**:
1. Replace `langchain_qdrant` with `qdrant-client`
2. Replace `FastEmbedSparse` with direct `fastembed`
3. Implement hybrid search manually

#### 4. `app/modules/agents/qna/graph.py` (Lines 1-235)

**Current**: Langgraph StateGraph with 6 nodes
**Migration**:
1. Convert to `pydantic-graph` nodes
2. Implement state as dataclass
3. Define edges via return types

---

## Risk Assessment

### High Risk Areas

| Area | Risk | Mitigation |
|------|------|------------|
| Streaming | Complex async logic | Extensive testing, feature flags |
| Tool Execution | Different calling conventions | Adapter layer during transition |
| Multi-provider Support | Some providers may not be supported | Custom model implementations |
| Graph Workflows | Different paradigm | Side-by-side testing |

### Medium Risk Areas

| Area | Risk | Mitigation |
|------|------|------------|
| Vector Store | Direct client more complex | Good abstraction layer |
| Output Parsing | Edge cases in validation | Comprehensive test cases |
| Message Conversion | Format differences | Utility conversion functions |

### Low Risk Areas

| Area | Risk | Mitigation |
|------|------|------------|
| Prompt Templates | Simple string replacement | Automated refactoring |
| Basic LLM Calls | Well-supported | Direct mapping |

---

## Testing Strategy

### Unit Tests

```python
# Test model factory
def test_model_factory_anthropic():
    model = ModelFactory.create_model({"provider": "anthropic", "modelName": "claude-sonnet-4-0"})
    assert isinstance(model, AnthropicModel)

# Test embedding service
async def test_openai_embeddings():
    service = OpenAIEmbeddings(api_key="test")
    embeddings = await service.embed_query("hello world")
    assert len(embeddings) == 1536

# Test agent tools
async def test_retrieve_documents_tool():
    deps = QnADependencies(org_id="test", ...)
    result = await retrieve_documents(ctx, "test query")
    assert isinstance(result, str)
```

### Integration Tests

```python
# Test full QnA flow
async def test_qna_agent_e2e():
    deps = create_test_dependencies()
    result = await qna_agent.run("What is PipesHub?", deps=deps)
    assert result.output.answer
    assert result.output.confidence > 0

# Test streaming
async def test_streaming_response():
    chunks = []
    async for chunk in stream_qna_response("test", deps, config):
        chunks.append(chunk)
    assert any(c["type"] == "text" for c in chunks)
    assert chunks[-1]["type"] == "complete"
```

### Parallel Testing Strategy

1. **Feature Flag**: Run both implementations side-by-side
2. **Shadow Mode**: Send requests to both, compare outputs
3. **Gradual Rollout**: 10% -> 50% -> 100% traffic

---

## Migration Checklist

### Phase 1: Foundation
- [ ] Create `ModelFactory` class
- [ ] Create `EmbeddingFactory` class
- [ ] Create custom message types
- [ ] Set up PydanticAI dependency
- [ ] Unit tests for factories

### Phase 2: Vector Store
- [ ] Implement `VectorStoreService` with qdrant-client
- [ ] Migrate hybrid search logic
- [ ] Update `retrieval_service.py`
- [ ] Integration tests for retrieval

### Phase 3: Agents
- [ ] Create `QnAAgent` with PydanticAI
- [ ] Migrate tools to `@agent.tool` decorators
- [ ] Implement `pydantic-graph` workflow
- [ ] E2E agent tests

### Phase 4: Output Parsing
- [ ] Migrate document extraction
- [ ] Migrate query decomposition
- [ ] Migrate domain extraction
- [ ] Validation tests

### Phase 5: Tools
- [ ] Migrate all BaseTool classes
- [ ] Update tool registry
- [ ] Tool execution tests

### Phase 6: Streaming
- [ ] Implement PydanticAI streaming
- [ ] Migrate tool execution streaming
- [ ] Update API endpoints
- [ ] Streaming tests

### Phase 7: Cleanup
- [ ] Remove Langchain dependencies
- [ ] Delete deprecated code
- [ ] Update documentation
- [ ] Performance benchmarks

---

## Resources

- [PydanticAI Documentation](https://ai.pydantic.dev/)
- [PydanticAI GitHub](https://github.com/pydantic/pydantic-ai)
- [pydantic-graph Overview](https://ai.pydantic.dev/graph/)
- [Multi-Agent Patterns](https://ai.pydantic.dev/multi-agent-applications/)
- [Qdrant Client Documentation](https://qdrant.tech/documentation/)
- [FastEmbed Documentation](https://github.com/qdrant/fastembed)

---

## Appendix A: Provider Support Matrix

| Provider | Langchain | PydanticAI | Migration Notes |
|----------|-----------|------------|-----------------|
| Anthropic | ✅ | ✅ | Direct support |
| OpenAI | ✅ | ✅ | Direct support |
| Azure OpenAI | ✅ | ✅ | Via OpenAI model with base_url |
| AWS Bedrock | ✅ | ✅ | Direct support |
| Google Gemini | ✅ | ✅ | Direct support |
| Groq | ✅ | ✅ | Direct support |
| Mistral | ✅ | ✅ | Direct support |
| Ollama | ✅ | ✅ | Direct support |
| Cohere | ✅ | ✅ | Direct support |
| Fireworks | ✅ | ⚠️ | Custom implementation needed |
| Together | ✅ | ✅ | Direct support |
| XAI | ✅ | ⚠️ | Custom implementation needed |
| Vertex AI | ✅ | ✅ | Via Gemini model |
| HuggingFace | ✅ | ⚠️ | Custom implementation needed |

---

## Appendix B: Estimated Effort

| Phase | Effort | Dependencies |
|-------|--------|--------------|
| Phase 1: Foundation | 3-4 days | None |
| Phase 2: Vector Store | 3-4 days | Phase 1 |
| Phase 3: Agents | 5-7 days | Phase 1, 2 |
| Phase 4: Output Parsing | 2-3 days | Phase 1 |
| Phase 5: Tools | 2-3 days | Phase 3 |
| Phase 6: Streaming | 4-5 days | Phase 3, 5 |
| Phase 7: Cleanup | 2-3 days | All phases |

**Total Estimated Effort**: 21-29 days (4-6 weeks)

---

*Document Version: 1.0*
*Created: 2025-11-30*
*Last Updated: 2025-11-30*
