# NEXUS v2 — AI-Powered Finance Research Assistant

**MGMT 69000: Mastering AI for Finance | Purdue University — Daniels School of Business**

NEXUS v2 is a RAG-powered research assistant grounded in the semiconductor export-control case study. It combines retrieval-augmented generation (RAG) with deterministic finance tools to deliver cited, evidence-based answers about Nvidia export controls, competitive dynamics, and market impacts.

> **Core principle:** All numbers come from deterministic tools. The LLM decides *which* tool to call and *explains* the output. It never fabricates data.

---

## Problem Statement

Finance research workflows break in two ways:

1. **Fragmented information** — Export control announcements (BIS), earnings, competitor moves, and hyperscaler CapEx are scattered across regulatory filings, news, and SEC documents.
2. **Ungrounded AI** — Generic LLM answers cannot be trusted for investment decisions. They hallucinate numbers and lack source attribution.

NEXUS v2 solves both by requiring every answer to include:
- **Citations** from retrieved case documents, and/or
- **Tool outputs** computed from live market data

If neither is available, the assistant refuses to speculate.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  STREAMLIT UI                                                │
│  Chat (core) | Event DB | Price Reaction | Ecosystem | Vol   │
├──────────────────────────┬──────────────────────────────────┤
│  CHAT ORCHESTRATOR       │  RAG retrieval → tool calling     │
│  (OpenAI function calling │  → grounding enforcement          │
│   + citation rules)       │  → structured answer format       │
├──────────────┬───────────┴──────────────┬───────────────────┤
│  RAG         │  TOOL SYSTEM             │  ANALYSIS ENGINE   │
│  ChromaDB    │  event_study_tool        │  car_analysis.py   │
│  embeddings  │  volatility_tool         │  options_data.py   │
│  15 docs     │  ecosystem_tool          │  competitor.py     │
│  citations   │  price_tool              │  price_data.py     │
├──────────────┴──────────────────────────┴───────────────────┤
│  DATA LAYER                                                  │
│  11 BIS events | 11 tickers | yfinance | case study docs     │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Grounded Chat** | Every response includes citations `[doc_id]` and/or tool evidence |
| **Event Study Tool** | Market-model CAR with configurable windows and benchmarks |
| **Volatility Tool** | Live IV surface, skew, term structure, historical realized vol |
| **Ecosystem Tool** | 11-ticker universe: NVDA, AMD, INTC, TSM, ASML, AVGO, GOOGL, AMZN, MSFT |
| **Price Tool** | Returns, correlations, drawdowns for any ticker combination |
| **RAG Knowledge Base** | 15 grounding documents (events, methodology, case study) in ChromaDB |
| **CI/CD Pipeline** | GitHub Actions with Python 3.11/3.12 matrix testing |

---

## Semiconductor Ecosystem (Expanded)

| Group | Tickers | Export Control Exposure |
|-------|---------|----------------------|
| GPU Leaders | NVDA, AMD | Direct — primary targets |
| Legacy Semi | INTC | Moderate — Gaudi + foundry |
| Foundry/Equipment | TSM, ASML | Indirect — manufacture & equip restricted chips |
| Networking | AVGO | Low — limited China AI exposure |
| Hyperscalers | GOOGL, AMZN, MSFT | Indirect — custom chips + cloud services |

---

## Setup

### 1. Clone and install

```bash
git clone <your-repo-url>
cd nexus
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API key

**Local development:**
```bash
export OPENAI_API_KEY="sk-your-key-here"
```

**Or** create `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "sk-your-key-here"
```

### 3. Build the vector index

```bash
python scripts/build_index.py
```

### 4. Run the app

```bash
streamlit run app.py
```

### 5. Run tests

```bash
pytest tests/ -v
```

---

## Methodology

### Event Study (CAR Analysis)
- **Model:** Market model (OLS regression: R_stock = α + β·R_market + ε)
- **Estimation window:** 120 trading days prior to event
- **Event window:** Configurable (default: -1 to +5 days)
- **Benchmark:** S&P 500 (SPY)
- **Significance:** Two-tailed t-test at p < 0.05
- **Key finding:** Only 1/11 events statistically significant (May 2023 earnings)

### RAG System
- **Embeddings:** OpenAI `text-embedding-3-small`
- **Vector store:** ChromaDB with cosine similarity
- **Chunking:** Markdown-aware, ~1000 tokens per chunk
- **Retrieval:** Top-6 similarity search with optional metadata filtering
- **Grounding:** Citations required; refusal when no evidence available

### Volatility Analysis
- **Implied vol:** Extracted from yfinance options chain (Black-Scholes)
- **Skew:** IV(OTM puts) − IV(ATM) at 30d and 60d tenors
- **Term structure:** ATM IV across expirations (contango/backwardation)
- **Historical proxy:** 20-day and 60-day annualized realized volatility

---

## Data Sources

| Source | What | Reliability |
|--------|------|-------------|
| BIS Federal Register | Export control announcements | Curated summaries |
| SEC filings | Nvidia 10-K, 10-Q, 8-K | Referenced in events |
| Yahoo Finance (yfinance) | Prices + options chain | Free tier, rate-limited |
| Course case materials | Case study narrative | Manually processed |

---

## Project Structure

```
nexus/
├── app.py                    # Streamlit entry point (5 tabs)
├── core/
│   ├── config.py             # Settings and constants
│   ├── llm/
│   │   ├── client.py         # OpenAI wrapper (chat + embeddings)
│   │   └── prompts.py        # System prompt with grounding rules
│   ├── rag/
│   │   ├── chunking.py       # Markdown-aware chunking
│   │   ├── vector_store.py   # ChromaDB persistence + search
│   │   └── retriever.py      # Context retrieval + formatting
│   └── chat/
│       └── agent.py          # Chat orchestrator (RAG + tools)
├── tools/
│   ├── event_study_tool.py   # CAR analysis tool
│   ├── volatility_tool.py    # IV surface + skew tool
│   ├── ecosystem_tool.py     # Multi-ticker comparison tool
│   └── price_tool.py         # Price/returns/correlation tool
├── data/
│   ├── export_control_events.py  # 11 curated BIS events
│   ├── price_data.py         # yfinance price fetcher
│   ├── car_analysis.py       # Event study engine
│   ├── competitor_analysis.py # Ecosystem analysis
│   ├── options_data.py       # Options chain + vol surface
│   └── universe.py           # Expanded ticker universe
├── knowledge/
│   └── processed/            # 15 grounding documents
├── tests/                    # 18 unit tests
├── scripts/
│   └── build_index.py        # Vector index builder
├── .github/workflows/ci.yml  # CI/CD pipeline
└── requirements.txt
```

---

## DRIVER Methodology

| Phase | Activity |
|-------|----------|
| **Define** | Scope: export controls + semiconductor ecosystem. Non-negotiable: grounded answers only. |
| **Represent** | Knowledge documents with metadata, tool I/O schemas, 11-ticker universe |
| **Implement** | ChromaDB vector store, 4 tool wrappers, chat orchestrator, 5-tab Streamlit UI |
| **Validate** | 18 unit tests, CI pipeline, grounding compliance checks, evaluation prompts |
| **Evolve** | Expanded from 3 → 11 tickers, added RAG + tools, added CI/CD |
| **Reflect** | Documented limitations, acknowledged statistical power issues, planned improvements |

---

## Evaluation Framework

### Grounding Compliance
Every answer must include citations `[doc_id]` and/or tool output references. Answers without evidence trigger refusal.

### Expected Tool Usage

| Question | Expected Tool |
|----------|--------------|
| "Run CAR for NVDA around Oct 7 rules" | `event_study_tool` |
| "Compare NVDA vs TSM vs ASML since 2023" | `ecosystem_tool` |
| "Is NVDA put skew elevated?" | `volatility_tool` |
| "What is NVDA current price and drawdown?" | `price_tool` |
| "Explain the October 7 controls" | RAG retrieval (no tool) |

### Finance Reasoning Rubric (1-5 per answer)
1. Grounding correctness (citations + tools used)
2. Parameter transparency (windows, benchmarks stated)
3. No hallucinated numbers
4. Correct interpretation (CAR ≠ raw return, IV ≠ RV)
5. Actionable insight

---

## Limitations

- **Options data** requires market hours for live chain; outside hours, only historical realized vol is shown
- **yfinance** is rate-limited and may fail under heavy concurrent use
- **CAR significance:** Only 1/11 events is statistically significant — this reflects genuine low statistical power, not a bug
- **Future events** (2025-2026) use estimated reactions where actual data is not yet available
- **RAG corpus** is limited to curated documents; adding SEC filings would improve coverage

## Future Improvements

- Add SEC filing ingestion (EdgarTools) for automatic document expansion
- Integrate paid data source (Polygon.io) for reliable historical IV
- Add conversation memory with summarization for long sessions
- Add factor model (Fama-French 3-factor) for more robust expected returns
- Add backtesting module for event-driven trading strategies

---

## Deployment (Streamlit Cloud)

1. Push to GitHub
2. Connect repo to [Streamlit Cloud](https://streamlit.io/cloud)
3. Set `OPENAI_API_KEY` in App Settings → Secrets
4. App auto-deploys on every push to `main`

---

*Built with DRIVER methodology | MGMT 69000 — Purdue University*
