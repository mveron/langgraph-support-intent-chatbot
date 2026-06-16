# LangGraph + Ollama Demo

Educational demo of a conditional graph built with LangGraph, Ollama, and the local `qwen3:4b` model. It includes a command-line interface and a Streamlit app that reuse the same graph to classify a question and route it to a technical or general answer.

## Graph Flow

```mermaid
flowchart LR
    START([START]) --> classify[classify]
    classify -->|technical| answer_technical[answer_technical]
    classify -->|general| answer_general[answer_general]
    answer_technical --> END([END])
    answer_general --> END
```

## Requirements

- Python 3.11+
- Ollama installed and running
- Local `qwen3:4b` model

## Installation

Run these commands from PowerShell at the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama list
```

If `ollama list` does not show `qwen3:4b`, download the model:

```powershell
ollama pull qwen3:4b
```

## Run the Demo

CLI:

```powershell
.\.venv\Scripts\python.exe cli.py
```

Streamlit web app:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

Tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

## What to Explain During the Demo

- `GraphState` is the shared state that moves through the graph: question, category, answer, and trace.
- Each node reads state and returns only partial updates, such as `category`, `answer`, or new `trace` entries.
- `add_conditional_edges` decides whether execution continues through `answer_technical` or `answer_general`.
- `stream_mode="updates"` lets you observe each graph update step by step.
- The CLI and web app reuse the same graph defined in `graph.py`; only the presentation layer changes.

## File Structure

- `graph.py`: defines `GraphState`, graph nodes, and conditional edges.
- `llm.py`: adapts Ollama as the local text model using `qwen3:4b`.
- `runner.py`: runs the full graph or streams it step by step.
- `cli.py`: command-line interface for asking questions.
- `app.py`: Streamlit interface with route visualization.
- `visualization.py`: generates the DOT diagram used by the web app.
- `tests/`: automated tests for the graph, runner, CLI, LLM, and visualization.

## Troubleshooting

If Ollama is unavailable, the CLI or Streamlit app will show an error explaining that `qwen3:4b` could not be used.

Check the following:

```powershell
ollama list
ollama serve
```

If the service is running but the model is missing:

```powershell
ollama pull qwen3:4b
```

Then run the CLI, web app, or tests again as needed.
