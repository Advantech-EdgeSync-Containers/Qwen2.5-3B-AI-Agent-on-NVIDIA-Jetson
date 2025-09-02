#!/bin/bash

# Check if MODEL_NAME is set
if [ -z "$MODEL_NAME" ]; then
    echo "MODEL_NAME is not defined in .env file."
    exit 1
fi

# Check if FASTAPI_LANGCHAIN_AGENT_PORT is set
if [ -z "$FASTAPI_LANGCHAIN_AGENT_PORT" ]; then
    echo "FASTAPI_LANGCHAIN_AGENT_PORT is not defined in .env file."
    exit 1
fi

echo "Starting Ollama server in background..."
nohup ollama serve > ollama.log 2>&1 &

# Save PID to a file
echo $! > ollama.pid
echo "Ollama server started (PID: $(cat ollama.pid))"

# Wait for server to become ready
echo "Waiting 5 seconds for server to initialize..."
sleep 5

# Pull the model using MODEL_NAME from .env
echo "Pulling model: $MODEL_NAME ..."
ollama pull "$MODEL_NAME"

# Check if the model was pulled successfully
if [ $? -eq 0 ]; then
    echo "Model '$MODEL_NAME' pulled successfully!"
else
    echo "Failed to pull model '$MODEL_NAME'."
    exit 1
fi

# Starting FastAPI Server
echo "Starting FastAPI (Langchain) server on port $FASTAPI_LANGCHAIN_AGENT_PORT..."
nohup uvicorn app:app --host 0.0.0.0 --port "$FASTAPI_LANGCHAIN_AGENT_PORT" > uvicorn.log 2>&1 &
echo $! > uvicorn.pid
echo "FastAPI server started (PID: $(cat uvicorn.pid))"