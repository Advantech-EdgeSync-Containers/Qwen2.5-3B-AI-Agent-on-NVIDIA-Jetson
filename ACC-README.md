Harness the power of the Qwen & LangChain-based AI Agent to interact with the EdgeSync Device Library on NVIDIA Jetson™. This container bundles the LangChain AI Agent integrated with the EdgeSync Device Library served via FastAPI, Ollama, and the Qwen 2.5 3B model. It features an AI agent with EdgeSync Device Library integration for natural language-driven control of peripherals and edge hardware.

# Qwen2.5 3B AI Agent on NVIDIA Jetson™

### About Advantech Container Catalog
The Advantech Container Catalog offers plug-and-play, GPU-accelerated container images for Edge AI development on NVIDIA Jetson™. These containers abstract hardware complexity, enabling developers to build and deploy AI solutions without worrying about drivers, runtime, or CUDA compatibility.

### Key benefits of the Container Catalog include:
| Feature / Benefit                              | Description                                                                |
|----------------------------------------------------|--------------------------------------------------------------------------------|
| Accelerated Edge AI Development                    | Ready-to-use containerized solutions for fast prototyping and deployment       |
| Hardware Compatibility Solved                      | Eliminates embedded hardware and AI software package incompatibility           |
| GPU/NPU Access Ready                               | Supports passthrough for efficient hardware acceleration                       |
| Model Conversion & Optimization                    | Built-in AI model quantization and format conversion support                   |
| Optimized for CV & LLM Applications                | Pre-optimized containers for computer vision and large language models         |
| Open Ecosystem                                     | 3rd-party developers can integrate new apps to expand the platform             |


## Container Overview

The Qwen2.5 3B AI Agent on NVIDIA Jetson™ provides a plug-and-play AI runtime for NVIDIA Jetson™ devices, integrating the Qwen 2.5 3B model (via Ollama) with a FastAPI-based LangChain AI Agent (integrated with EdgeSync Device Library) and OpenWebUI interface. This container offers:

- Offline, on-device LLM inference using Qwen 2.5 3B via Ollama (no internet required post-setup)
- LangChain middleware with FastAPI for orchestrating modular pipelines
- Built-in FAISS vector database for efficient semantic search and RAG use case
- Agent support to enable autonomous, multi-step task execution and decision-making
- Prompt memory and context handling for smarter conversations
- Streaming chat UI via OpenWebUI
- OpenAI-compatible API endpoints for seamless integration
- Customizable model parameters via modelfile & environment variables
- AI Agent integrated with EdgeSync Device Library for calling various peripheral functions via natural language prompts
- Predefined LangChain tools (functions) registered for the agent to call hardware APIs

## Container Demo
![Demo](data%2Fgifs%2Fqwen-ai-agent.gif)

## Use Cases

- Predictive Maintenance Chatbots: Integrate with edge telemetry or logs to summarize anomalies, explain error codes, or recommend corrective actions using historical context.
- Compliance and Audit Q&A: Run offline LLMs trained on local policy or compliance data to assist with audits or generate summaries of regulatory alignment—ensuring data never leaves the premises.
- Safety Manual Conversational Agents: Deploy LLMs to provide instant answers from on-site safety manuals or procedures, reducing downtime and improving adherence to protocols.
- Technician Support Bots: Field service engineers can interact with the bot to troubleshoot equipment based on past repair logs, parts catalogs, and service manuals.
- Smart Edge Controllers: LLMs can translate human intent (e.g., “reduce line 2 speed by 10%”) into control commands for industrial PLCs or middleware using AI agents.
- Conversational Retrieval (RAG): Integrate with vector databases (like FAISS and ChromaDB) to retrieve relevant context from local documents and enable conversational Q&A over your custom data.
- Tool-Enabled Agents: Create intelligent agents that use calculators, APIs, or search tools as part of their reasoning process—LangChain handles the logic and LLM interface.
- Factory Incident Reporting: Ingest logs or voice input → extract incident type → summarize → trigger automated alerts or next steps
- Custom Tool-Driven Agents: Expand the system with new LangChain tools to call additional hardware functions, fetch local metrics, or trigger external workflows—all via natural language.

## Key Features

- LangChain Middleware: Agent logic with memory and modular chains
- Ollama Integration: Lightweight inference engine for quantized models
- Complete AI Framework Stack: PyTorch, TensorFlow, ONNX Runtime, and TensorRT™
- Industrial Vision Support: Accelerated OpenCV and GStreamer pipelines
- Edge AI Capabilities: Support for computer vision, LLMs, and time-series analysis
- Performance Optimized: Tuned specifically for NVIDIA® Jetson Orin™ NX 8GB
- EdgeSync Integration with Agent Integration of the EdgeSync Device Library with the agent to interact with low-level edge hardware components via natural language

## Host Device Prerequisites
| Item | Specification                                                                                                                                                             |
|-----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Compatible Hardware | Advantech devices accelerated by NVIDIA Jetson™—refer to [Compatible Hardware](https://catalog.advantech.com/en-us/containers/jetson-gpu-passthrough/compatible-hardware) |
| NVIDIA Jetson™ Version | 5.x                                                                                                                                                                       |
|Host OS          | Ubuntu 20.04                                                                                                                                                              |
| Required Software Packages | Refer to Below                                                                                                                                                            |
| Software Installation | [NVIDIA Jetson™ Software Package Installation](https://developer.advantech.com/EdgeSync/Containers/Environment/NVIDIA)                                                    |                                                                                                        |


## Container Environment Overview

### Software Components on Container Image

| Component    | Version        | Description                                                                                                      |
|--------------|----------------|------------------------------------------------------------------------------------------------------------------|
| CUDA®        | 11.4.315       | GPU computing platform                                                                                           |
| cuDNN        | 8.6.0          | Deep Neural Network library                                                                                      |
| TensorRT™    | 8.5.2.2        | Inference optimizer and runtime                                                                                  |
| PyTorch      | 2.0.0+nv23.02  | Deep learning framework                                                                                          |
| TensorFlow   | 2.12.0 | Machine learning framework                                                                                       |
| ONNX Runtime | 1.16.3         | Cross-platform inference engine                                                                                  |
| OpenCV       | 4.5.0          | Computer vision library with CUDA®                                                                               |
| GStreamer    | 1.16.2         | Multimedia framework                                                                                             |
| Ollama       | 0.5.7          | LLM inference engine                                                                                             |
| LangChain    | 0.2.17         | Orchestration layer for memory, RAG, and agent workflows                                                         |
| FastAPI      | 0.115.12       | API service exposing LangChain interface                                                                         |
| OpenWebUI    | 0.6.5          | Web interface for chat interactions                                                                              |
| FAISS        | 1.8.0.post1    | Vector store for RAG pipelines                                                                                   |
| EdgeSync     | 1.0.0          | EdgeSync is provided as part of the container image for low-level edge hardware components interaction with the AI Agent. |


### Container Quick Start Guide
For container quick start, including the docker-compose file and more, please refer to [README.](https://github.com/Advantech-EdgeSync-Containers/Nagarro-Container-Project/blob/main/Qwen2.5-3B-AI-Agent-on-NVIDIA-Jetson/README.md)

### Supported AI Capabilities

#### Vision Models

| Model Family    | Versions                                                                   | Performance (FPS)                                                            | Quantization Support |
|-----------------|----------------------------------------------------------------------------|------------------------------------------------------------------------------|----------------------|
| YOLO            | v3/v4/v5 (up to v5.6.0), v6 (up to v6.2), v7 (up to v7.0), v8 (up to v8.0) | YOLOv5s: 45-60 @ 640x640, YOLOv8n: 40-55 @ 640x640, YOLOv8s: 30-40 @ 640x640 | INT8, FP16, FP32     |
| SSD             | MobileNetV1/V2 SSD, EfficientDet-D0/D1                                     | MobileNetV2 SSD: 50-65 @ 300x300, EfficientDet-D0: 25-35 @ 512x512           | INT8, FP16, FP32     |
| Faster R-CNN    | ResNet50/ResNet101 backbones                                               | ResNet50: 3-5 @ 1024x1024                                                    | FP16, FP32           |
| Segmentation    | DeepLabV3+, UNet                                                           | DeepLabV3+ (MobileNetV2): 12-20 @ 512x512                                    | INT8, FP16, FP32     |
| Classification  | ResNet (18/50), MobileNet (V1/V2/V3), EfficientNet (B0-B2)                 | ResNet18: 120-150 @ 224x224, MobileNetV2: 180-210 @ 224x224                  | INT8, FP16, FP32     |
| Pose Estimation | PoseNet, HRNet (up to W18)                                                 | PoseNet: 15-25 @ 256x256                                                     | FP16, FP32           |

### Language Models Recommendation

| Model Family | Parameters | Quantization | Size | Performance  |
|--------------|------------|--------------|------|--------------|
| DeepSeek R1 | 1.5 B | Q4_K_M | 1.1 GB | ~15-17 tokens/sec |
| DeepSeek R1 | 7 B | Q4_K_M | 4.7 GB | ~5-7 tokens/sec |
| DeepSeek Coder | 1.3 B | Q4_0 | 776 MB | ~20-25 tokens/sec |
| Llama 3.2 | 1 B | Q8_0 | 1.3 GB | ~17-20 tokens/sec |
| Llama 3.2 Instruct | 1 B | Q4_0 | ~0.8 GB | ~17-20 tokens/sec |
| Llama 3.2 | 3 B | Q4_K_M | 2 GB | ~10-12 tokens/sec |
| Llama 2 | 7 B | Q4_0 | 3.8 GB | ~5-7 tokens/sec |
| Tinyllama | 1.1 B | Q4_0 | 637 MB | ~22-27 tokens/sec |
| Qwen 2.5 | 0.5 B | Q4_K_M | 398 MB | ~25-30 tokens/sec |
| Qwen 2.5 | 1.5 B | Q4_K_M | 986 MB | ~15-17 tokens/sec |
| Qwen 2.5 Coder | 0.5 B | Q8_0 | 531 MB | ~25-30 tokens/sec |
| Qwen 2.5 Coder | 1.5 B | Q4_K_M | 986 MB | ~15-17 tokens/sec |
| Qwen | 0.5 B | Q4_0 | 395 MB | ~25-30 tokens/sec |
| Qwen | 1.8 B | Q4_0 | 1.1 GB | ~15-20 tokens/sec |
| Gemma 2 | 2 B | Q4_0 | 1.6 GB | ~10-12 tokens/sec |
| Mistral | 7 B | Q4_0 | 4.1 GB | ~5-7 tokens/sec |                                     |

*Tuning Tips for Efficient RAG and Agent Workflows:**
- Use asynchronous chains and streaming response handlers to reduce latency in FastAPI endpoints.
- For RAG pipelines, use efficient vector stores (e.g., FAISS with cosine or inner product) and pre-filter data when possible.
- Avoid long chain dependencies; break workflows into smaller composable components.
- Cache prompt templates and tool results when applicable to reduce unnecessary recomputation
- For agent-based flows, limit tool calls per loop to avoid runaway execution or high memory usage.
- Log intermediate steps (using LangChain’s callbacks) for better debugging and observability
- Use models with ≥3B parameters (e.g., Llama 3.2 3B or larger) for agent development to ensure better reasoning depth and tool usage reliability.

## Supported AI Model Formats

| Format | Support Level | Compatible Versions | Notes |
|--------|---------------|---------------------|-------|
| ONNX | Full | 1.10.0 - 1.16.3 | Recommended for cross-framework compatibility |
| TensorRT™ | Full | 7.x - 8.5.x | Best for performance-critical applications |
| PyTorch (JIT) | Full | 1.8.0 - 2.0.0 | Native support via TorchScript |
| TensorFlow SavedModel | Full | 2.8.0 - 2.12.0 | Recommended TF deployment format |
| TFLite | Partial | Up to 2.12.0 | May have limited hardware acceleration |
| GGUF | Full | v3 | Format used by Ollama backend |

## Hardware Acceleration Support

| Accelerator | Support Level | Compatible Libraries | Notes |
|-------------|---------------|----------------------|-------|
| CUDA® | Full | PyTorch, TensorFlow, OpenCV, ONNX Runtime | Primary acceleration method |
| TensorRT™ | Full | ONNX, TensorFlow, PyTorch (via export) | Recommended for inference optimization |
| cuDNN | Full | PyTorch, TensorFlow | Accelerates deep learning primitives |
| NVDEC | Full | GStreamer, FFmpeg | Hardware video decoding |
| NVENC | Full | GStreamer, FFmpeg | Hardware video encoding |
| DLA | Partial | TensorRT™ | Requires specific model optimization |
