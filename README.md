# JHS Chatbot Optimizer
This is the repository of the RAG Optimization for Conversational Chatbots. The main idea is to add an additional filtering step in the retrieval part of any RAG system using fine-tuned lightweight transformer models. 

## Setup & Prerequisites
The package requires Python 3.12. Please install the necessary packages using `pip install -r requirements.txt`. A GPU for fine-tuning the transformer models is advantageous, but not required.
Please check config.json for basic configuration settings (e.g., http(s) ports).

## Achitecture
The package uses a backend for administration and fine-tuning and a frontend (chatbot) for end users. Besides, it provides a Python library for a default LangChain chatbot implementation. To start the backend, run backend/server.py using Python CLI (not recommended) or a WSGI server. Browse the backend using the specified IP/Port in the config file. The setup process will guide you to the fine-tuning process.

After setup, you can start the default frontend. Start it executing frontend/chat.py using Python CLI or a WSGI server.

### Custom Frondends
Feel free to customize the frontend using the configuration pane in the backend or develop your own frontend and attach it to the provided API using the /sresponse (stream response) endpoint:

```curl --location '/sresponse' \
--header 'Content-Type: application/x-www-form-urlencoded' \
--data-urlencode 'history=[{"role":"user","message":"Your question here"}]'```

The request opens a stream to receive the chat response.

## References
Lowin, Maximilian and Mihale-Wilson, Cristina, "Economic Retrieval-Augmented Generation for Large Language Model Services" (2025). SIG SVC Pre-ICIS Workshop 2024. 7.
https://aisel.aisnet.org/sprouts_proceedings_sigsvc_2024/7
