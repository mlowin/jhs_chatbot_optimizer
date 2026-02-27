# JHS Chatbot Optimizer
This is the repository of the RAG Optimization for Conversational Chatbots. The main idea is to add an additional filtering step in the retrieval part of any RAG system using fine-tuned lightweight transformer models. 

## Setup & Prerequisites
The package requires Python 3.12. Please install the necessary packages using `pip install -r requirements.txt`. A GPU for fine-tuning the transformer models is advantageous, but not required.

## Architecture
The package uses a backend for administration and fine-tuning and a frontend (chatbot) for end users. Besides, it provides a Python library for a default LangChain chatbot implementation. To start the backend, run "cd backend" and "python server.py" using Python CLI (not recommended) or a WSGI server. Browse the backend using the specified IP/Port in the server.py file. The setup process will guide you to the fine-tuning process. Besides, you need to run the task handling service. Run "python llm_server.py" in a second thread or a dedicated WSGI server. After setup, you can start the default frontend. Start it executing frontend/chat.py using Python CLI or a WSGI server. Please read the section "Manual" that describes the complete process. Each service (backend, llm_server and frontend chat) needs to run in its own instance (an own WSGI server or an own terminal window/session).

### Custom Frondends
Feel free to customize the frontend using the configuration pane in the backend or develop your own frontend and attach it to the provided API using the /sresponse (stream response) endpoint:

```bash
curl --location '/sresponse' \
--header 'Content-Type: application/x-www-form-urlencoded' \
--data-urlencode 'history=[{"role":"user","message":"Your question here"}]'
```

The request opens a stream to receive the chat response.

## Manual
After successful installation of the python packages and running the backend service ("pyhon server.py" or WSGI), you can access the admin dashboard via the specified port (default is 2337). You will find an empy instance of the JHS tool.

### Setting up LLM
Please start with setting up a first LLM. This can be a local one or a hosted one. For example, when using OpenRouter, go to "LLMs", click on "Add LLM" and enter a title for your LLM (custom label), the model name, e.g. "openai/gpt-5.2", the URL to the api, e.g., "https://openrouter.ai/api/v1" and your API token. Besides, you need to add at least one LLM for backend usage, otherwise you cannot start the JHS integration process. Important: Once you changed something in the configuration, you need to restart the llm_server (run "python llm_server.py" or [re]start the WSGI server). Besides, you also need to specify at least one LLM for frontend use. As you cannot start the frontend until you successfully completed the full integration process, you can also do that later. Please note that one LLM instance can be used for frontend and backend at the same time.

### Starting the Integration
You can now start the JHS integration process by clicking on "Integrations" in the JHS backend. It asks you to upload your dataset. Your dataset should contain several products with an ID, title, description text(s) and some possible filters. The dataset can be CSV, XLSX or JSON. After selecting the file, click on "upload and process". 

The llm_server will start the column extraction process. This might take some time depending on the selected LLM and dataset. Typically, after 30 seconds, you should see an overview of the dataset columns and the respective data types of the values and the determined role of each field. The role is crucial for the JHS tool. There are five different roles available:
* id: A unique identifier for each product.
* title: A title or describing name for each product.
* description: A text description for each product. 
* filter: A column that can be used as a filter to filter the products.
* no-filter: A column that does not match the previous defintions. It will be ignored by the JHS tool.

Important: Please note that there can be multiple description columns. Please select all accordingly. Each column will be used for filter creation and extraction as well as serve as an input for the RAG.

After veryfing the correct column types, please click on "save and continue". 

The JHS tool now asks you regarding the type of products you have in your dataset. Please enter something like "board games" or "staples". Besides, it asks you regarding the number of articles it should use to create synthetic filters for the dataset. This should be a significant share of products to ensure the JHS tool can generate reasonable filters. We suggest 10% of all articles. Please note that this number is an absolute number you need to specify here (500 means 500 products). Thus, it depends on the number of items you have in your dataset. Click then on "save and generate". It will now take some time to extract and merge potential filters. Afterward, you can review the extracted filters and modify them accordingly. Please add all filters you think are necessary and meaningful for your setting. By clicking on "save and continue", the JHS tool will now extract the specified filters from all the products you provided in the dataset. Consequently, it creates a vector database for all products. Depending on your dataset size and hardware, it can take a couple of minutes, up to an hour. 

Note: The vector database creation is prone to errors. Please watch the status of the llm_server accordingly and restart it, if it crashes.

After vector database creation the JHS tool will create synthetic user queries. Therefore, it first creates persona of potential customers. Please review the persona creation prompt for your scenario accordingly. You can then create persona. We recommend to create approximately 50 persona. After creating the persona, you can review the persona definition. Please ensure that you only save valid JSON. 

In the next step, the synthetic user query generation starts. The JHS tool now shows you the creation prompt. The JHS tool will run this query depending on the amount you specify in the field "number of user queries to generate". It will run, e.g., 5000 times and attach one persona in each iteration. Besides, it can append some product descriptions, so the user queries fit to the product assortment. You can specify the number of articles you want to have for each iteration in the respective input field. 

If you hit "create user queries", the tool will create the specified amout of user queries. Afterward, it extracts the filters from the first half of the JHS process on these user queries. Then, it fine tunes a transformer model on this synthetic dataset that will be used in the frontend chat. This process can take a couple of hours, depending on the specified amout of user queries, the respective hardware and the selected LLM. It will show if the pipeline is finished.

### Starting the Frontend
Now, you can start the frontend chatbot. If you have not specified a LLM for frontend usage, please do it now in the backend template under the menu point "LLMs". You can then start the frontend server (cd to frontend/ and run python chat.py or start an WSGI server). The default port for the chat is 2339.

### Observing the performance
In the backend you can open the menu point "eports". Here, you can see the entered user queries in your chatbot. When inspecting a chat, you can see the extracted filters and products from the vector database that the JHS chatbot sends to the LLM.

## References
Lowin, Maximilian and Mihale-Wilson, Cristina, "Economic Retrieval-Augmented Generation for Large Language Model Services" (2025). SIG SVC Pre-ICIS Workshop 2024. 7.
https://aisel.aisnet.org/sprouts_proceedings_sigsvc_2024/7
