# setup
## 1. to start service
docker compose -f .\compose.yml -d up 
wait until allservice are up and airflow-init goes down

## 2.mlflow registry cold start
this is needed to bge run to obtain base line model for production
1. cd to ./lib/model_registry/
1.5. you can select whatever venv youlike python version 3.10
2. pip install -r requirements.txt
3. docker compose -f .\docker_compose.yaml up
4. python model_trial.py  

## 3.run psuedo gpu server on colab
1. clone colab from this link: https://colab.research.google.com/drive/1n2vmSNX4U9VfPFaUxQijNzb1--8S3hX-?usp=sharing
2. then register ngrok and get auth token
3. put auth token in your token feild in the notebook
4. run then copy forwarded end point to API_ENDPOINT in ./dag/.env file  

## 4. set up database
1. get amazon s3 account
2. get credential and put it in in ./dag/.env file according to ./dag/example.env

## 5. run out service
after our gpuserver is on you can run dags you can check status of each pipeline which run on its own. as mlops enginneer you can observe resut from the model monitoring pipeline to schedule finetuning llm as need (this help minimize the unneccessary cost as complex operation is required to obtain new model. which cannot be automated and should not be)

you can access services via these port
- localhost:8080 -> airflow-web
- localhost:5000 -> mlflow-server
