# setup
## mlflow registry cold start
1. cd to ./lib/model_registry/
1.5. you can select whatever venv youlike python version 3.10
2. pip install -r requirements.txt
3. docker compose -f .\docker_compose.yaml up
4. python model_trial.py  
## to start service
docker compose -f .\compose.yml -d up 
