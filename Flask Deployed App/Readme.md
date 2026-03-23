# 🌟Make Sure You Read all Instructions :

##### This code is fully funcional web app which is already deployed in heroku cloud server

##### You have install requirements.txt for run this code in your pc

##### For heroku we also have to create on Procfile

##### Your "plant_disease_model_1_latest.pt" should be in this folder. You have to trained that model in your pc/laptop and drag it to this folder

##### First check the Model section of this Repo. After that you can understand deployed app.

##### Make sure if you change the model name then also change the name of the model argument in the app.py


# ⭐Requirements 
#### You have to Installed all the requirments. Save all the below requirements in requirements.txt
#### Run this line in cmd/shell :  pip install -r requirements.txt


# ✅ Python version note (Windows)

This project’s original `requirements.txt` is pinned to older packages (not compatible with Python 3.13).

If you are on a newer Python, use Python **3.11** and install with:

1) Install app dependencies:

`pip install -r "requirements-py311.txt"`

2) Install CPU PyTorch + torchvision:

`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`


# ✅ Hugging Face token note

Hugging Face has migrated to **Inference Providers**. Make sure your `HF_API_TOKEN` is a **fine-grained** token with permission: **Make calls to Inference Providers**. Otherwise chat requests will fail with HTTP 403.
