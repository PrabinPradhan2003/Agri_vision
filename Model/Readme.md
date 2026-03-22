# 🌟Model Architecture :
<center><img src ="model.JPG"></center>

## 🌟NOTE :-
* If pdf or ipython notebook will not load then please check markdown filde (.md)
* Download the `plant_disease_model_1_latest.pt` file from [Here](https://drive.google.com/drive/folders/1ewJWAiduGuld_9oGSrTuLumg9y62qS6A?usp=share_link)

## 🌟Train on PlantVillage (recommended)
If you have the PlantVillage dataset locally (39 class folders), you can train and export weights compatible with the Flask app:

`python Model/train_plantvillage.py --data Dataset --epochs 10 --output "Flask Deployed App/plant_disease_model_1_latest.pt"`

## 🌟Blog Link ( Dataset link is in blog ):
<a href="https://medium.com/analytics-vidhya/plant-disease-detection-using-convolutional-neural-networks-and-pytorch-87c00c54c88f" target="_blank">Plant Disease Detection using Convolutional Neural Network with PyTorch Implementation</a>
