# DSPS24 Student Competition

[TEAM JAPA](https://dsps-1e998.web.app/leaderboard)
> [DSPS 24](https://dsps-1e998.web.app/)
```python
"""This is the third student competition on the application of AI for pavement condition monitoring.
Participants will use novel machine learning algorithms to predict the pavement condition index (PCI)
for road sections based on images captured from infrastructure - mounted sensors. 
Training datasets provided consists of Top-down views of pavement image data and corresponding pavement condition indices. 
Participants are free to annotate training datasets and use any model architecture to predict the PCI
of the road section."""
```

> In this competion, we ensemble 3 deep learning models using percentile-based aggregation method.
> 
> The Three models ensembled: <br>
> ```ResNet50``` <br>
> ```ResNet101``` <br>
> `YOLOv8l-cls` <br>
>
>To train the ResNet models use this [notebook](https://github.com/Blessing988/DSPS24/blob/main/Train-ResNet-Model-DSPS24.ipynb) <br>
> Train the ```YOLOv8l-cls``` model using this [notebook](https://github.com/Blessing988/DSPS24/blob/main/Train-YOLOv8-cls-model-DSPS24.ipynb)


> ### Model Checkpoints :
> Download the model checkpoints from the following links: <br>
>
>|Model|Checkpoint|
>|------|----------|
>|ResNet50|[model](https://drive.google.com/file/d/1Jk10bgNx9w4FoJJDi-F2nS6kUhRR_Iv3/view?usp=drive_link)|
> |ResNet101|[model checkpoint](https://drive.google.com/file/d/1m-DWqJTdERL_G9M1nRbbVxav8cokTb2a/view?usp=sharing)|
> |YOLOv8l-cls|[checkpoint](https://drive.google.com/file/d/1q9hR1XHXMjwb68VOOM83ZZBYNnOj2MvR/view?usp=drive_link)|

> You can downlaod the model checkpoins using ```gdown```
> 
>```python
>!gdown --id '1Jk10bgNx9w4FoJJDi-F2nS6kUhRR_Iv3' #ResNet50
>!gdown --id '1m-DWqJTdERL_G9M1nRbbVxav8cokTb2a' # ResNet101
>!gdown --id '1q9hR1XHXMjwb68VOOM83ZZBYNnOj2MvR' #yolov8-cls

> To load any if the pretrained ResNet models: <br>
>E.g: To load ResNet50 : <br>

```python
import torch
import torchvision.models as models
# Load the checkpoint
PATH_TO_MODEL_CHECKPOINT = '<path_to_downloaded_model_checkpoint>'
model = models.resnet50(weights=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(PATH_TO_MODEL_CHECKPOINT)

model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
```
>
> To load a YOLO model
>- Install ```ultralytics``` using `pip` e.g. ```pip install ultralytics```
>```python
>from ultralytics import YOLO
>PATH_TO_MODEL_CHECKPOINT = '<path_to_downloaded_model_checkpoint>'
>model = YOLO(PATH_TO_MODEL_CHECKPOINT)
  
> To generate our final `submission.json` file for the competition : <br>
> Run the cells in the [Inference Notebook](https://github.com/Blessing988/DSPS24/blob/main/Inference_Notebook.ipynb)
