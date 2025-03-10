# SPL-PlaneTR for Robust and Generalizable Indoor Plane Segmentation

## Getting Start

Clone the repository:
```bash
git clone https://github.com/KepWalk/SPL-PlaneTR.git
```

Build the Pytorch environment:
```bash
conda env create -f environment.yaml
```
Make sure to modify the prefix field in the environment.yaml file to specify your own desired virtual environment path.

## Data Preparation
We trained our network using ScanNet and evaluated its performance on the same dataset. Additionally, we utilized Matterport3D, ICL-NUIM, and S2D3DS datasets to validate the generalization capability of our model.
For ScanNet, we used the same plane annotations as those employed in [PlaneTR3D](https://github.com/IceTTTb/PlaneTR3D). For the other datasets used for validation, they can be obtained via [this](https://pan.baidu.com/s/1-TOD1HaO_XT5v3ODQJQOMA?pwd=f2aj).

## Training
Before training, please download the pre-trained weights of [PlaneTR3D](https://github.com/IceTTTb/PlaneTR3D).
Specify the dataset name and path in the configuration file and set the training parameters. To start training, execute the following script: 
```bash
python train_SPLplaneTR.py
```

## Evaluation on ScanNet
You can download the pre-trained model weights via this [link](https://drive.google.com/file/d/1LS1XuyUqspToj-GmwxIn6FF-SoWVW7Ny/view?usp=drive_link).
Before starting, please set the dataset name and path in the configuration file used for validation.Run the following command to evaluate the performance:
```bash
python eval_SPLplaneTR.py
```

## Generalization Evaluation
Evaluate the performance on the other threes datasets:
```bash
python eval_otherDataset.py
```

## Acknowledgements
This code is based on the [PlaneTR3D](https://github.com/IceTTTb/PlaneTR3D) repository. We would like to acknowledge the authors for their work and for providing the foundation upon which this project is built.
