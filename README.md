# LaDEEP
This is the implement of "LaDEEP: Machine Learning for Large Deformation of Elastic-Plastic Solid"

## Usage

We recommend that `python>=3.10`.

### Install packages

```
git clone git@github.com:SuperJokerayo/LaDEEP.git
cd ./LaDEEP
pip install -r ./requirements.txt
```

### Dataset

Due to certain confidentiality principles, the dataset has not been published on the internet. However, we provide it for educational purposes. If you need it, you can contact the email shilongtao@stu.pku.edu.cn and state the reason for use. We will respond as soon as possible.

After obtaining the dataset, put it in the `./data` folder. You can also change the data folder whatever you want in `./config.ini` file.

### Train

Several hyperparameters are recorded in `./config.ini` file. You can change them to train the model. The default settings are corresponding to those used in the paper.

Then use the below command to start the train:

```
mkdir ./checkpoints
mkdir ./logs

# Display training process on the frontend
python main.py

# Display training process on the backend
nohup python main.py >> ./train.log 2>&1 &
```

The training and evaluation losses are illustrated by `tensorboard`:

```
tensorboard --logdir ./logs --port 8888
```

Then you can monitor the training and evaluation details by open `localhost:8888` on your browser.

### Test

We have provided model weights for [download]().

To test the model, we first need to change the mode in `config.ini`:

```
mode = test
```

and the `mode_id` is the folder `./checkpoints/train_{mode_id}` that saves the checkpoints you wanna use. Then test the model:

```
python main.py
```

and the results will be save in `./data/prediction_results/test_{mode_id}`.
