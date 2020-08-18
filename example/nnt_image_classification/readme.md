# Example folder for deep learning image classification project

data: different data set for training and external validation

pretrained: pretrained models

res: results

utilis: function and script needed for data preparation

image_classify_transfer.py: the example script for main model training

parameter_sampler.R: the R script to start multiple different model with different hyperparameters

plot.mse.epoch.small.r: plotting model performance through epoch and summary table for model performance

res_visual_trainedmodel.py: looking at model performance and plotting

submit.sh: example submission script

submitlist.tab: example table for different training models

Example running command:

```
time python3 image_classify_transfer.py  --batch-size 15 --test-batch-size 15 --epochs 500 --learning-rate 0.001 --seed 2 --net-struct resnet50 --optimizer sgd   --pretrained 1 --freeze-till layer3  1>> ./model.out 2>> ./model.err
```

Before you run this file, please make sure you have gpu to run, have the image data for training, and have the pretrained model ready.

example folder structure on sever:
  image_classify_transfer.py
  submitlist.tab
  submit.sh
  pretrained/
  data/
