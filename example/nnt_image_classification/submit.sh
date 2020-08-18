#PBS -S /bin/bash
#PBS -q gpu_q
#PBS -N resnet_model
#PBS -l nodes=1:ppn=12:gpus=1:P100:default
#PBS -l walltime=200:00:00
#PBS -l mem=60gb
#PBS -M yuewu_mike@163.com
#PBS -m abe
cd $PBS_O_WORKDIR
echo $PBS_GPUFILE


module load PyTorch/1.2.0_conda

# source activate ${PYTORCHROOT}
source activate /home/yw44924/methods/pytorch_env

time python3 image_classify_transfer.py  --batch-size 15 --test-batch-size 15 --epochs 20 --learning-rate 0.001 --seed 1 --gpu-use 1 --net-struct resnet34 --pretrained 1 --freeze-till layer4 --optimizer sgd 1>> ./testmodel.${PBS_JOBID}.out 2>> ./testmodel.${PBS_JOBID}.err

source deactivate
