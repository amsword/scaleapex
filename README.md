# ScaleApex
This repo implements a simple way to combine [apex](https://github.com/NVIDIA/apex) 
and [fairscale](https://github.com/facebookresearch/fairscale).
Some background can be found [here](http://jianfengwang.me/A-simple-way-to-combine-fairscale-and-apex/).

# Installation
```shell
git clone https://github.com/amsword/scaleapex
cd scaleapex
python setup build develop
```

# Example
```shell
mpirun -n 2 python example/example.py
```
