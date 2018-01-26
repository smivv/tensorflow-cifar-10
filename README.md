# CIFAR-10 dataset convolutional neural network

Project is written to show the usage of Tensorflow High-Level API with Cifar-10 dataset.
[Experiment](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/Experiment),
[Estimator](https://www.tensorflow.org/api_docs/python/tf/estimator), Dataset, Model and [SessianRunHook](https://www.tensorflow.org/api_docs/python/tf/train/SessionRunHook) classes were implemented.  
In addition [embeddings](https://www.tensorflow.org/programmers_guide/embedding) training implemented.

Test accuracy = 0.7166, loss = 1.09975.

#### Architecture diagram

<div align=center><img src="https://user-images.githubusercontent.com/17829173/34811773-94bbe0a4-f6b3-11e7-88d9-6ac9c464f98c.png"/></div>

#### Embeddings visualization

![Embeddings visualization](https://user-images.githubusercontent.com/17829173/34813290-76aff0b2-f6ba-11e7-8928-7f8b365a687c.PNG)

![Embeddings visualization](https://user-images.githubusercontent.com/17829173/34813291-76cae41c-f6ba-11e7-8717-d9e6edb1dbe3.PNG)

### Installing

What you need to do

```
git clone https://github.com/smivv/tensorflow-cifar-10
cd tensorflow-cifar-10-master
sh install.sh
```

## Usage Example

### Training

```
python run.py --train --checkpoint_dir=/home/smirnvla/PycharmProjects/python-tensorflow-cifar-10/tmp/checkpoints/ --dataset_dir=/workspace/Datasets/cifar-10/
```

### Evaluation

```
python run.py --evaluate --checkpoint_dir=/home/smirnvla/PycharmProjects/python-tensorflow-cifar-10/tmp/checkpoints/ --dataset_dir=/workspace/Datasets/cifar-10/
```

## Built With

* [Python](https://www.python.org/) - The language used.
* [Tensorflow](https://www.tensorflow.org/) - The library used.

## Authors

* **Vladimir Smirnov** - *Initial work* - [smivv](https://github.com/smivv)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details