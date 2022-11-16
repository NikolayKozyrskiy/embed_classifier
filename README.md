# Embeddings classifier

## Installation
1. Clone repository with submodules:
```bash
git clone --recurse-submodules git@github.com:NikolayKozyrskiy/embed_classifier
```
2. Go to the root dir of the project:
```shell
cd embed_classifier
```
3. Run script `scripts/create_env.sh` to create new conda environment named `emb_clsr`, activate it and install all needed dependencies:
```bash
source scripts/create_env.sh
```

[CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset will be downloaded on demand automatically.

## Structure of the project
* The architecture of the autoencoder and classifier are in `classifier/models/*`.
* The pipeline with autoencoder and classifier functions is `classifier/pipeline.py`.
* Losses and metrics are implemented and processed in `classifier/output_dispatcher.py`.
* Main train and validation functions are in `classifier/train.py`.
* Configs are organized as python objects powered by `BaseModel` from `pydantic` package. The main config is implemented in `classifier/config.py`. The particular configs for various experiments are in `classifier/configs/*`
* The data is stored in dir `_data` (created automatically on demand).
* The training logs are in `logs`(created automatically on demand).
* Some useful bash scripts for running experiments are in `scripts/*`.

## Training and validation
All scripts must be run from the project root.
To reproduce the latest results run script:
```bash
sh scripts/train_ae_and_vae.sh
```
After the script finishes, to train classifiers run:
```bash
sh scripts/train_classifier.sh
```
To validate the trained models use the following scripts for autoencoder and classifier correspondingly:
```bash
sh scripts/eval_ae.sh
sh scripts/eval_classifier.sh
```
You need to change the `config_path`, `logdir` and `checkpoint` values in these scripts according to your onw ones. The attached scripts can treated as examples.


The random seeds are fixed so that the experiments are reproducible. 

For baseline architectures I took ligh-weigth neural networks for fast training.
[wandb](https://wandb.ai) is used to log results. On the first run you will be asked your authorization token if wandb is not initialized on your machine.
All training and validation loss and metrics values are logged locally and sent to wandb. During training procedure of autoencoder the gt, reconstructed and sampled images are logged to wandb. 

I also use the wandb functionality to vizualize the embeddings. On the project page on wandb one can choose the dimentionality reduction method on the corresponding panel. Among the available methods are PCA, t-SNE and UMAP. The visualization is only possible in 2-dimensions, i.e. plain image. According to the obtained visualizations the embeddings can't be classified well in 2-dimensional case even with non-linear reduction methods like t-SNE and UMAP.


## Baseline
I used the light-weight vanilla encoder and decoder modules with different latent space size. A small MLP with 1 hidden layer is used as classifier. I trained autoencoders and classifiers on embeddings with 32, 64, 128 and 256 latent space sizes. According to the obtained results the smallest reconstruction error is acheived with the biggest latent space size, no surprizes here. However, the classifier overfits on embeddings of size 256 after several training steps. The classifier on embeddings of size 128 also overfits but much slower. Both classifiers on embeddings of sizes 64 and 32 show quite good convergence and don't overfit. In result, the latent space with size 64 is the most suitable in such task as the classifier shows the best performance in terms of accuracy without overfitting by the end of training procedure. In general, the samples from CIFAR10 dataset have shapes `32x32x3 = 3072` and represent 10 different classes of objects from wild world. The compression ratio is such case will be `3072 / 64 = 48` which seems reasonable for classification task on image embeddings. As a simple autoencoder is trained on basic MSE loss without any regularizations and priors on the distributions, the constructed latent space is unstructed, that is why the reconstructed samples from standard Gaussian noise don't often represent the samples from the training dataset. The generated samples are logged during training procedure and can be viewed on wandb project page.

## Improvements
One of the straight-forward improvements is to use the Variational Autoencoder (VAE) as it is learnt in a way to construct the latent space of given samples with particular prior so that the obtained latent space will have a specific induced structure, if to be short. With such intuition one can try to train classifier on the obtained embeddings of given samples which now have meaningful place in latent space. Moreover, if trained correctly and VAE didn't collapse, the VAE decoder can generate quite similar images to the ones from the dataset using samples from prior distribution (standard Normal in my experiments). For reconstruction loss I used simple MSE, though it might be not the best choice.

I ran the same experiments as for the baseline vanilla autoencoder with almost the same simple architectures of encoder and decoders. The classifier on embeddings produced by VAEs perform better than on simple autoencoder ones. 

Another direction of improvement is to use more powerful and efficient Encoder and Decoder architectures which can extract better features from dataset images in order to be able to construct a better latent space. Ecoder and Decoder with Resnet18 backbones are implemented in the project as examples.

One more possible approach to improve classifier performance is add the Cross-Entropy loss in training VAE which will introduce another corresponding restrictions on the latent space construction.

Among other regularizations on the latent space can be the something like anti-correlation loss, which will force the embeddings of samples of different classes to be uncorrelated or in the best case orthogonal. Such embeddings should be easily classfied even by classical ML algorithms.

## Experiment plots

As mentioned above, all autoencoders converged quite fast without overfitting. I used cosine annealing learning rate scheduler without restarts. To monitor quality of the reconstructed images I used PSNR. I don't add the reconstructed examples here as it's already obvious that with PSNR value in range 20-26 the reconstructed images are meaningfull and quite simialar to original ones. Moreover, they are in extremely low resolution 32x32 so that reconstruction artifacts are smoothed strongly.

![valid_mse](resources/valid_mse.png)
![valid_psnr](resources/valid_psnr.png)

Tha small MPL classifier was able to achieve about 85% accuracy on train dataset on latent space of size 256.

![train_ce](resources/train_ce.png)
![train_acc](resources/train_acc.png)

However as already discussed above this classifier strongly overfits on 256-sized embeddings, especially for simple autoencoder. One can also see that for each pair basic autoencoder - VAE with the same latent space sizes the corresponding VAEs produce better embreddings for further classification.

![valid_ce](resources/valid_ce.png)
![valid_acc](resources/valid_acc.png)

The image below demonstrates the generative ability of different autoencoders:

![sampled_imgs](resources/sampled_imgs.png)

As one can see, all basic autoencoders can't generate meaningfull results from vector sampled from standard Normal distribution. At that time VAE are able to generate some close images to the ones from dataset in some sense. The best illustration is for VAE with 64 latent space size, where one can clearly see the image of the puppy. These images are randomly chosen and are not cherry-picked.