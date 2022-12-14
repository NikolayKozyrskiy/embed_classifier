conda create -n emb_clsr python=3.9 -y
conda activate emb_clsr
conda install pytorch=1.11.0 torchvision=0.12.0 cudatoolkit=11.3.1 -c pytorch -y
conda install scipy=1.8.1 -c conda-forge -y
conda install black=22.6.0 -c conda-forge -y
conda install typer=0.6.1 -c conda-forge -y
conda install wandb=0.13.5 -c conda-forge -y
conda install dataclasses=0.8 -c conda-forge -y
conda install imageio=2.19.3 -c conda-forge -y
conda install pydantic=1.9.0 -c conda-forge -y
conda install matplotlib=3.5.2 -c conda-forge -y
conda install tensorflow=2.4.1 -c conda-forge -y
conda install tensorboard=2.4.1 -c conda-forge -y
conda install ipython=8.6.0 -c anaconda -y
conda install notebook=6.5.2 -c anaconda -y
conda install pandas=1.4.4 -c anaconda -y
pip3 install ./libs/matches