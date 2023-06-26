mamba create -y -n cursopytorch -c anaconda -c conda-forge -c huggingface ipykernel numpy matplotlib gitpython pandas scikit-image scikit-learn=1.1 fastprogress opencv transformers ipywidgets
conda activate cursopytorch
mamba install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia


conda create -n cursopytorch python=3.8
mamba install -y -c anaconda ipykernel
mamba install -y -c anaconda numpy 
mamba install -y -c conda-forge matplotlib
mamba install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
mamba install -y -c conda-forge gitpython
mamba install -y -c anaconda pandas
mamba install -y -c anaconda scikit-image
mamba install -y -c conda-forge transformers