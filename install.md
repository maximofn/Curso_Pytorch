# Instalar misntras se crea el entorno
mamba create -y -n cursopytorch -c anaconda -c conda-forge -c huggingface ipykernel numpy matplotlib gitpython pandas scikit-image scikit-learn=1.1 fastprogress opencv transformers datasets ipywidgets tiktoken
conda activate cursopytorch
mamba install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Crear el entorno e instalar todo
conda create -y -n cursopytorch python=3.8
conda activate cursopytorch
mamba install -y -c anaconda -c conda-forge -c fastai -c huggingface ipykernel numpy matplotlib gitpython pandas scikit-image scikit-learn=1.1 fastprogress opencv transformers datasets ipywidgets tiktoken
mamba install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Crear el entorno e instalar uno a uno
conda create -y -n cursopytorch python=3.8
conda activate cursopytorch
mamba install -y -c anaconda ipykernel
mamba install -y -c anaconda numpy 
mamba install -y -c conda-forge matplotlib
mamba install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
mamba install -y -c conda-forge gitpython
mamba install -y -c anaconda pandas
mamba install -y -c anaconda scikit-image
mamba install -y -c anaconda scikit-learn=1.1
mamba install -y -c fastai fastprogress
mamba install -y -c conda-forge opencv
mamba install -y -c conda-forge transformers
mamba install -y -c huggingface -c conda-forge datasets
mamba install -y -c anaconda ipywidgets
mamba install -y -c conda-forge tiktoken