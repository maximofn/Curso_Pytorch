# Instalar mientras se crea el entorno
conda install -y -c conda-forge mamba

conda create -y -n cursopytorch -c anaconda -c conda-forge -c huggingface -c fastai ipykernel numpy matplotlib gitpython pandas scikit-image scikit-learn fastprogress opencv transformers onnx datasets ipywidgets sacrebleu nltk python=3.11

conda activate cursopytorch

pip install rouge

mamba install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Crear el entorno e instalar todo
conda install -y -c conda-forge mamba

conda create -y -n cursopytorch python=3.11

conda activate cursopytorch

mamba install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

mamba install -y -c anaconda -c conda-forge -c huggingface -c fastai ipykernel numpy matplotlib gitpython pandas scikit-image scikit-learn fastprogress opencv transformers onnx datasets ipywidgets sacrebleu nltk

pip install rouge

# Crear el entorno e instalar uno a uno
conda install -y -c conda-forge mamba

conda create -y -n cursopytorch python=3.11

conda activate cursopytorch

mamba install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

mamba install -y -c anaconda ipykernel

mamba install -y -c anaconda ipywidgets

mamba install -y -c huggingface -c conda-forge datasets

mamba install -y -c anaconda numpy 

mamba install -y -c conda-forge matplotlib

mamba install -y -c conda-forge gitpython

mamba install -y -c anaconda pandas

mamba install -y -c anaconda scikit-image

mamba install -y -c anaconda scikit-learn

mamba install -y -c fastai fastprogress

mamba install -y -c conda-forge opencv

mamba install -y -c conda-forge transformers

mamba install -y -c conda-forge onnx

mamba install -y -c conda-forge sacrebleu

mamba install -y nltk

pip install rouge