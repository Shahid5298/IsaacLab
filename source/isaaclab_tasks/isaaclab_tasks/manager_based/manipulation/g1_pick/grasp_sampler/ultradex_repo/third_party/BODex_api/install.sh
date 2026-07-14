#!/bin/bash

# conda create -n bodex_api python=3.10 -y
# source $HOME_PATH/software/miniconda3/etc/profile.d/conda.sh; conda activate bodex_api

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install -e . --no-build-isolation
pip install trimesh scipy==1.14.1 usd-core
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu118.html
conda install coal -c conda-forge -y
# conda install boost=1.84.0
cd src/bodex/geom/cpp; python setup.py install
pip install numpy==1.25.1 plyfile ninja

# bash _isaac_sim_410/python.sh -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.2+cu118.html
# bash _isaac_sim_410/python.sh -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu118.html
