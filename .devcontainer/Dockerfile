FROM nvcr.io/nvidia/pytorch:21.08-py3

RUN pip install -U transformers==4.10.2 \
                   sentence-transformers==2.0.0 \
                   pandas==1.3.2 \
                   jupyter-client==6.1.12 \
                   jupyter-core==4.7.1 \
                   ipywidgets==7.6.4

RUN pip install --no-binary :all: nmslib==2.1.1