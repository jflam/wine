FROM nvcr.io/nvidia/pytorch:21.08-py3

COPY requirements.txt /tmp/requirements.txt
WORKDIR /tmp
RUN pip install -v -r requirements.txt
RUN pip install --no-binary :all: nmslib