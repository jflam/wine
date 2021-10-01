FROM nvcr.io/nvidia/pytorch:21.08-py3

COPY requirements.txt /tmp/requirements.txt
COPY cache_model.py /tmp/cache_model.py
COPY constants.py /tmp/constants.py
WORKDIR /tmp
RUN pip install -v -r requirements.txt
RUN python cache_model.py