
FROM nvcr.io/nvidia/pytorch:21.08-py3

RUN useradd -r -u 1000 -m -d /home/ezuser ezuser
USER ezuser
WORKDIR /home/ezuser/tmp

COPY requirements.txt .
COPY constants.py .
COPY cache_model.py .
RUN pip install -v -r requirements.txt
RUN python cache_model.py
