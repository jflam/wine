
FROM nvcr.io/nvidia/pytorch:21.08-py3

COPY requirements.txt /tmp/requirements.txt
COPY constants.py /tmp/constants.py
COPY cache_model.py /tmp/cache_model.py

WORKDIR /tmp
RUN pip install -v -r requirements.txt
RUN python cache_model.py

# COPY .ssh/config /root/.ssh/config
# COPY .ssh/id_rsa_github /root/.ssh/id_rsa_github
# COPY .ssh/id_rsa_github.pub /root/.ssh/id_rsa_github.pub
