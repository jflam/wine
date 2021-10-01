# Model used by Dockerfile to pull the model into the Docker image
#%%
import os
import constants as C

from sentence_transformers import SentenceTransformer

def cache_model(model_name):
    """Loads model from Huggingface model hub"""
    try:
        if os.path.exists(C.LOCAL_MODEL_PATH):
            model = SentenceTransformer(C.LOCAL_MODEL_PATH)
        else:
            model = SentenceTransformer(model_name)
            model.save('./model')
    except Exception as e:
        raise(e)

cache_model(C.SENTENCE_TRANSFORMER_MODEL_NAME)