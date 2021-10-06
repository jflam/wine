# Model used by Dockerfile to pull the model into the Docker image
#%%
import os
import constants as C

from sentence_transformers import SentenceTransformer

def cache_model(model_name):
    """Loads model from Huggingface model hub"""
    try:
        model_path = os.path.expanduser(C.LOCAL_MODEL_PATH)
        if os.path.exists(model_path):
            model = SentenceTransformer(model_path)
        else:
            model = SentenceTransformer(model_name)
            model.save(model_path)
    except Exception as e:
        raise(e)

cache_model(C.SENTENCE_TRANSFORMER_MODEL_NAME)