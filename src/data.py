import os

KAGGLE_API_TOKEN = os.getenv('KAGGLE_API_TOKEN')
KAGGLE_KEY = os.getenv('KAGGLE_KEY')

# Existing behavior for handling KAGGLE_KEY
if KAGGLE_KEY is None and KAGGLE_API_TOKEN is not None:
    KAGGLE_KEY = KAGGLE_API_TOKEN

def _kaggle_creds_available():
    username = os.getenv('KAGGLE_USERNAME')
    if username:
        return True
    elif KAGGLE_API_TOKEN:
        global KAGGLE_LAST_ERROR
        KAGGLE_LAST_ERROR = "KAGGLE_USERNAME is missing, but KAGGLE_API_TOKEN is set."
        return False
    return False