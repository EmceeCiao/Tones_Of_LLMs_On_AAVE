# Backend Instructions
You will need to create a service account token through Firebase and download the json file. Rename the json file to `firebase_service_acc.json`

## Start the API service: 

In the terminal, run `uvicorn backend.main:app --reload`


## API Documentation: 
http://127.0.0.1:8000/docs#

## Run tests:  

`pytest tests/test_api.py && pytest tests/test_db.py`
