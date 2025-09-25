A simple ml model lifecycle project for the ecommerce dataset given by bigspark

Breaks down ml model lifecyle from train to deployment

•	Code Structure: The project is a single script model_train_test_serve.py that houses loading the data, feature engineering and training the model. This also houses serializing the model using pickle ready to be used for inference.

•	Deployment: The model is deployed then loaded from the filesystem and with inference post data (that is one-hot encoded) a prediction is made. I have created a simple FLASK REST API endpoint (‘/predict_fraud’) that takes new order data as input and returns a fraud prediction. This can then be served to a live application.

Served from: http://127.0.0.1:5000/predict_fraud

Example JSON body: {'order_value': 1500, 'country_code': 'GB', 'order_date': '2025-05-15'}
