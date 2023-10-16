import json
import boto3
from sms_spam_classifier_utilities import one_hot_encode,vectorize_sequences
import os
from email.parser import Parser
import numpy as np
    
sm_client = boto3.client('sagemaker-runtime')
ses_client = boto3.client('ses')
s3_client=boto3.client('s3')

def lambda_handler(event, context):
    #print(event)
    s3=event["Records"][0]["s3"]
    bucket_name=s3["bucket"]["name"]
    key=s3["object"]["key"]
    obj = s3_client.get_object(Bucket=bucket_name, Key=key)
    #print(obj)
    
    origin_email = obj['Body'].read().decode('UTF-8')
    parser = Parser()
    email_str = parser.parsestr(origin_email)
    
    print(email_str)

    from_addr=email_str.get('From')
    # print(from_addr)
    to_addr=email_str.get('To')
    # print(to_addr)

    body=""
    for part in email_str.walk():
        if part.get_content_type()=='text/plain':
            body=part.get_payload(decode=True).decode("utf-8") 
    # print(body)
    body_msg=[body.replace("\n",'')]
    # print(body)
    date_time = email_str.get('Date')
    subject = email_str.get('Subject')
    # print(date_time)
    # print(subject)
  
    vocabulary_length = 9013
    one_hot_data = one_hot_encode(body_msg, vocabulary_length)
    encoded_messages = vectorize_sequences(one_hot_data, vocabulary_length)
    # print(encoded_messages)

    body_js=json.dumps(encoded_messages.tolist())
    request=sm_client.invoke_endpoint(EndpointName=os.environ['Endpoint'],Body=body_js,ContentType='application/json')
    response=json.loads(request["Body"].read().decode('utf-8'))
    print("response: ", response)
    
    pred_label=response['predicted_label'][0][0]
    probability=response['predicted_probability'][0][0]*100
    print(pred_label)
    print(probability)
    
    if(len(body) > 240):
        body = body[0:240]
    
    if(pred_label == 1):
        pred_class= 'SPAM'
    else: # pred_label == 0
        pred_class= 'HAM'
        probability = 100-probability # the original probability means the probability that the email is a spam
            
    
    data = 'We recieved your email at ' + date_time + ' with the subject ' + subject + '.\n'
    data += 'Here is a 240 character sample of the email body: ' + body + '\n'
    data += 'The email was categorized as ' + pred_class+ ' with a ' + str(probability) + '% confidence.\n'    
    
    response = ses_client.send_email(
        Destination={
            'ToAddresses': [
                from_addr
            ],
        },
        Message={
            'Body': {
                'Text': {
                    'Charset': 'UTF-8',
                    'Data': data,
                },
            },
            'Subject': {
                'Charset': 'UTF-8',
                'Data': 'SMS Spam Classifier',
            },
        },
        Source=to_addr,
    )
    
    print(response)    
