import requests
import os
import json


# Specify a URL that resolves to your workspace
# URL = 'http://127.0.0.1'
URL = 'http://0.0.0.0'

# Call each API endpoint and store the responses
# response1 = requests.post(URL + ':8000/prediction?filename=./testdata/testdata.csv').content  # returns list predictions
response1 = requests.post(URL + ':8000/inference').content  # returns list predictions

print(response1)
#response1 = eval(response1.decode('ascii'))

# pop status from all responses
response1.pop('status')

print('response_1: {}'.format(response1))

# combine all API responses
#responses = response1 | response2 | response3 | response4
#print('combined responses = {}'.format(responses))

# write the responses to your workspace
# full_output_path = os.path.join(output_path, 'apireturns.txt')
# with open(full_output_path, 'w') as file:
#     file.write(json.dumps(responses))
# print('OK - apicalls completed')
