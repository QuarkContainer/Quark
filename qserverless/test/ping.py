import requests

# The API endpoint
#url = "https://jsonplaceholder.typicode.com/posts/1"
#url = "http://localhost:1234/?namespace=ns1"
url = "http://172.64.129.35/posts/1"

# A GET request to the API
response = requests.get(url)

# Print the response
#response_json = response.json()
print(response)