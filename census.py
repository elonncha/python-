import json, os, requests,urllib, ssl
import numpy as np
os.getcwd()
#### Example: Github API Calls
urlprefix = 'https://api.github.com/search/repositories?'
API_KEY = 'YOUR_API_KEY_GOES_HERE'
# get response using 'request' package
resp = requests.get(url)
outcome = resp.json()
# get response using 'urllib' ,'json', and 'ssl' packages
query = urllib.parse.urlencode({'q':'language:python','sort':'stars'})
url = urlprefix + query
gcontext = ssl.SSLContext()
gresp = urllib.request.urlopen(url, context=gcontext)
rjson = json.loads(gresp.read())

######## make census.gov api calls
# total variable list
variable_url = 'https://api.census.gov/data/2018/acs/acs5/variables.json'
resp = requests.get(variable_url)
variables = resp.json()    ## a full list of variables and descriptions
#create query
var_list = ['B19001B_014E', 'B19101A_004E']
var_string = ''
for var in var_list:
    var_string += var + ','
var_string = var_string[:-1]
# make api call
census_prefix = "https://api.census.gov/data/2018/acs/acs5?"
census_query = urllib.parse.urlencode({'get':'NAME,'+ var_string, 'for':'tract:*', 'in':'state:13', 'key':API_KEY})
census_url = census_prefix + census_query
gresp = urllib.request.urlopen(census_url, context = gcontext)
rjson = json.loads(gresp.read())
table = np.array(rjson)
