import requests

url = 'http://localhost:5000/ap'
long_text = """
It’s 6:12 am and near me I hear my alarm clock going off playing a heavy metal song called “In Waves,” I quickly get up to turn off my alarm and start getting ready for school, as I walk out of the room I make my way down to the restroom to brush my teeth and change into my outfit for the day, once out I go down the hallway to grab my contacts from a table, I grab my contacts and go back down to the hallway where there is a mirror that I use to help me apply my contacts, once done I use the same mirror to help me apply my makeup, then I head down to the kitchen and go to the fridge to get me a few snacks for school, after I grab my bag and get out of the house to make my way to school.
"""
# long_text = """
# Mamma mia
#  """
data = {
    'long_text': long_text
}

response = requests.post(url, json=data)

if response.status_code == 200:
    result = response.json()
    summary = result.get('summary', '')
    print("Summary:", summary)
else:
    print("Error:", response.text)
