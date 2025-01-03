from ufc import get_fighter
import json

with open('John_Jones.json', 'w') as f:
    json.dump(get_fighter('Jon Jones'), f)
