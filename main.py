from ufc import get_fighter
import json

with open('Alexandre Pantoja.json', 'w') as f:
    json.dump(get_fighter('Alexandre Pantoja'), f)
