import unicodedata

def normalize_city(city):
    # Remove accents
    city = unicodedata.normalize('NFKD', city).encode('ASCII', 'ignore').decode()
    # Lowercase, strip, collapse spaces
    return ' '.join(city.lower().strip().split())

from collections import defaultdict, Counter

def deduplicate_cities(city_list):
    norm_map = defaultdict(list)

    for city in city_list:
        norm = normalize_city(city)
        norm_map[norm].append(city)

    # Choose canonical version (most frequent or shortest)
    canonical_map = {
        norm: Counter(names).most_common(1)[0][0]
        for norm, names in norm_map.items()
    }

    # Return cleaned list with unique canonical names
    return list(canonical_map.values()), canonical_map

import pickle

# Load cities
with open("cities.pkl", "rb") as f:
    raw_cities = pickle.load(f)

# Clean and deduplicate
unique_cities, canonical_mapping = deduplicate_cities(raw_cities)

# Save cleaned list
with open("cleaned_cities.pkl", "wb") as f:
    pickle.dump(unique_cities, f)

print(f"Original count: {len(raw_cities)}")
print(f"Unique count: {len(unique_cities)}")
