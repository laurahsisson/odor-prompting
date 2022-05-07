# python3 -m charts.descriptor_cooccur
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns

import util.read_dataset as dataset
INPUT_FILE = "all_lemmatized"
dataset.read_all_words(INPUT_FILE)

drv_words = [
    'citrus', 'lemon', 'grapefruit', 'orange', 'fruity', 'pineapple', 'grape',
    'strawberry', 'apple', 'pear', 'cantaloupe', 'peach', 'banana', 'floral',
    'rose', 'violets', 'lavender', 'cologne', 'musk', 'perfumery', 'fragrant',
    'aromatic', 'honey', 'cherry', 'almond', 'acetone', 'nutty', 'spicy',
    'clove', 'cinnamon', 'laurel', 'tea', 'seasoning', 'pepper', 'peppers',
    'dill', 'caraway', 'cognac', 'resinous', 'cedarwood', 'mothballs',
    'peppermint', 'camphor', 'eucalyptus', 'chocolate', 'vanilla', 'sweet',
    'maple', 'caramel', 'malty', 'raisins', 'molasses', 'coconut', 'anise',
    'alcoholic', 'gasoline', 'turpentine', 'geranium', 'celery', 'weeds',
    'grass', 'herbal', 'cucumber', 'hay', 'grainy', 'yeasty', 'bakery',
    'fermented', 'beery', 'soapy', 'leather', 'cardboard', 'rope', 'stale',
    'musty', 'potatoes', 'mouse', 'mushroom', 'peanuts', 'beany', 'eggy',
    'bark', 'cork', 'smoky', 'incense', 'coffee', 'tar', 'creosote',
    'carbolic', 'medicinal', 'chemical', 'bitter', 'pungent', 'vinegar',
    'sauerkraut', 'ammonia', 'urine', 'urine', 'fishy', 'kipper', 'semen',
    'rubber', 'sooty', 'kerosene', 'oily', 'buttery', 'paint', 'varnish',
    'popcorn', 'meaty', 'soupy', 'rancid', 'sweaty', 'cheesy', 'methane',
    'sulfurous', 'garlic', 'metallic', 'blood', 'animal', 'sewer', 'putrid',
    'fecal', 'cadaverous', 'sickening', 'powdery', 'chalky', 'light', 'heavy',
    'cooling', 'warm'
]
drm_words = [
    'bakery', 'sweet', 'fruit', 'fish', 'garlic', 'spices', 'cold', 'sour',
    'burnt', 'acid', 'warm', 'musky', 'sweaty', 'ammonia', 'decayed', 'wood',
    'grass', 'flower', 'chemical'
]
combined_words = drv_words + drm_words
no_occurs = [w for w in combined_words if dataset.occurs[w] == 0]
print("These words never occured")
print(no_occurs)

jaccard_matrix = []

for w1 in drm_words:
    js = []
    for w2 in drv_words:
        wp = dataset.get_word_pair(w1, w2)
        w1unionw2 = (dataset.occurs[w1] + dataset.occurs[w2] - dataset.cooccur[wp])
        jaccard = 0
        if w1unionw2 != 0:
            jaccard = dataset.cooccur[wp] / w1unionw2
        js.append(jaccard)
        if jaccard == 1:
            print(f"\"{w1}\"")
    jaccard_matrix.append(js)

# Raising jaccard matrix to ^2 so that it is more visually discernible.
sns.heatmap(np.power(jaccard_matrix, 1 / 2),
            cmap="CMRmap_r",
            yticklabels=drm_words)
plt.savefig("figures/descriptor_cooccur.png")
plt.savefig("figures/descriptor_cooccur.svg")