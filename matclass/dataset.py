import numpy as np


def _hex_to_rgb(hex_str):
    colorstring = hex_str.strip()
    if colorstring[0] == '#':
        colorstring = colorstring[1:]
    if len(colorstring) != 6:
        raise ValueError("input #%s is not in #RRGGBB format" % colorstring)
    r, g, b = colorstring[:2], colorstring[2:4], colorstring[4:]
    r, g, b = [int(n, 16) for n in (r, g, b)]
    return (r, g, b)


NLABELS = 23

LABEL_COLORS_HEX = [
    '#771111', '#cac690', '#eeeeee', '#7c8fa6', '#597d31', '#104410',
    '#bb819c', '#d0ce48', '#622745', '#666666', '#d54a31', '#101044',
    '#444126', '#75d646', '#dd4348', '#5c8577', '#c78472', '#75d6d0',
    '#5b4586', '#c04393', '#d69948', '#7370d8', '#7a3622', '#000000',
]

LABEL_COLORS = np.array([_hex_to_rgb(h) for h in LABEL_COLORS_HEX], dtype=np.uint8)

COLOR_TUPLE_TO_NETCAT = {
    _hex_to_rgb(h): i for i, h in enumerate(LABEL_COLORS_HEX)
}

NETCAT_TO_COLOR_TUPLE = {
    i: h for (h, i) in COLOR_TUPLE_TO_NETCAT.iteritems()
}

NETCAT_TO_NAME = [
    'brick', 'carpet', 'ceramic', 'fabric', 'foliage', 'food', 'glass', 'hair',
    'leather', 'metal', 'mirror', 'other', 'painted', 'paper', 'plastic',
    'polishedstone', 'skin', 'sky', 'stone', 'tile', 'wallpaper', 'water',
    'wood', 'unknown',
]

NAME_TO_NETCAT = {
    n: i for (i, n) in enumerate(NETCAT_TO_NAME)
}

GROUND_SET_TO_NAME = {
    'bark': 'foliage',
    'bread': 'food',
    'brick': 'brick',
    'carpet': 'carpet',
    'ceramic': 'ceramic',
    'chalkboard': 'other',
    'cork': 'other',
    'fabric': 'fabric',
    'fire': 'other',
    'flower': 'foliage',
    'foliage': 'foliage',
    'food': 'food',
    'fruit': 'food',
    'fur': 'other',
    'glass': 'glass',
    'granite': 'polishedstone',
    'grass': 'foliage',
    'hair': 'hair',
    'laminate': 'other',
    'leaf': 'foliage',
    'leather': 'leather',
    'linoleum': 'other',
    'marble': 'polishedstone',
    'meat': 'food',
    'metal': 'metal',
    'mirror': 'mirror',
    'painted': 'painted',
    'paintedwood': 'painted',
    'paper': 'paper',
    'plastic': 'plastic',
    'polishedstone': 'polishedstone',
    'redbrick': 'brick',
    'rubber': 'other',
    'shingle': 'other',
    'skin': 'skin',
    'sky': 'sky',
    'sponge': 'other',
    'styrofoam': 'other',
    'tile': 'tile',
    'tuftedfabric': 'fabric',
    'unpolishedstone': 'stone',
    'vegetable': 'food',
    'wallpaper': 'wallpaper',
    'water': 'water',
    'wax': 'other',
    'wicker': 'other',
    'wood': 'wood',

    'uncertain': 'unknown',
    'unknown': 'unknown',
}

SUBSTANCE_NAME_TO_GROUND_SET = {
    "i can't tell": 'uncertain',
    "not on list": 'unknown',
    "more than one material": 'unknown',

    'brick': 'brick',
    'brick - black brick': 'brick',
    'brick - blue brick': 'brick',
    'brick - gray brick': 'brick',
    'brick - red brick': 'redbrick',
    'brick - white brick': 'brick',

    'ceramic': 'ceramic',
    'ceramic/porcelain': 'ceramic',
    'porcelain': 'ceramic',

    'chalkboard/blackboard': 'chalkboard',

    'cork': 'cork',
    'cork/corkboard': 'cork',

    'fabric': 'fabric',
    'carpet/rug': 'carpet',
    'fabric - carpet/rug': 'carpet',
    'rug - damask': 'carpet',
    'fabric - chenille': 'fabric',
    'fabric - cotton': 'fabric',
    'fabric - damask': 'fabric',
    'fabric - felt': 'fabric',
    'fabric - fur/hide': 'fur',
    'fabric - gabordine': 'fabric',
    'fabric - knitted': 'fabric',
    'fabric - microfiber': 'fabric',
    'fabric - quilted': 'fabric',
    'fabric - satin': 'fabric',
    'fabric - silk': 'fabric',
    'fabric - tapestry': 'fabric',
    'fabric - tufted': 'tuftedfabric',
    'fabric - tweed': 'fabric',
    'fabric - velvet': 'fabric',
    'fabric - wool': 'fabric',

    'fire': 'fire',

    'foliage': 'foliage',
    'foliage - bark': 'bark',
    'foliage - cactus': 'foliage',
    'foliage - flower': 'flower',
    'foliage - grass': 'grass',
    'foliage - leaf/stem': 'leaf',
    'foliage - leaves': 'leaf',
    'foliage - moss': 'foliage',
    'foliage - pine': 'foliage',

    'food': 'food',
    'food - bread': 'bread',
    'food - fruit': 'fruit',
    'food - meat/fish': 'meat',
    'food - vegetable': 'vegetable',

    'fur': 'fur',
    'fur/hide (animal)': 'fur',
    'fur/hide': 'fur',
    'animal fur': 'fur',
    'sheepskin': 'fur',
    'animal - fur/hide': 'fur',
    'skin - animal': 'fur',
    'hide': 'fur',
    'animal hide': 'fur',
    'cowhide': 'fur',
    'zebra skin': 'fur',

    'glass': 'glass',
    'glass - frosted': 'glass',
    'glass - recycled': 'glass',
    'glass - stained': 'glass',
    'mirror': 'mirror',
    'glass - mirror': 'mirror',

    'hair (human)': 'hair',
    'human - hair': 'hair',
    'hair - human': 'hair',

    'laminate': 'laminate',

    'leather': 'leather',
    'leather - tufted': 'leather',
    'leather - worn': 'leather',

    'metal (unpainted)': 'metal',
    'metal - aluminum': 'metal',
    'metal - brass': 'metal',
    'metal - bronze': 'metal',
    'metal - chrome': 'metal',
    'metal - copper': 'metal',
    'metal - gold': 'metal',
    'metal - iron': 'metal',
    'metal - nickel': 'metal',
    'metal - rusted': 'metal',
    'metal - silver': 'metal',
    'metal - stainless steel': 'metal',
    'metal - unpainted': 'metal',
    'metal - zinc': 'metal',

    'painted': 'painted',
    'brick - painted': 'painted',
    'metal (painted)': 'painted',
    'metal - painted': 'painted',
    'siding - painted': 'painted',
    'wood - painted': 'paintedwood',

    'paper': 'paper',
    'paper/tissue/cardboard': 'paper',
    'paper tissue': 'paper',
    'paper/tissue': 'paper',
    'paper towel/tissue': 'paper',
    'cardboard': 'paper',
    'paper tissue': 'paper',

    'plastic': 'plastic',
    'plastic - clear': 'plastic',
    'plastic - opaque': 'plastic',
    'resin': 'plastic',

    'rubber/latex': 'rubber',

    'shingles': 'shingle',

    'skin (human)': 'skin',
    'human - skin': 'skin',
    'skin - human': 'skin',

    'sky': 'sky',

    'sponge': 'sponge',

    'granite/marble/quartz/etc.': 'polishedstone',
    'concrete - polished': 'polishedstone',
    'stone/concrete - polished': 'polishedstone',
    'stone - engineered quartz': 'polishedstone',
    'stone - limestone': 'polishedstone',
    'stone - onyx': 'polishedstone',
    'stone - quartzite': 'polishedstone',
    'stone - slate': 'polishedstone',
    'stone - soapstone': 'polishedstone',
    'stone - solid surface': 'polishedstone',
    'stone - travertine': 'polishedstone',
    'stone - granite': 'granite',
    'stone - marble': 'marble',

    'stone/concrete (unpolished)': 'unpolishedstone',
    'stone/concrete - unpolished': 'unpolishedstone',
    'concrete': 'unpolishedstone',
    'stone - gravel': 'unpolishedstone',
    'stone - sand': 'unpolishedstone',
    'stone - stacked': 'unpolishedstone',

    'styrofoam': 'styrofoam',

    'linoleum': 'linoleum',

    'tile': 'tile',
    'tile - ceramic/porcelain': 'tile',
    'tile - glass': 'tile',
    'tile - marble': 'tile',
    'tile - metal': 'tile',
    'tile - mosaic': 'tile',
    'tile - mosaic, subway': 'tile',
    'tile - porcelain': 'tile',
    'tile - quartzite': 'tile',
    'tile - slate': 'tile',
    'tile - subway': 'tile',
    'tile - terra-cotta': 'tile',
    'tile - travertine': 'tile',

    'wallpaper': 'wallpaper',

    'water (liquid)': 'water',
    'water': 'water',

    'wax': 'wax',

    'wicker': 'wicker',

    'wood (unpainted)': 'wood',
    'wood - unpainted': 'wood',
    'shingles - wood': 'shingle',
    'wood - ash': 'wood',
    'wood - aspen': 'wood',
    'wood - balsa': 'wood',
    'wood - bamboo': 'wood',
    'wood - basswood': 'wood',
    'wood - beech': 'wood',
    'wood - birch': 'wood',
    'wood - black walnut': 'wood',
    'wood - blackwood': 'wood',
    'wood - boxwood': 'wood',
    'wood - cedar': 'wood',
    'wood - cherry': 'wood',
    'wood - chestnut': 'wood',
    'wood - cottonwood': 'wood',
    'wood - cypress': 'wood',
    'wood - dark hardwood': 'wood',
    'wood - ebony': 'wood',
    'wood - elm': 'wood',
    'wood - fir': 'wood',
    'wood - gum': 'wood',
    'wood - hardwood': 'wood',
    'wood - hemlock': 'wood',
    'wood - hickory': 'wood',
    'wood - holly': 'wood',
    'wood - hornbeam': 'wood',
    'wood - juniper': 'wood',
    'wood - larch': 'wood',
    'wood - light hardwood': 'wood',
    'wood - lymba': 'wood',
    'wood - magnolia': 'wood',
    'wood - mahogany': 'wood',
    'wood - maple': 'wood',
    'wood - medium hardwood': 'wood',
    'wood - meranti': 'wood',
    'wood - mesquite': 'wood',
    'wood - myrtle': 'wood',
    'wood - natural color': 'wood',
    'wood - oak': 'wood',
    'wood - pine': 'wood',
    'wood - plywood': 'wood',
    'wood - poplar': 'wood',
    'wood - rosewood': 'wood',
    'wood - sassafras': 'wood',
    'wood - satinwood': 'wood',
    'wood - spruce': 'wood',
    'wood - sycamore': 'wood',
    'wood - teak': 'wood',
    'wood - tigerwood': 'wood',
    'wood - walnut': 'wood',
}

SUBSTANCE_NAME_TO_NETCAT = {
    n: NAME_TO_NETCAT[GROUND_SET_TO_NAME[SUBSTANCE_NAME_TO_GROUND_SET[n]]]
    for n in SUBSTANCE_NAME_TO_GROUND_SET
}

UNKNOWN_LABEL = NAME_TO_NETCAT['unknown']

# sanity checks
assert len(COLOR_TUPLE_TO_NETCAT) == NLABELS + 1
assert len(set(SUBSTANCE_NAME_TO_NETCAT.values())) == NLABELS + 1
assert len(COLOR_TUPLE_TO_NETCAT) == NLABELS + 1
assert len(NETCAT_TO_NAME) == NLABELS + 1
assert len(NAME_TO_NETCAT) == NLABELS + 1
assert UNKNOWN_LABEL == NLABELS
