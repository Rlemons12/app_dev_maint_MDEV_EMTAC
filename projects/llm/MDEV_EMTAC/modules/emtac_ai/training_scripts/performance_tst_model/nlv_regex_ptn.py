# Natural language variations for each regex pattern type

NATURAL_LANGUAGE_VARIATIONS = {

    # ITEMNUM Pattern Variations
    'ITEMNUM': {
        'formal': [
            "part number {itemnum}",
            "item {itemnum}",
            "catalog number {itemnum}",
            "stock number {itemnum}",
            "SKU {itemnum}"
        ],
        'casual': [
            "part {itemnum}",
            "item# {itemnum}",
            "number {itemnum}",
            "code {itemnum}",
            "{itemnum}"
        ],
        'contextual': [
            "the {itemnum} part",
            "anything with {itemnum}",
            "that {itemnum} item",
            "part called {itemnum}",
            "item labeled {itemnum}"
        ]
    },

    # DESCRIPTION Pattern Variations
    'DESCRIPTION': {
        'formal': [
            "{description}",
            "a {description}",
            "the {description}",
            "{description} component",
            "{description} part"
        ],
        'casual': [
            "some {description}",
            "any {description}",
            "that {description}",
            "{description} thing",
            "one of those {description}"
        ],
        'contextual': [
            "something like a {description}",
            "type of {description}",
            "kind of {description}",
            "{description} or similar",
            "{description} style part"
        ]
    },

    # OEMMFG Pattern Variations
    'OEMMFG': {
        'formal': [
            "manufactured by {manufacturer}",
            "made by {manufacturer}",
            "from {manufacturer}",
            "{manufacturer} brand",
            "{manufacturer} manufactured"
        ],
        'casual': [
            "{manufacturer} stuff",
            "{manufacturer} parts",
            "any {manufacturer}",
            "{manufacturer} made",
            "some {manufacturer}"
        ],
        'contextual': [
            "something from {manufacturer}",
            "anything {manufacturer} makes",
            "whatever {manufacturer} has",
            "that {manufacturer} brand",
            "{manufacturer} or equivalent"
        ]
    },

    # MODEL Pattern Variations
    'MODEL': {
        'formal': [
            "model {model}",
            "model number {model}",
            "part model {model}",
            "manufacturer model {model}",
            "OEM model {model}"
        ],
        'casual': [
            "model {model}",
            "{model} model",
            "the {model}",
            "{model} version",
            "type {model}"
        ],
        'contextual': [
            "something like model {model}",
            "similar to {model}",
            "{model} or equivalent",
            "compatible with {model}",
            "replaces {model}"
        ]
    }
}

# Combined query templates with natural language variations
ENHANCED_QUERY_TEMPLATES = [

    # Single entity queries (15)
    "I need {itemnum_formal}",
    "Do you have {description_formal}?",
    "I'm looking for {manufacturer_formal}",
    "Can I get {model_formal}?",
    "Do you stock {itemnum_casual}?",
    "I need {description_casual}",
    "Looking for {manufacturer_casual}",
    "Can you find {model_casual}?",
    "Do you carry {itemnum_contextual}?",
    "I'm searching for {description_contextual}",
    "Need {manufacturer_contextual}",
    "Any {model_contextual} available?",
    "I require {itemnum_formal}",
    "Show me {description_formal}",
    "Find {manufacturer_formal}",

    # Two entity combinations (15)
    "I need {description_formal} from {manufacturer_formal}",
    "Do you have {description_casual} by {manufacturer_casual}?",
    "I'm looking for {manufacturer_formal} {description_formal}",
    "Can I get {description_formal} model {model_formal}?",
    "Do you stock {itemnum_formal} from {manufacturer_formal}?",
    "I need {manufacturer_casual} {description_casual}",
    "Looking for {description_contextual} made by {manufacturer_contextual}",
    "Can you find {model_formal} by {manufacturer_formal}?",
    "Do you carry {description_formal}, {manufacturer_casual} brand?",
    "I'm searching for {itemnum_casual} or {model_casual}",
    "Need {description_casual} from {manufacturer_formal}",
    "Any {manufacturer_contextual} {description_contextual} available?",
    "I require {model_formal} from {manufacturer_formal}",
    "Show me {description_formal}, model {model_casual}",
    "Find {itemnum_formal} made by {manufacturer_formal}",

    # Three entity combinations (15)
    "I need {description_formal} from {manufacturer_formal}, model {model_formal}",
    "Do you have {manufacturer_casual} {description_casual} model {model_casual}?",
    "I'm looking for {itemnum_formal}, {description_formal} from {manufacturer_formal}",
    "Can I get {description_contextual} by {manufacturer_contextual}, model {model_contextual}?",
    "Do you stock {itemnum_casual} which is {description_casual} from {manufacturer_casual}?",
    "I need {manufacturer_formal} {description_formal}, part number {itemnum_formal}",
    "Looking for {description_casual} made by {manufacturer_casual}, model {model_casual}",
    "Can you find {itemnum_contextual}, that's the {description_contextual} from {manufacturer_contextual}?",
    "Do you carry {manufacturer_formal} part {itemnum_formal}, the {description_formal}?",
    "I'm searching for {model_formal} by {manufacturer_formal}, it's a {description_formal}",
    "Need {description_casual} from {manufacturer_contextual}, model {model_formal}",
    "Any {manufacturer_casual} {description_contextual} model {model_casual} available?",
    "I require {itemnum_formal}, {description_formal} manufactured by {manufacturer_formal}",
    "Show me {manufacturer_formal} model {model_formal}, the {description_casual}",
    "Find {description_formal} part {itemnum_formal} from {manufacturer_contextual}"
]


def generate_enhanced_query(row: pd.Series, template: str) -> Tuple[str, Dict]:
    """Generate enhanced query with natural language variations."""

    # Extract base values
    description = str(row.get('DESCRIPTION', '')).strip().lower()
    manufacturer = str(row.get('OEMMFG', '')).strip().lower()
    itemnum = str(row.get('ITEMNUM', '')).strip()
    model = str(row.get('MODEL', '')).strip()

    # Create variation dictionaries
    variations = {}

    # Add itemnum variations
    for style in ['formal', 'casual', 'contextual']:
        for variation in NATURAL_LANGUAGE_VARIATIONS['ITEMNUM'][style]:
            variations[f'itemnum_{style}'] = variation.format(itemnum=itemnum)

    # Add description variations
    for style in ['formal', 'casual', 'contextual']:
        for variation in NATURAL_LANGUAGE_VARIATIONS['DESCRIPTION'][style]:
            variations[f'description_{style}'] = variation.format(description=description)

    # Add manufacturer variations
    for style in ['formal', 'casual', 'contextual']:
        for variation in NATURAL_LANGUAGE_VARIATIONS['OEMMFG'][style]:
            variations[f'manufacturer_{style}'] = variation.format(manufacturer=manufacturer)

    # Add model variations
    for style in ['formal', 'casual', 'contextual']:
        for variation in NATURAL_LANGUAGE_VARIATIONS['MODEL'][style]:
            variations[f'model_{style}'] = variation.format(model=model)

    # Generate query from template
    query = template.format(**variations)

    # Determine expected entities based on template content
    expected = {}
    if 'itemnum' in template:
        expected['PART_NUMBER'] = itemnum
    if 'description' in template:
        expected['PART_NAME'] = description.upper()
    if 'manufacturer' in template:
        expected['MANUFACTURER'] = manufacturer.upper()
    if 'model' in template:
        expected['MODEL'] = model

    return query, expected

# Example generated queries:
# "I need part number A101576"
# "Do you have some filter tube by balston filt?"
# "I'm looking for something from parker, model p123-456"
# "Can I get hydraulic pump manufactured by emerson, part number A101576?"
# "Do you stock the A101576 part which is filter tube from balston filt?"