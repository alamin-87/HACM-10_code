PREFIX_TO_CLASS = {
    'CA': 'Calculator',
    'CR': 'Charger',
    'HE': 'Headset',
    'KE': 'Keyboard',
    'MU': 'Mouse',
    'PH': 'Phone',
    'PN': 'Pen',
    'RE': 'Remote',
    'WL': 'Wallet',
    'WT': 'Watch',
}

CLASS_NAMES = ['Calculator', 'Charger', 'Headset', 'Keyboard', 'Mouse',
               'Pen', 'Phone', 'Remote', 'Wallet', 'Watch']

CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


def get_true_label(filename):
    """
    Returns the actual ground-truth class from the filename prefix.
    The photographer knew which object they were shooting, so the filename
    is reliable ground truth. Majority-vote annotator labels are NOT, since
    ~21% of images are degraded enough that annotators misidentify them.
    """
    prefix = filename.split('_')[0]
    if prefix not in PREFIX_TO_CLASS:
        raise ValueError(f"Unknown filename prefix '{prefix}' in {filename}")
    return PREFIX_TO_CLASS[prefix]


def get_true_label_idx(filename):
    return CLASS_TO_IDX[get_true_label(filename)]