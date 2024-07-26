warmup_config = {
    'cifar10': {
        "resnet18": [30 for i in range(10)],
        "cnn": [20 for i in range(10)],
    },
    'pacs': {
        "resnet18": [20 for i in range(4)],
        "cnn": [20 for i in range(4)],
    },
    'oc10': {
        "resnet18": [20 for i in range(4)],
        "cnn": [20 for i in range(4)],
    },
    'tiny': {
        "resnet18": [20 for i in range(10)],
        "cnn": [20 for i in range(10)],
    }
}