optimization_method = "none"
mu = 1
hyperparameters_one_shot = {
    "cifar10": [
        {
            "lr": 5e-5,
            # "lr": 0.001,
            "local_ep": 200,
            "transform": None,
            "mu": mu,
            "optimization_method": optimization_method,
            "optimizer": "adam",
            "weight_decay": 1e-4,
            "alpha": 0.01,
            "beta": 0.01,
        },
    ],
    "pacs": [
        {
            "lr": 5e-5,
            "local_ep": 200,
            "transform": None,
            "mu": mu,
            "optimization_method": optimization_method,
            "optimizer": "adam",
            "weight_decay": 1e-4,
        },
    ],
    "oc10": [
    {
        "lr": 0.001,
        "local_ep": 100,
        "transform": None,
        "mu": mu,
        "optimization_method": optimization_method,
        "optimizer": "adam",
        "weight_decay": 1e-4,
    }],
    "tiny": [
    {
        "lr": 5e-5,
        "local_ep": 100,
        "transform": None,
        "mu": mu,
        "optimization_method": optimization_method,
        "optimizer": "adam",
        "weight_decay": 1e-4,
    }]
}

hyperparameters = hyperparameters_one_shot