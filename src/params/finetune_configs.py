"""
Store the finetune configs here for the consistency between test and finetune.
"""

datasets = [
    # "ACIDS",
    # "Parkland",
    "RealWorld_HAR",
    # "PAMAP2",
]

models = [
    "TransformerV4",
    # "DeepSense",
]

learn_frameworks = [
    # "SimCLR",
    # "MoCo",
    # "CMC",
    # "CMCV2",
    "MAE",
    # "Cosmo",
    # "Cocoa",
    # "MTSS",
    # "TS2Vec",
    # "GMC",
    # "TNC",
    # "TSTCC",
]

tasks = {
    "ACIDS": [
        "vehicle_classification",
    ],
    "Parkland": [
        # "vehicle_classification",
        # "distance_classification",
        # "speed_classification",
        "vehicle_classification_diff_loc",
        "vehicle_classification_diff_car",
        "vehicle_classification_diff_env",
    ],
    "RealWorld_HAR": [
        "activity_classification",
    ],
    "PAMAP2": [
        "activity_classification",
    ],
}

label_ratios = {
    "finetune": [
        1.0,
        # 0.1,
        # 0.01,
    ],
    "knn": [
        1.0,
        0.1,
        0.01,
    ],
    "cluster": [
        1.0,
    ],
    "tsne": [
        1.0,
    ],
}

runs = {"finetune": 5, "knn": 5, "cluster": 1, "tsne": 1}
