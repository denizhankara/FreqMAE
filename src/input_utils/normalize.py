all_value_ranges = {
    "RealWorld_HAR": {
        "time": {
            "acc": 19.60910987854004,
            "gyr": 10.25952434539795,
            "mag": 222.63572692871094,
            "lig": 58753.0,
        },
        "frequency": {
            "acc": 980.4555053710938,
            "gyr": 498.501708984375,
            "mag": 5807.8935546875,
            "lig": 2937650.0,
        },
        "feature": {
            "DeepSense": {
                "head": 51.56364059448242,
                "chest": 45.52693176269531,
                "upperarm": 31.24556541442871,
                "waist": 29.469738006591797,
                "shin": 20.929248809814453,
            },
            "ResNet": {
                "head": 68.58202362060547,
                "chest": 47.05370330810547,
                "upperarm": 33.268001556396484,
                "waist": 35.740535736083984,
                "shin": 28.322349548339844,
            },
            "Transformer": {
                "head": 5.076425552368164,
                "chest": 4.651660919189453,
                "upperarm": 4.930847644805908,
                "waist": 5.0765061378479,
                "shin": 5.074939250946045,
            },
        },
    },
    "Parkland": {
        "time": {
            "audio": 44778.1953125,
            "seismic": 71805.0,
        },
        "frequency": {
            "audio": 1023106.0,
            "seismic": 14450094.0,
        },
        "feature": {
            "DeepSense": {
                "shake": 28.01055145263672,
            },
            "ResNet": {
                "shake": 143.04722595214844,
            },
            "Transformer": {
                "shake": 5.549056053161621,
            },
        },
    },
    "Parkland1107": {
        "time": {
            "audio": 44778.1953125,
            "seismic": 71805.0,
        },
        "frequency": {
            "audio": 1023106.0,
            "seismic": 14450094.0,
        },
    },
    "ACIDS": {
        "time": {
            "audio": 32768.0,
            "seismic": 32768.0,
        },
    },
    "WESAD": {
        "time": {
            "EMG": 0.45025634765625,
            "EDA": 11.130523681640625,
            "Resp": 37.88604736328125,
        },
        "frequency": {
            "EMG": 3.3664937019348145,
            "EDA": 1552.31787109375,
            "Resp": 5198.353515625,
        },
        "feature": {
            "DeepSense": {
                "chest": 11.82695484161377,
            },
            "ResNet": {
                "chest": 21.00111961364746,
            },
            "Transformer": {
                "chest": 4.362126350402832,
            },
        },
    },
}


def normalize_input(loc_inputs, args, level="time"):
    """Normalize the data between [-1, 1]"""
    normed_loc_inuts = {}

    if level == "feature":
        for loc in loc_inputs:
            max_abs = all_value_ranges[args.dataset][level][args.model][loc]
            normed_loc_inuts[loc] = loc_inputs[loc] / max_abs
    else:

        for loc in loc_inputs:
            normed_loc_inuts[loc] = {}
            for mod in loc_inputs[loc]:
                max_abs = all_value_ranges[args.dataset][level][mod]
                normed_loc_inuts[loc][mod] = loc_inputs[loc][mod] / max_abs

    return normed_loc_inuts
