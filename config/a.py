CONFIG = {
    "config": {
        "aaa": 1,
        "bbb": 2,
        "ccc": '1',
    },
    "cmdargs": {
        '--aaa': "aaa",
        '--bbb': "bbb",
    }, 
    "scan": {
        'aaa': [1,2,3,4,5],
        'bbb': {
            'bbb':[1,2,3],
            'ccc':[1,2,3],
            ":gpuusg": [1,2,False]
        }
    },
    "run": {
        "gpuusg": 3,
    }
}