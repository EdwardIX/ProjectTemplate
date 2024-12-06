CONFIG = {
    "args": {
        "aaa": 1,
        "bbb": 2,
        "ccc": '1',
    },
    "reqs": {
        "gpuusg": 3,
    },
    "cmdargs": {
        '--aaa': "aaa",
        '--bbb': "bbb",
    }, 
    "scan": {
        'aaa': [1,2],
        'bbb': {
            'bbb':[1,2,3],
            'ccc':[1,2,3],
        }
    }
}