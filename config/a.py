CONFIG = {
    "args": {
        "aaa": 1,
        "bbb": "${:gpuusg}",
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
            'ccc':[1,2,3],
        },
    }
}