import numpy as np


aa = np.array(2)
print('aa.shape = ', aa.shape)
print('aa.ndim = ', aa.ndim)

aaa = np.array([[1, 2, 3, 4, 5]])
print('aaa.shape = ', aaa.shape)
print('aaa.ndim = ', aaa.ndim)


b = [[1, 2, 3, 4, 5], [6, 7, 8, 8, 9]]

bb = np.array(b)
print('bb.shape = ', bb.shape)
print('bb.ndim = ', bb.ndim)

c = [
    [
        [1, 2, 3, 4, 5],
        [4, 5, 6, 7, 8]
    ],
    [
        [7, 8, 9, 10, 11],
        [10, 11, 12, 13, 14]
    ]
]

cc = np.array(c)
print('cc.shape = ', cc.shape)
print('cc.ndim = ', cc.ndim)
