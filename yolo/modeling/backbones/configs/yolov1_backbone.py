"""
(name, numberinblock, filters, kernel_size, strides, padding, downsample, output, output_name, use_bn)
"""
backbone = [
    ("DarkConv", 1, 64, 7, 2, "same", False, False, None, False),
    ("MaxPool", 1, None, 2, 2, "valid", False, False, None, False),

    ("DarkConv", 1, 192, 3, 1, "same", False, False, None, False),
    ("MaxPool", 1, None, 2, 2, "valid", False, False, None, False),

    ("DarkRouteProcess", 1, 256, None, None, None, False, False, None, False),
    ("DarkRouteProcess", 1, 512, None, None, None, False, False, None, False),
    ("MaxPool", 1, None, 2, 2, "valid", False, False, None, False),

    ("DarkRouteProcess", 4, 512, None, None, None, False, False, None, False),
    ("DarkRouteProcess", 1, 1024, None, None, None, False, False, None, False),
    ("MaxPool", 1, None, 2, 2, "valid", False, False, None, False),

    ("DarkRouteProcess", 2, 1024, None, None, None, False, True, None, False),
]
