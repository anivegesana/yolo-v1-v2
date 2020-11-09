"""
(name, numberinblock, filters, kernel_size, strides, padding, downsample, output, output_name, use_bn)
"""
backbone = [
    ("DarkConv", 1, 32, 3, 1, "same", False, False, None, True), #224x224
    ("MaxPool", 1, None, 2, 2, "valid", False, False, "32", False), #112x112

    ("DarkConv", 1, 64, 3, 1, "same", False, False, None, True), #112x112
    ("MaxPool", 1, None, 2, 2, "valid", False, False, "64", False), #56x56

    ("DarkRouteProcess", 1, 128, None, None, None, False, False, None, True), #56x56

    ("DarkConv", 1, 128, 3, 1, "same", False, False, None, True), #56x56
    ("MaxPool", 1, None, 2, 2, "valid", False, False, "128", False), #28x28

    ("DarkRouteProcess", 1, 256, None, None, None, False, False, None, True), #28x28

    ("DarkConv", 1, 256, 3, 1, "same", False, False, None, True), #28x28
    ("MaxPool", 1, None, 2, 2, "valid", False, False, "256", False), #14x14

    ("DarkRouteProcess", 1, 512, None, None, None, False, False, None, True), #14x14

    ("DarkConv", 1, 512, 3, 1, "same", False, False, None, True), #14x14
    ("MaxPool", 1, None, 2, 2, "valid", False, False, "512", False), #7x7

    ("DarkRouteProcess", 2, 1024, None, None, None, False, False, None, True), #7x7

    ("DarkConv", 1, 1024, 3, 1, "same", False, False, None, True) #7x7
]