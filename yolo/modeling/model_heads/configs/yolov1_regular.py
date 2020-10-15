# (name, filters, kernal_size, strides, padding, activation)
head = [
    ("DarkConv", 1024, 3, 1, "valid", "leaky"),
    ("DarkConv", 1024, 3, 2, "valid", "leaky"),
    ("DarkConv", 1024, 3, 1, "valid", "leaky"),
    ("DarkConv", 1024, 3, 1, "valid", "leaky"),
    ("Local", 256, 3, 1, "valid", "leaky"),
    ("Connected", 1715, None, None, None, "linear")
]