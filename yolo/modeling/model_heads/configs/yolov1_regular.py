# (name, filters, kernel_size, strides, padding, activation)
head = [
    ("DarkConv", 1024, 3, 1, "same", "leaky"),
    ("DarkConv", 1024, 3, 2, "same", "leaky"),
    ("DarkConv", 1024, 3, 1, "same", "leaky"),
    ("DarkConv", 1024, 3, 1, "same", "leaky"),
    ("Local", 256, 3, 1, "same", "leaky"),
    ("Flatten", None, None, None, None, None),
    ("Connected", 1715, None, None, None, "linear")
]
