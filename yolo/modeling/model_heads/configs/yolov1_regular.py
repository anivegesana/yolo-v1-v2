# (name, filters, kernel_size, strides, padding, activation)
head = [
    ("DarkConv", 1024, 3, 1, "same", "leaky"),
    ("DarkConv", 1024, 3, 2, "same", "leaky"),
    ("DarkConv", 1024, 3, 1, "same", "leaky"),
    ("DarkConv", 1024, 3, 1, "same", "leaky"),
    #("Connected", 4096, None, None, None, "linear"),
    ("Local", 256, 3, 1, 1, "leaky"),
    ("Connected", 1715 // (7*7), None, None, None, "linear")
]