
"""
This is a list of the object label to their respective "R" and "G" encodings for color
"""


class Label:
    def __init__(self, text_name, rg_code):
        self.name = text_name
        self.rg_code = rg_code


FLOOR = Label('floor', [30, 208])
CARPET = Label('carpet', [80, 130])
WALL = Label('wall', [110, 162])
WINDOW = Label('window', [110, 239])