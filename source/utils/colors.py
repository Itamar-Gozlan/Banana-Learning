class CT:  # coloring types - format [bool isBold, str color]
    header = 'selected'
    bold = 'bold'
    white = 'white'
    black = 'black'  # unused
    green = 'green'
    orange = 'orange'  # unused
    red = 'red'
    yellow = 'yellow'
    turquoise = 'turquoise'
    underline = 'underline'
    pink = 'pink'  # unused


class Colors:
    def __init__(self):
        self.end = '\33[0m'
        self.white = '\33[0m'
        self.bold = '\33[1m'
        self.underline = '\33[4m'
        self.selected = '\33[7m'
        self.black = '\33[30m'
        self.red = '\33[31m'
        self.green = '\33[32m'
        self.yellow = '\33[93m'
        self.pink = '\33[35m'
        self.orange = '\033[0;33m'
        self.turquoise = '\033[1;36m'

    def cs(self, ct, msg):  # color_string
        # print(msg, ct)
        styled_str = self.white  # default
        for t in ct:
            if hasattr(self, t):
                # print(t)
                value = getattr(self, t)
                styled_str += value
        styled_str += msg
        styled_str += self.end  # default
        return styled_str
