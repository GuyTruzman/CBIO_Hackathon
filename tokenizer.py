import re


class Tokenizer:
    def __init__(self, lines):
        self.tokens = list(re.findall(r'([A-Za-z0-9.\-_]+|[:;{}])', lines))
        self.len = len(self.tokens)
        self.index = 0

    def advance(self):
        self.index += 1
        if self.index > self.len:
            return None
        return self.tokens[self.index - 1]

    def go_back(self):
        if self.index > 0:
            self.index -= 1

    def not_empty(self):
        if self.index < self.len:
            return True
        return False
