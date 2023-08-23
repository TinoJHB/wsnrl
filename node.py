



class Node:
    def __init__(self, E_level, loc, RSSI):
        self.E_level = E_level
        self.loc = loc
        self.RSSI = RSSI

    def get_state(self):
        return [self.E_level, self.loc, self.RSSI]

