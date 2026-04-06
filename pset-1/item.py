# DO NOT modify the following class definition
class Item(object):
    def __init__(self, name, val, vol, weight, cannot_carry = False,
                 cannot_check = False):
        if cannot_carry and cannot_check:
            raise ValueError("Error: cannot_carry and cannot_check can't both be true")
        self._name = name
        self._val = val
        self._vol = vol
        self._weight = weight
        self._cannot_carry = cannot_carry
        self._cannot_check = cannot_check

    def get_info(self):
        return (self._val, self._vol, self._weight, self._cannot_carry,
                self._cannot_check)

    def get_name(self):
        return self._name

    def get_value(self):
        return self._val

    def get_volume(self):
        return self._vol

    def get_weight(self):
        return self._weight

    def cannot_carry(self):
        return self._cannot_carry

    def cannot_check(self):
        return self._cannot_check

    def __str__(self):
        if self._cannot_carry:
            constraint = ', cannot carry'
        elif self._cannot_check:
            constraint = ', cannot check'
        else:
            constraint = ''

        return (f'{self._name}: val = {self._val}, vol = {self._vol}, ' +
                f'weight = {self._weight}{constraint}')

    def __repr__(self):
        # controls what gets printed out in a list of Items
        return self.__str__()
