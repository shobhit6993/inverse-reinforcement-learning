from utils.params import NUM_SLOTS
from utils.params import UserStateStatus


class UserState(object):
    def __init__(self):
        self.slots = {}
        self.system_act = None  # AgentAction

        self._init_slots()

    def __str__(self):
        return "Slots: {}, System-Act: {}".format(self.slots,
                                                  self.system_act.type.value)

    def __hash__(self):
        """Returns a hash value for the `UserState` object.
        Only `system_act` attribute is used for hashing since it's the only
        consequential part of the `UserState`.

        Returns:
            int: Hash value of this class object.
        """
        return hash(self.system_act)

    def __eq__(self, other):
        """Tests if the other `UserState` object's value is same as this one's.
        Only the `system_act` attribute is used to test equality.

        Args:
            other: The object to be compared with.

        Returns:
            Boolean: True if the values of the two objects are the same.
        """
        if type(other) is UserState:
            return self.system_act is other.system_act
        else:
            return False

    def __ne__(self, other):
        """Tests if the other `UserState` object's value is different from
        this one. Only the `system_act` attribute is used to test inequality.

        Args:
            other: The object to be compared with.

        Returns:
            Boolean: True if the values of the two objects are different.
        """
        return not (self == other)

    def mark_slot_as_empty(self, slot_id):
        self.slots[slot_id] = UserStateStatus.EMPTY

    def mark_slot_as_provided(self, slot_id):
        self.slots[slot_id] = UserStateStatus.PROVIDED

    def mark_slot_as_comfirmed(self, slot_id):
        self.slots[slot_id] = UserStateStatus.CONFIRMED

    def _init_slots(self):
        for id_ in xrange(NUM_SLOTS):
            self.mark_slot_as_empty(id_)
