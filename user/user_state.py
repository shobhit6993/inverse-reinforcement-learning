from utils.params import UserStateStatus, NUM_SLOTS


class DialogState:
    def __init__(self):
        self.slots = {}
        self.system_act = None

        self.__init_slots()

    def mark_slot_as_empty(self, slot_id):
        self.slots[slot_id] = UserStateStatus.EMPTY

    def mark_slot_as_provided(self, slot_id):
        self.slots[slot_id] = UserStateStatus.PROVIDED

    def mark_slot_as_comfirmed(self, slot_id):
        self.slots[slot_id] = UserStateStatus.CONFIRMED

    def get_empty_slot(self):
        """Returns the slot identifier for an empty slot. If no slot is empty,
        None is returned.

        Returns:
            Slot identifier or None
        """
        for slot_id, val in self.slots.iteritems():
            if val is UserStateStatus.EMPTY:
                return slot_id
        return None

    def __init_slots(self):
        for id_ in xrange(NUM_SLOTS):
            self.mark_slot_as_empty(id_)
