from utils.params import NUM_SLOTS
from utils.params import UserStateStatus


class UserState(object):
    def __init__(self):
        self.slots = {}
        self.system_act = None

        self._init_slots()

    def __str__(self):
        return "Slots: {}, System-Act: {}".format(self.slots,
                                                  self.system_act.value)

    def mark_slot_as_empty(self, slot_id):
        self.slots[slot_id] = UserStateStatus.EMPTY

    def mark_slot_as_provided(self, slot_id):
        self.slots[slot_id] = UserStateStatus.PROVIDED

    def mark_slot_as_comfirmed(self, slot_id):
        self.slots[slot_id] = UserStateStatus.CONFIRMED

    def _init_slots(self):
        for id_ in xrange(NUM_SLOTS):
            self.mark_slot_as_empty(id_)
