from utils.params import AgentStateStatus
from utils.params import NUM_SLOTS


class AgentState(object):
    def __init__(self):
        self.slots = {}

        self._init_slots()

    def __str__(self):
        return "Slots: {}".format(self.slots)

    def mark_slot_as_empty(self, slot_id):
        self.slots[slot_id] = AgentStateStatus.EMPTY

    def mark_slot_as_obtained(self, slot_id):
        self.slots[slot_id] = AgentStateStatus.OBTAINED

    def mark_slot_as_confirmed(self, slot_id):
        self.slots[slot_id] = AgentStateStatus.CONFIRMED

    def get_empty_slot(self):
        """Returns the slot identifier for an empty slot. If no slot is empty,
        None is returned.

        Returns:
            Slot identifier or None
        """
        for slot_id, val in self.slots.iteritems():
            if val is AgentStateStatus.EMPTY:
                return slot_id
        return None

    def get_unconfirmed_slot(self):
        """Returns the slot identifier for an unconfirmed slot. If there is no
        such slot, None is returned.

        An "unconfirmed slot" is defined as one which has been "OBTAINED", but
        not yet "CONFIRMED".

        Returns:
            Slot identifier or None
        """
        for slot_id, val in self.slots.iteritems():
            if val is AgentStateStatus.OBTAINED:
                return slot_id
        return None

    def _init_slots(self):
        for id_ in xrange(NUM_SLOTS):
            self.mark_slot_as_empty(id_)
