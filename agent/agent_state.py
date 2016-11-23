from utils.params import NUM_SLOTS, AgentStateStatus


class AgentState:
    def __init__(self):
        self.slots = {}
        self.num_filled = 0

        self.__init_slots()

    def mark_slot_as_empty(self, slot_id):
        self.slots[slot_id] = AgentStateStatus.EMPTY

    def mark_slot_as_provided(self, slot_id):
        self.slots[slot_id] = AgentStateStatus.OBTAINED

    def mark_slot_as_comfirmed(self, slot_id):
        self.slots[slot_id] = AgentStateStatus.CONFIRMED

    def __init_slots(self):
        for id_ in xrange(NUM_SLOTS):
            self.mark_slot_as_empty(id_)
