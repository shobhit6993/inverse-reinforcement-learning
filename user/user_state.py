"""User's state."""

from utils.params import NUM_SLOTS, UserStateStatus


class UserState(object):
    """State class for user. The user's state consists of status
        of all the slots and the most recent action of the agent.

    Attributes:
        slots (dict): Dictionary of slot-status pairs. The dictionary is keyed
            by slot identifiers, and the values correspond to the
            status (`UserStateStatus`) of each slot.
        system_act (AgentAction): Most recent action taken by the agent.
    """

    def __init__(self):
        self.slots = {}
        self.system_act = None

        self._init_slots()

    def __str__(self):
        return "Slots: {}, System-Act: {}".format(self.slots,
                                                  self.system_act.type.value)

    def __hash__(self):
        # Only `system_act` attribute is used for hashing since it's the only
        # consequential part of the `UserState`.

        return hash(self.system_act)

    def __eq__(self, other):
        # Only the `system_act` attribute is used to test equality.
        if type(other) is UserState:
            return self.system_act is other.system_act
        else:
            return False

    def __ne__(self, other):
        # Only the `system_act` attribute is used to test inequality.
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
