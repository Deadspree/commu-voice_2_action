"""Only commit replies after Unity confirms that the user heard them completely."""
class ConversationSession:
    def __init__(self, history_turns=10):
        self.history = []
        self.history_turns = history_turns
        self.active_turn = None
        self.pending = None

    def begin(self, turn_id):
        self.active_turn = turn_id
        self.pending = None

    def cancel(self, turn_id):
        if self.active_turn == turn_id:
            self.active_turn = None
            self.pending = None

    def commit(self, turn_id):
        if self.active_turn != turn_id or self.pending is None:
            return False
        user, assistant = self.pending
        self.history.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
        self.history = self.history[-self.history_turns * 2:]
        self.pending = None
        self.active_turn = None
        return True
