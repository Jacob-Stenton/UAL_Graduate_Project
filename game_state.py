from enum import Enum, auto

class ScreenState(Enum):
    LANDING = auto()
    INFORMATION = auto()
    GAMEPLAY = auto()
    RESULTS = auto()
    END = auto()

class GameState():
    def __init__(self):
        self.round_num = 1
        self.max_rounds = 15
        self.org_points = 0
        self.personal_points = 0
        self.trust = 0.5
        self.running = True
        self.current_screen = ScreenState.LANDING
        self.player_choice = None
        self.state_seq = []
        self.payoff = []
        self.move_history=[]
        self.episode = []
        self.hidden_state = None

    def reset(self):
        self.round_num = 1
        self.max_rounds = 15
        self.org_points = 0
        self.personal_points = 0
        self.trust = 0.5
        self.running = True
        self.current_screen = ScreenState.LANDING
        self.player_choice = None
        self.state_seq = []
        self.payoff = []
        self.move_history=[]
        self.episode = []
        self.hidden_state = None

class PlayerAction(Enum):
    SYNERGIZE_EXTRACTION = auto()
    TARGET_RICH_VEIN = auto()

class AIAction(Enum):
    SYNERGIZE_EXTRACTION = auto()
    DIVERT_RESOURCES = auto()



