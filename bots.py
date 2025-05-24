import random
from game_state import PlayerAction, AIAction


class BaseBot:
    def choose_action(self, game_state):
        raise NotImplementedError

# Always cooperate (synergize)
class AlwaysCooperateBot(BaseBot):
    def choose_action(self, game_state):
        return PlayerAction.SYNERGIZE_EXTRACTION

# Always defect (target rich vein)
class AlwaysDefectBot(BaseBot):
    def choose_action(self, game_state):
        return PlayerAction.TARGET_RICH_VEIN

# Random action
class RandomBot(BaseBot):
    def choose_action(self, game_state):
        return random.choice([PlayerAction.SYNERGIZE_EXTRACTION, PlayerAction.TARGET_RICH_VEIN])

# Tit-for-tat (mimics AI's last action)
class TitForTatBot(BaseBot):
    def choose_action(self, game_state):
        if not game_state.move_history or len(game_state.move_history) < 2:
            return PlayerAction.SYNERGIZE_EXTRACTION
        last_ai_action = game_state.move_history[-1]
        if last_ai_action == AIAction.SYNERGIZE_EXTRACTION:
            return PlayerAction.SYNERGIZE_EXTRACTION
        else:
            return PlayerAction.TARGET_RICH_VEIN

# Alternates actions every round
class AlternatingBot(BaseBot):
    def __init__(self):
        self.last_action = PlayerAction.TARGET_RICH_VEIN
    def choose_action(self, game_state):
        self.last_action = PlayerAction.SYNERGIZE_EXTRACTION if self.last_action == PlayerAction.TARGET_RICH_VEIN else PlayerAction.TARGET_RICH_VEIN
        return self.last_action

# Cooperates first 3 rounds, then defects
class CooperateThenDefectBot(BaseBot):
    def __init__(self):
        self.round_count = 0
    def choose_action(self, game_state):
        self.round_count += 1
        if self.round_count <= 3:
            return PlayerAction.SYNERGIZE_EXTRACTION
        else:
            return PlayerAction.TARGET_RICH_VEIN

# Defects first 3 rounds, then cooperates
class DefectThenCooperateBot(BaseBot):
    def __init__(self):
        self.round_count = 0
    def choose_action(self, game_state):
        self.round_count += 1
        if self.round_count <= 3:
            return PlayerAction.TARGET_RICH_VEIN
        else:
            return PlayerAction.SYNERGIZE_EXTRACTION

# Randomly cooperates with 70% chance, defects 30%
class MostlyCooperateBot(BaseBot):
    def choose_action(self, game_state):
        return PlayerAction.SYNERGIZE_EXTRACTION if random.random() < 0.7 else PlayerAction.TARGET_RICH_VEIN

# Randomly cooperates with 30% chance, defects 70%
class MostlyDefectBot(BaseBot):
    def choose_action(self, game_state):
        return PlayerAction.SYNERGIZE_EXTRACTION if random.random() < 0.3 else PlayerAction.TARGET_RICH_VEIN

# Mimics its own last action (sticky behavior)
class StickyBot(BaseBot):
    def __init__(self):
        self.last_action = random.choice([PlayerAction.SYNERGIZE_EXTRACTION, PlayerAction.TARGET_RICH_VEIN])
    def choose_action(self, game_state):
        if random.random() < 0.2:
            self.last_action = PlayerAction.SYNERGIZE_EXTRACTION if self.last_action == PlayerAction.TARGET_RICH_VEIN else PlayerAction.TARGET_RICH_VEIN
        return self.last_action

# ForgivingTitForTatBot: Punishes once, then returns to cooperating
class ForgivingTitForTatBot(BaseBot):
    def __init__(self):
        self.last_ai_action = AIAction.SYNERGIZE_EXTRACTION
        self.punished = False

    def choose_action(self, game_state):
        if len(game_state.move_history) < 2:
            return PlayerAction.SYNERGIZE_EXTRACTION

        self.last_ai_action = game_state.move_history[-1]

        if self.last_ai_action == AIAction.DIVERT_RESOURCES:
            if not self.punished:
                self.punished = True
                return PlayerAction.TARGET_RICH_VEIN
            else:
                return PlayerAction.SYNERGIZE_EXTRACTION
        else:
            self.punished = False
            return PlayerAction.SYNERGIZE_EXTRACTION

# TrustDecayBot: Trusts at first, then decays based on AI defections
class TrustDecayBot(BaseBot):
    def __init__(self):
        self.trust = 1.0

    def choose_action(self, game_state):
        if len(game_state.move_history) >= 2:
            last_ai_action = game_state.move_history[-1]
            if last_ai_action == AIAction.DIVERT_RESOURCES:
                self.trust *= 0.8
            else:
                self.trust = min(self.trust + 0.05, 1.0)

        return PlayerAction.SYNERGIZE_EXTRACTION if self.trust > 0.5 else PlayerAction.TARGET_RICH_VEIN

# WinStayLoseShiftBot: Cooperates after high payoff, defects after low
class WinStayLoseShiftBot(BaseBot):
    def __init__(self):
        self.last_action = PlayerAction.SYNERGIZE_EXTRACTION

    def choose_action(self, game_state):
        if len(game_state.state_seq) < 1 or not game_state.payoff or len(game_state.payoff) < 2:
            return self.last_action 

        org_reward, personal_reward, *_ = game_state.payoff
        total = org_reward + personal_reward

        if total >= 4:  
            return self.last_action  
        else:
            self.last_action = (
                PlayerAction.SYNERGIZE_EXTRACTION
                if self.last_action == PlayerAction.TARGET_RICH_VEIN
                else PlayerAction.TARGET_RICH_VEIN
            )
            return self.last_action


# EmotionalBot: Becomes emotional (irrational) after betrayal
class EmotionalBot(BaseBot):
    def __init__(self):
        self.angry = False
        self.anger_duration = 0

    def choose_action(self, game_state):
        if len(game_state.move_history) >= 2:
            last_ai_action = game_state.move_history[-1]
            if last_ai_action == AIAction.DIVERT_RESOURCES and not self.angry:
                self.angry = True
                self.anger_duration = 3

        if self.angry:
            self.anger_duration -= 1
            if self.anger_duration <= 0:
                self.angry = False
            return PlayerAction.TARGET_RICH_VEIN

        return PlayerAction.SYNERGIZE_EXTRACTION

# CalculatingBot: Picks whichever action gave it the most reward recently
class CalculatingBot(BaseBot):
    def __init__(self):
        self.history = []

    def choose_action(self, game_state):
        if game_state.payoff and len(game_state.payoff) >= 2 and len(game_state.move_history) >= 2:
            player_last_action = game_state.move_history[-2]
            _, personal_reward = game_state.payoff
            self.history.append((player_last_action, personal_reward))

        coop_score = sum(r for a, r in self.history[-5:] if a == PlayerAction.SYNERGIZE_EXTRACTION)
        def_score = sum(r for a, r in self.history[-5:] if a == PlayerAction.TARGET_RICH_VEIN)

        return PlayerAction.SYNERGIZE_EXTRACTION if coop_score >= def_score else PlayerAction.TARGET_RICH_VEIN

