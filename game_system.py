from game_state import PlayerAction, AIAction
from events import ALL_EVENTS

points_matrix = {
        # player, ai, orgpoints, playerpoints,
        (PlayerAction.SYNERGIZE_EXTRACTION, AIAction.SYNERGIZE_EXTRACTION): [100, 60],
        (PlayerAction.SYNERGIZE_EXTRACTION, AIAction.DIVERT_RESOURCES): [50, 0],
        (PlayerAction.TARGET_RICH_VEIN, AIAction.SYNERGIZE_EXTRACTION): [5, 120],
        (PlayerAction.TARGET_RICH_VEIN, AIAction.DIVERT_RESOURCES): [10, 50]
    }

# points shouldn't be equal beacuase of training trust ratio

def calculate_points(player_action, ai_action, event=None):
    
    points_payoff = list(points_matrix[(player_action, ai_action)])

    if event:
        if (player_action,ai_action) == event.requirement:
            if event.target == "org_points":
                points_payoff[0] = int(points_payoff[0] * event.effect)
            elif event.target == "personal_points":
                points_payoff[1] = int(points_payoff[1] * event.effect)
                
    return points_payoff


