#!/bin/bash

CONTROLLER_SCRIPT="./pi_controller.py"

GAME_COMMAND="python ./main_pi.py"

echo "Starting controller script..."
python "$CONTROLLER_SCRIPT" &
CONTROLLER_PID=$! 

echo "Controller PID: $CONTROLLER_PID"
echo "Starting game..."

cleanup() {
    echo "Game exited. Stopping controller script (PID: $CONTROLLER_PID)..."
    if kill "$CONTROLLER_PID" > /dev/null 2>&1; then
        wait "$CONTROLLER_PID" 2>/dev/null # Wait for it to actually terminate
        echo "Controller stopped."
    else
        echo "Controller (PID: $CONTROLLER_PID) was already stopped or not found."
    fi
}

trap cleanup EXIT SIGINT
eval "$GAME_COMMAND"

echo "Script finished."