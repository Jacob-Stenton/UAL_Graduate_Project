This project demonstrates the development of an AI agent capable of learning and modeling cognitive trust in a human collaborator. It showcases a novel approach to human-AI interaction within a simulated organizational environment. The core of the project is an interactive game where the AI adapts its behavior based on a human player's actions, highlighting the emergence of a bidirectional trust relationship.

Technical Stack:
 - Python: The entire project is implemented in Python.
 - Deep Reinforcement Learning (DRQN): The AI agent's core is a Deep Recurrent Q-Network (DRQN) built using TensorFlow and Keras. This architecture includes an LSTM (Long Short-Term Memory) layer, enabling the agent to learn from sequences of past actions and states, which is critical for modeling the evolving trust relationship over time.
 - Continuous Learning: The agent is designed to continue learning and adapting from real-time human gameplay, showcasing a robust, practical application of reinforcement learning.
 - User Interface: The project features a professional, modern terminal-based UI developed with the Rich library, providing a clean and engaging user experience.
 - Training: The agent is trained on a suite of pre-programmed bots that exhibit various strategies (e.g., cooperating, defecting, and retaliatory behaviors), demonstrating a systematic and thorough approach to model training.

<img width="2784" height="2060" alt="Game Screens" src="https://github.com/user-attachments/assets/a7b42bfe-5472-45c4-a313-1bf690e0c629" />

![IMG-20250623-WA0000](https://github.com/user-attachments/assets/7c2dadeb-096d-4f97-b49a-da054aac8765)
This casing was designed in Blender 3D and 3d printed for a final showcase.
 - This was an extension on top of my final graduation project.

Input Handling: 
 - An Arduino handles button inputs and sends them via serial to a Raspberry Pi, which is running the game and RL model. This was done because the screen required every pin on the RPi to function.
 - Player input is managed with the Pynput library, which allows for real-time keyboard control within the terminal environment. The arduino emulated keyboard controls sending either up, down or enter.
