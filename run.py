from playAgent import playAgent

### Create the AI 
ai = playAgent()

### Find game window with computer vision
# Find game grid
ai.find_game_grid() # ai.game_grid

# Initialize game
ai.initialize_game_grid() # ai.game

# Play the game
ai.play()

# Messagebox/UI
