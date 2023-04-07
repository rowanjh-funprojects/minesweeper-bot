from playAgent import playAgent

### Create the AI 
ai = playAgent()

### Find game window with computer vision
# Find game grid
ai.find_game_grid() # ai.game_grid

# Initialize game
ai.initialize_game_grid() # ai.game

ai.play(nmoves = 100)
ai.show_beliefs()
# Play the game

# Messagebox/UI
