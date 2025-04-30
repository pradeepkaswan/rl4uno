import sys
import time
import os
import pygame
import numpy as np
from environment import UnoEnvironment
from renderer import *

MODEL_PATH = 'example_model.h5'

MOVE_TIME = 0
SHOW_NON_HUMAN_CARDS = False

FONT = 'arial'
FONT_SIZE_LARGE = 25
FONT_SIZE_SMALL = 18
GAME_MESSAGE_DURATION = 3
"""
Colour choice configuration:
HUMAN_COLOUR_SELECTION: 'prompt' or 'random' for human players
AI_COLOUR_SELECTION: 'heuristic' or 'random' for AI players
"""
HUMAN_COLOUR_SELECTION = 'prompt'
AI_COLOUR_SELECTION = 'heuristic'

WINDOW_SIZE = (1200, 600)
BACKGROUND = (180, 180, 180)
POSSIBLE_PLAYER_TYPES = ['AI', 'Human', 'Naive']

# check command line arguments
if len(sys.argv) < 3:
    print(f'Not enough players specified ({len(sys.argv) - 1}).', end='')
    player_options_str = ', '.join(POSSIBLE_PLAYER_TYPES)
    print(f'Select player types with command line attributes. (options: {player_options_str})')
    exit()

# extract player types (0 for AI, 1 for human)
player_types = []
player_names = []
for player in sys.argv[1:]:
    found = False
    # find player type and add to list
    for i, possible in enumerate(POSSIBLE_PLAYER_TYPES):
        if player.lower() == possible.lower():
            player_types.append(i)
            player_names.append(f'{possible}-{np.sum([i == curr for curr in player_types])}')
            found = True
            break

    if not found:
        # unknown player type found
        player_options_str = ', '.join(POSSIBLE_PLAYER_TYPES)
        print(f'Unknown player type "{player}". Please select from {player_options_str}.')
        exit()
# keep a copy for restarts
initial_player_types = player_types.copy()
initial_player_names = player_names.copy()

# initialize pygame
print('Initializing pygame...')
pygame.init()
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption(f'Uno game - {" vs. ".join([POSSIBLE_PLAYER_TYPES[i] for i in player_types])}')
clock = pygame.time.Clock()
font_large = pygame.font.SysFont(FONT, FONT_SIZE_LARGE, bold=True)
font_small = pygame.font.SysFont(FONT, FONT_SIZE_SMALL)

# check if AI player is present
# Load AI model for inference only (no optimizer/compile needed)
seq_model = False
model = None
if 0 in player_types:
    # try loading AI model; if missing, fall back to naive moves
    if MODEL_PATH and os.path.exists(MODEL_PATH):
        print('Loading model (inference mode)...')
        from keras.models import load_model
        try:
            model = load_model(MODEL_PATH, compile=False)
            # detect if model expects sequence input (3D: batch, time_steps, features)
            inp_shape = getattr(model, 'input_shape', None)
            if isinstance(inp_shape, tuple) and len(inp_shape) == 3:
                seq_model = True
        except Exception as e:
            print(f'Warning: failed to load model: {e}. AI will play naively.')
            model = None
    else:
        print(f'Warning: model file "{MODEL_PATH}" not found. AI will play naively.')

# initialize game variables
game_messages = []
last_move = time.time()

print('Initializing game environment...')
env = UnoEnvironment(len(player_types))

# init flags
mouse_down = False
clicked = False
done = False
game_finished = False

print('Done! Running game loop...')
 
def prompt_colour():
    """Prompt human to choose a colour for wild or draw-4 cards."""
    global screen, font_large, clock
    width, height = WINDOW_SIZE
    # layout for colour selection boxes
    box_size = 60
    margin = 20
    colours = CARD_COLOURS[:UnoEnvironment.NUM_COLOURS]
    total_width = len(colours) * box_size + (len(colours) - 1) * margin
    start_x = (width - total_width) // 2
    start_y = (height - box_size) // 2
    selecting = True
    chosen = None
    while selecting:
        # draw prompt UI
        screen.fill(BACKGROUND)
        # draw instruction text
        text = font_large.render('Choose next colour', True, (0, 0, 0))
        text_rect = text.get_rect(center=(width // 2, start_y - box_size // 2 - font_large.get_height()))
        screen.blit(text, text_rect)
        # draw colour boxes
        boxes = []
        for idx, color in enumerate(colours):
            rect = pygame.Rect(start_x + idx * (box_size + margin), start_y, box_size, box_size)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (0, 0, 0), rect, 2)
            boxes.append(rect)
        pygame.display.flip()
        # handle events
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                pos = ev.pos
                for idx, rect in enumerate(boxes):
                    if rect.collidepoint(pos):
                        chosen = idx
                        selecting = False
                        break
        clock.tick(30)
    return chosen
while not done:
    clicked = False
    # event handling
    for event in pygame.event.get():
        # quit window
        if event.type == pygame.QUIT:
            done = True
        # restart if game over and R pressed
        elif event.type == pygame.KEYDOWN and game_finished and event.key == pygame.K_r:
            # reset environment and players
            env.reset()
            player_types = initial_player_types.copy()
            player_names = initial_player_names.copy()
            game_messages = []
            game_finished = False
            last_move = time.time()
        # mouse handling
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_down = True
                clicked = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                mouse_down = False

    if not game_finished:
        # reset screen
        screen.fill(BACKGROUND)
        # render objects
        card_rects = draw_env(env, screen, font_large, player_names, player_types, draw_non_human=SHOW_NON_HUMAN_CARDS)
        draw_messages(game_messages, screen, font_small)
        pygame.display.flip()

        # game logic
        if player_types[env.turn] == 0:
            if time.time() - last_move < MOVE_TIME:
                # wait until move delay is reached
                action = None
            else:
                # AI player
                if model is not None:
                    # run model inference, handling both 2D and sequence (3D) inputs
                    state = env.get_state()
                    if seq_model:
                        state_input = state.reshape((1, 1, -1))
                        preds = model.predict(state_input, verbose=0)[0, 0]
                    else:
                        state_input = state.reshape((1, -1))
                        preds = model.predict(state_input, verbose=0)[0]
                    action = int(np.argmax(preds))
                    # fallback to random legal if model picks illegal
                    if not env.legal_move(action):
                        game_messages.append((time.time(), f'{player_names[env.turn]} selected an illegal action, playing random card.'))
                        action = np.random.randint(env.action_count())
                        while not env.legal_move(action):
                            action = np.random.randint(env.action_count())
                else:
                    # naive fallback: play first legal action
                    action = 0
                    while not env.legal_move(action):
                        action += 1
        elif player_types[env.turn] == 1:
            # human player
            card_selected = False
            if clicked:
                # check if one of the player's cards was selected
                mouse_pos = pygame.mouse.get_pos()
                index = np.argwhere([rect.contains(mouse_pos + (0, 0)) for rect in card_rects])
                if len(index) > 0:
                    if index[0,0] == np.sum(env.players[env.turn].cards):
                        # draw from stack selected
                        action = len(UnoEnvironment.CARD_TYPES)
                    else:
                        # one of the player's cards was clicked
                        cards = [[index] * int(count) for index, count in enumerate(env.players[env.turn].cards) if count > 0]
                        cards = np.concatenate(cards)
                        # get the selected card index
                        action = cards[index[0,0]]

                    if env.legal_move(action):
                        # set selected to true if the selected action is legal
                        card_selected = True
                    else:
                        game_messages.append((time.time(), 'Illegal move!'))
            if not card_selected:
                # no card was selected
                action = None
        elif player_types[env.turn] == 2:
            if time.time() - last_move < MOVE_TIME:
                # wait until move delay is reached
                action = None
            else:
                # naive player
                action = 0
                # search for first legal move
                while not env.legal_move(action):
                    action += 1

        if action is not None:
            # determine chosen colour for wild or draw-4 cards
            chosen_colour = None
            # identify card type if action is a card play
            if action < len(UnoEnvironment.CARD_TYPES):
                card_type = UnoEnvironment.CARD_TYPES[action][1]
                if card_type in (13, 14):
                    # human player choice via prompt
                    if player_types[env.turn] == 1 and HUMAN_COLOUR_SELECTION == 'prompt':
                        chosen_colour = prompt_colour()
                    # AI player heuristic choice
                    elif player_types[env.turn] == 0 and AI_COLOUR_SELECTION == 'heuristic':
                        # count cards by colour in hand
                        counts = []
                        for col in range(UnoEnvironment.NUM_COLOURS):
                            mask = UnoEnvironment.CARD_TYPES[:,0] == col
                            counts.append(env.players[env.turn].cards[mask].sum())
                        chosen_colour = int(np.argmax(counts))
            # play the selected action with optional chosen colour
            _, _, game_finished, step_info = env.step(action, chosen_colour)
            last_move = time.time()

            turn = step_info['turn']
            player_status = step_info['player']

            # check if the current player is out of the game
            if player_status == -1 or player_status == 2:
                if player_status == -1:
                    game_messages.append((time.time(), f'{player_names[turn]} eliminated due to illegal move.'))
                elif player_status == 2:
                    game_messages.append((time.time(), f'{player_names[turn]} has finished!'))
                del player_types[turn]
                del player_names[turn]

            # update game screen once after game has finished
            if game_finished:
                screen.fill(BACKGROUND)
                draw_env(env, screen, font_large, player_names, player_types, draw_non_human=True)
                draw_messages(game_messages, screen, font_small)
                pygame.display.flip()

    # remove game messages older than GAME_MESSAGE_DURATION
    game_messages = [msg for msg in game_messages if time.time() - msg[0] < GAME_MESSAGE_DURATION]

    # limit the frame rate
    clock.tick(30)

pygame.quit()
