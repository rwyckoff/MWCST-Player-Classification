import numpy as np
from psychopy import visual, core, event, gui, data
import os
import math

"""
The Modified Wisconsin Card Sorting Game. Allows participants to enter their data and then play the game.
Per-session data is recorded and saved as CSV files to be later cleaned, summarized, and run through
the machine learning algorithms.
"""


# Instantiate information about the experiment session.
exp_name = u'MWCST'
exp_info = {u'session': u'1', u'Participant ID': u'00000', 'date': data.getDateStr(), 'exp_name': exp_name,
            'DOB': data.getDateStr(), 'Sex': 'F', 'Country': 'USA', 'State': 'Colorado', 'City': 'Colorado Springs',
            'Diagnosis (optional)': 'None'}

# Instantiate pre-experiment dialog box for participants to edit data.
experiment_dialog = gui.DlgFromDict(title="Wisconsin Card Sorting Task", dictionary=exp_info)

# If user presses cancel, quit early.
if not experiment_dialog.OK:
    core.quit()

# Define the data folder and sessions sub-folder.
data_folder = 'Data'
ind_session_folder = 'Sessions'
output_dir = os.path.join(data_folder, ind_session_folder)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
ind_session_folder = output_dir

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
data_file = os.path.join(ind_session_folder,
                         '%s_%s_%s_%s.csv' % (exp_info['Participant ID'],
                                              exp_name, exp_info['date'], exp_info['session']))

print(f"Beginning M-WCST Experiment Session, saving to file {data_file}")

# Set the wait times.
display_choice_time = 0.1
feedback_display_time = 0.5
instruction_time = 1

# Set the starting x and y coordinate values.
x = 0
y = 1

# Set the card order file, which defines the order the cards will be shown.
response_card_order_file = 'M_Response_Card_Order.csv'
response_card_order = np.genfromtxt(response_card_order_file, delimiter=',')

instructions = ('Instructions:\nEach trial, match the bottom card with one of the four cards at the '
                'top of the screen by clicking the card at top that you think matches. The cards may '
                'be matched according to one of three card attributes: symbol color, symbol shape, '
                'or number of symbols. Upon clicking your chosen card, you will be given feedback on '
                'whether your selection was right or wrong. Use the feedback to determine which attribute '
                'needs to be considered to match the cards. The attribute rule may eventually change, but '
                'you will be notified when it does.\n\nPress any key to begin. Good luck!')


def point_in_triangle(v1, v2, v3, pt):
    """
    Determines if point pt is located inside the triangle defined by the vertices v1, v2, and v3.
    :return: True if pt is inside the triangle and false otherwise.
    """

    def sign(p1, p2, p3):
        return (p1[x] - p3[x]) * (p2[y] - p3[y]) - (p2[x] - p3[x]) * (p1[y] - p3[y])

    b1 = sign(pt, v1, v2) < 0
    b2 = sign(pt, v2, v3) < 0
    b3 = sign(pt, v3, v1) < 0
    return b1 == b2 and b2 == b3


def draw_triangle(symbol, vertices, value=1):
    """
    Draws the triangle symbol of a card, where symbol.shape is the shape of the ndarray as a tuple.
    :return: A triangle symbol
    """
    for i in range(symbol.shape[0]):
        for j in range(symbol.shape[1]):
            if point_in_triangle(vertices[0], vertices[1], vertices[2], (i, j)):
                symbol[i, j] = value
    return symbol


def draw_circle(symbol, pos, radius, value=1):
    """
    Draws the circle symbol of a card, where symbol.shape is the shape of the ndarray as a tuple.
    :return: A circle symbol
    """
    for i in range(symbol.shape[0]):
        for j in range(symbol.shape[1]):
            if np.sqrt((i - pos[x]) ** 2 + (j - pos[y]) ** 2) < radius:
                symbol[i, j] = value
    return symbol


def draw_star(symbol, pos, num_vertices, outer_radius, inner_radius):
    """
    Draws the star symbol of a card.
    :param outer_radius: Radius of the outer circle used in drawing the star.
    :param inner_radius: Radius of the inner circle used in drawing the star.
    :return: a star symbol
    """
    symbol = draw_circle(symbol, pos, inner_radius)
    phi = np.linspace(0, 2 * np.pi, num_vertices + 1)[:-1]
    ops = np.array([np.cos(phi) * outer_radius, np.sin(phi) * outer_radius]) + np.array(pos, ndmin=2).T
    phi += phi[1] / 2.0
    ips = np.array([np.cos(phi) * inner_radius, np.sin(phi) * inner_radius]) + np.array(pos, ndmin=2).T
    for i in range(phi.size):
        symbol = draw_triangle(symbol, [ips[:, i - 1], ips[:, i], ops[:, i]])
    return symbol


def calc_velocity(start_m_pos, end_m_pos, m_time):
    """
    Calculates the average mouse velocity for a given trial.
    :param start_m_pos: X and y mouse coordinates at the start of the trial.
    :param end_m_pos:  X and y mouse cooridinates at the end of the trial.
    :param m_time: The time betweent the two mouse clicks.
    :return: the average mouse velocity for the trial.
    """
    if m_time == 0:
        return 0
    else:
        start_x = start_m_pos[0]
        start_y = start_m_pos[1]
        end_x = end_m_pos[0]
        end_y = end_m_pos[1]
        dist_x = end_x - start_x
        dist_y = end_y - start_y
        dist = math.sqrt(math.pow(dist_x, 2) + math.pow(dist_y, 2))
        return dist / m_time


class Experiment:

    def __init__(self):
        self.output = None

        # Initialize the game window.
        self.window = visual.Window(size=screen_resolution, color="black", units='height', fullscr=False,
                                    monitor='testMonitor', winType="pyglet",
                                    allowGUI=True, waitBlanking=True)

        # Draw the title text.
        self.title = visual.TextStim(self.window, pos=TITLE_POS, color='white', height=0.05, units='height')

        # Initialize the mouse event to detect mouse clicks and position.
        self.mouse = event.Mouse(True, None, win=self.window)

        # Initialize the previous rule to 5, which doesn't exist.
        self.prev_rule = None

        # Initialize the lists of cards and elements (the card symbols).
        self.cards = []
        self.elems = []
        element_size = 0.05  # Size of the elements(card symbols).

        # Add the stimulus cards
        for i in range(4):
            self.cards.append(visual.Rect(self.window, CARD_W, CARD_H, fillColor='white',
                                          pos=((i - 1.5) * (CARD_X + CARD_W), CARD_Y), lineColor='black',
                                          interpolate=False, units='height'))
            self.elems.append(visual.ElementArrayStim(self.window, nElements=4, sizes=element_size, colors='black',
                                                      fieldPos=((i - 1.5) * (CARD_X + CARD_W), CARD_Y), elementTex=None,
                                                      units='height'))

        # Add the response card.
        self.cards.append(visual.Rect(self.window, CARD_W, CARD_H, fillColor='white',
                                      pos=RESPONSE_POS, lineColor='black', interpolate=False, units='height'))
        self.elems.append(visual.ElementArrayStim(self.window, nElements=4, sizes=element_size, colors='black',
                                                  fieldPos=RESPONSE_POS, elementTex=None, units='height'))

        # Make the discard pile.
        self.cards.append(visual.Rect(self.window, CARD_W, CARD_H, fillColor='black',
                                      pos=DISCARD_POS, lineColor='black', interpolate=False, units='height'))
        self.elems.append(visual.ElementArrayStim(self.window, nElements=4, sizes=element_size, colors='black',
                                                  fieldPos=DISCARD_POS, elementTex=None, units='height'))

        # Initialize the feedback text options.
        self.text = visual.TextStim(self.window, pos=FEEDBACK_POS, color='blue', height=0.08, units='height')

    def run_trial(self, trial, response_card):
        """
        Performs one trial with one response card. The trial ends after the participant selects a stimulus card.
        :param trial: The trial number.
        :param response_card: The response card the participant recieves for this trial.
        """
        clock = core.Clock()  # Initializes a clock to keep trial time.
        start_m_pos = self.mouse.getPos()  # Get the trial's initial mouse position
        misclicks = 0  # Initialize misclicks.

        # Set up the four cards to be fixed, with each trial having the same four.
        stim_cards = [[0, 1, 0], [1, 2, 1], [3, 3, 2], [2, 0, 3]]

        # Put all stimulus cards on the screen
        for i in range(len(stim_cards)):
            self.cards[i].draw()
            self.elems[i].setColors(COLORS[stim_cards[i][0]])
            self.elems[i].setMask(SHAPES[stim_cards[i][1]])
            self.elems[i].setXYs(SHAPE_POS[stim_cards[i][2]])
            self.elems[i].draw()

        # Draw the response card card on the screen (on the bottom)
        self.cards[4].setPos(RESPONSE_POS)
        self.cards[4].draw()
        self.elems[4].setFieldPos(RESPONSE_POS)
        self.elems[4].setColors(COLORS[response_card[0]])
        self.elems[4].setMask(SHAPES[response_card[1]])
        self.elems[4].setXYs(SHAPE_POS[response_card[2]])
        self.elems[4].draw()

        # "Flips" the drawn layers around so that what was at the back is now at the front and visible.
        self.window.flip()

        # Wait for a click from the participant.
        self.mouse.clickReset()
        while True:
            # getPressed() returns which button was pressed (m_key) and the time it was pressed (m_time).
            m_key, m_time = self.mouse.getPressed(getTime=True)

            # If it was the mouse button pressed, detect if the cursor is inside a card.
            if sum(m_key) > 0:
                selected_card = -1  # Set a default card-identifying value.

                # Gets the position of the mouse at the time of the button press.
                m_pos = self.mouse.getPos()
                for i in range(4):
                    if self.cards[i].contains(self.mouse):
                        selected_card = i
                        m_time = m_time[0]
                # If one of the cards was clicked, record and break from loop
                if selected_card > -1:
                    print(f"Mouse-click coordinates: {m_pos}")  # Prints the mouse position to the screen.
                    print(f"Response time: {m_time}")
                    end_m_pos = m_pos  # Sets end mouse position as the current mouse pos.
                    break
                # Otherwise, record a misclick and keep looping.
                else:
                    core.wait(0.05)  # Pause briefly to not allow too many misclick recordings.
                    misclicks = misclicks + 1
                    core.wait(0.05)
                    self.mouse.clickReset()

            keys_pressed = event.getKeys(keyList=['1', '2', '3', '4', 'escape'])

            # If the participant has pressed escape, quit now.
            if "escape" in keys_pressed:
                print("**********-----Session ended early-----**********")
                core.quit()

            # If there was a valid key press (1-4) not escape, print the key pressed and select the appropriate card.
            if len(keys_pressed) > 0 and KEYBOARD_SELECTION_ON:
                end_m_pos = self.mouse.getPos()
                print(f"Key pressed: {keys_pressed[-1]}")
                selected_card = int(keys_pressed[-1]) - 1
                m_time = clock.getTime()
                break

        # Determine the mouse velocity based on the beginning mouse position, end mouse position, and the response time.
        # This isn't a perfect way to measure velocity, but it will give us an idea of how quickly the user is using
        # the mouse.
        m_velocity = calc_velocity(start_m_pos, end_m_pos, m_time)

        # Print what the selected card was.
        print(f"Selected stimulus card: {str(selected_card + 1)}\n")

        # Update the new cards and draw them.
        self.cards[5].fillColor = self.cards[selected_card].fillColor
        self.cards[5].setPos(DISCARD_POS)
        self.elems[5].setFieldPos(RESPONSE_POS)
        self.elems[5].setColors(COLORS[response_card[0]])
        self.elems[5].setMask(SHAPES[response_card[1]])
        self.elems[5].setXYs(SHAPE_POS[response_card[2]])
        self.elems[5].setFieldPos(DISCARD_POS)

        for i in range(4):
            self.cards[i].draw()
            self.elems[i].draw()

        # Draw the response card.
        self.cards[5].draw()
        self.elems[5].draw()

        # Flip the window to display the newly-drawn cards.
        self.window.flip()

        # Wait a set amount of time before displaying the next set of cards.
        core.wait(display_choice_time)

        # Write data collected during this trial to the csv data file.
        self.output.write(f'{trial + 1},{selected_card + 1},{self.rule},'
                          f'{m_time}, {start_m_pos[0]}, {start_m_pos[1]},'
                          f' {end_m_pos[0]}, {end_m_pos[1]}, {m_velocity}, {misclicks},')

        # Display feedback text to participant and record in the data file if there was an error.
        # In the data file, 0 means no error, 1 means non-perseverative error, and 2 means perseverative error.
        if response_card[self.rule] == stim_cards[selected_card][self.rule]:
            self.text.setText("That's right!")
            self.text.setColor('green')
            self.correct_streak += 1  # Increase the correct selection streak by one.
            self.rule_correct_streak += 1
            self.output.write('0,')

        else:
            self.text.setText("That's wrong.")
            self.text.setColor('red')
            self.correct_streak = 0  # Reset the correct selection streak.
            self.rule_correct_streak = 0

            # Determine if the error was perseverative. If it was, output a 1 to the next column. Otherwise, a 0.
            if self.prev_rule is not None:
                if stim_cards[selected_card][self.prev_rule] == response_card[self.prev_rule]:
                    print("Perseverative error!")
                    self.output.write('2,')
                else:
                    print("Non-perseverative error.")
                    self.output.write('1,')
            else:
                self.output.write('1,')

        # Set the previous rule
        for i in range(len(stim_cards[selected_card])):
            if response_card[i] == stim_cards[selected_card][i]:
                self.prev_rule = i

        for i in range(4):
            self.cards[i].draw()
            self.elems[i].draw()

        # Record the current correct streak
        self.output.write(f'{self.correct_streak},')

        # Write the response_card card info to the data file.
        self.output.write('%d,%d,%d,' % tuple(response_card))

        # Display the feedback text.
        self.text.draw()
        self.title.draw()

        # Draw the cards again.
        self.cards[5].draw()
        self.elems[5].draw()
        self.title.draw()
        self.window.flip()
        core.wait(feedback_display_time)

    def run(self, num_trials, streak_rule_change=6):
        """
        Run all the trials for this session.
        :param num_trials: The number of trials to run.
        :param streak_rule_change: The number of correct responses until the rule changes.
        """
        # Draw the title at the top.
        self.title.setText("Modified Wisconsin Card Sorting Test")
        self.title.draw()

        # Clear the discard pile(s).
        self.cards[5].fillColor = 'black'
        self.elems[5].color = 'black'
        self.cards[5].draw()

        # Set streak to 0.
        self.correct_streak = 0
        self.rule_correct_streak = 0  # The number of consecutive correct selections during the current rule.

        # Set rule randomly to one of the three
        sel = np.array([0, 1, 2])
        self.rule = sel[np.random.randint(3)]
        print(f"RULE CHANGED TO: {self.rule}")

        # Run all the trials
        for t in range(num_trials):
            self.title.draw()

            # Determine the next response_card card.
            response_card = [int(response_card_order[t, 0]),
                             int(response_card_order[t, 1]),
                             int(response_card_order[t, 2])]

            print('Trial num=%d, Streak = %d, Rule = %d' % (t, self.correct_streak, self.rule))
            print(f"Response card: {response_card}")

            if t > 0:
                # If the participant has a high enough correct selection streak, change the rule.
                if self.rule_correct_streak == streak_rule_change:
                    self.rule_correct_streak = 0
                    self.prev_rule = None  # Set previous rule to a non-existent one.

                    # Show participant that the rule is changing. Comment out to disable.
                    self.text.setText("RULE CHANGE")
                    self.text.setColor('white')
                    self.text.draw()
                    self.title.draw()

                    # Change the rule randomly, excluding the current rule from the random draw.
                    if self.rule == 2:
                        # self.rule = 0
                        sel = np.array([0, 1])
                        self.rule = sel[np.random.randint(2)]
                        print(f"RULE CHANGED TO: {self.rule}")
                    elif self.rule == 1:
                        # self.rule += 1
                        sel = np.array([0, 2])
                        self.rule = sel[np.random.randint(2)]
                        print(f"RULE CHANGED TO: {self.rule}")
                    else:
                        sel = np.array([1, 2])
                        self.rule = sel[np.random.randint(2)]
                        print(f"RULE CHANGED TO: {self.rule}")

                    self.output.write('1,')  # Record '1' if category was completed this round.
                else:
                    self.output.write('0,')  # Record '0' if category was not completed this round.

                # Start the next line of the data file.
                self.output.write('\n')

                # Flushes the internal buffer, freeing the buffer.
                self.output.flush()

            # Run the next trial
            self.run_trial(t, response_card)

        # Check one last time to see if participant completed a category in their final trial.
        if self.rule_correct_streak == streak_rule_change:
            self.output.write('1,')  # Record '1' if category was completed this round.
        else:
            self.output.write('0,')  # Record '0' if category was not completed this round.

    def instruct(self, inst_text, go_text):
        """
        Displays the instruction text before beginning the session.
        :param inst_text: The text of the instructions.
        :param go_text: The text displayed just before the session begins.
        """
        inst = visual.TextStim(self.window, pos=(0, 0), height=0.04, units='height', alignHoriz='center', wrapWidth=1.0)
        inst.setText(inst_text)
        inst.draw()
        self.window.flip()

        # Initialize keys.
        keys_pressed = ['']
        key_press_count = 0

        # Wait for any key press. If key pressed is escape, quit early.
        while keys_pressed[0] not in ['escape', 'esc'] and key_press_count < 1:
            keys_pressed = event.waitKeys()
            if "escape" in keys_pressed:
                print("**********-----Session ended early-----**********")
                core.quit()
            key_press_count += 1

        inst.setText(go_text)
        inst.draw()
        self.window.flip()

        core.wait(instruction_time)  # Pause before starting the session.
        inst.setText('')
        inst.draw()
        self.window.flip()
        core.wait(0.5)  # Pause briefly before allowing the participant to act.

    def thank_you(self):
        """
        Displays the thank you text at the end of the session.
        """
        thank_you_visual = visual.TextStim(self.window, pos=(0, 0), height=1.4, units='deg')
        thank_you_visual.setText('Session concluded.\nThank you for participating!')
        thank_you_visual.draw()
        self.window.flip()
        core.wait(2)
        self.window.flip()

    def instruct_pause(self):
        """
        Display the cards on the screen to allow for verbal instructions provided by the experimenter.
        """
        inst = visual.TextStim(self.window, pos=(0, 0), height=0.05, alignHoriz='center', wrapWidth=22, units='height')
        choice = [[0, 1, 0], [1, 2, 1], [3, 3, 2], [2, 0, 3]]

        target = np.random.randint(4, size=3)

        # Display problem
        for i in range(4):
            self.cards[i].draw()
            self.elems[i].setColors(COLORS[choice[i][0]])
            self.elems[i].setMask(SHAPES[choice[i][1]])
            self.elems[i].setXYs(SHAPE_POS[choice[i][2]])
            self.elems[i].draw()
        self.window.flip()
        k = ['']
        KeyBoardCount = 0
        while k[0] not in ['escape', 'esc'] and KeyBoardCount < 1:
            k = event.waitKeys()
            KeyBoardCount += 1

        inst.setText('The test is about to begin...')
        inst.draw()
        self.window.flip()
        core.wait(1)
        inst.setText('')
        inst.draw()
        self.window.flip()
        core.wait(1)


# Initialize the masks for the various shapes.
N = 128
mid = N / 2 - 0.5

CIRCLE = np.ones((N, N)) * -1
CIRCLE = draw_circle(CIRCLE, (mid, mid), N / 2)
CROSS = np.ones((N, N)) * -1
w = N / 4
for i in range(CROSS.shape[0]):
    for j in range(CROSS.shape[1]):
        if mid - w / 2 < i < mid + w / 2 or mid - w / 2 < j < mid + w / 2:
            CROSS[i, j] = 1
STAR = np.ones((N, N)) * -1
STAR = draw_star(STAR, (mid, mid), 5, N / 2, N / 5)
TRIANGLE = draw_triangle(np.ones((N, N)) * -1, [[0, 0], [N, N / 2 - 0.5], [0, N]])

# Set default screen size in terms of resolution.
screen_resolution = (1024, 768)

# Other settings
KEYBOARD_SELECTION_ON = False
RESPONSE_POS = (0, -0.3)  # (0,-9) # Response card position.
TITLE_POS = (0, 0.45)
FEEDBACK_POS = (0, 0.15)  # (0,3.8) # Feedback position.
CARD_Y = 0.3  # 9 # Cards' vertical position.
PREV_CARD_Y = 0
DISCARD_POS = (0, PREV_CARD_Y)  # Discard pile position.
CARD_X = 0.05  # 1 # Horizontal space between cards.
CARD_W = 0.15  # 4 # Card width.
CARD_H = 0.225  # 6 # Card height.
COLORS = ['red', 'green', 'blue', 'orange']
SHAPES = [CIRCLE, TRIANGLE, STAR, CROSS]

SHAPE_POS = [[[0, 0], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]],
             [[0, CARD_H / 4.0], [0, -CARD_H / 4.0], [np.nan, np.nan], [np.nan, np.nan]],
             [[-CARD_W / 4.0, CARD_H / 3.0], [-CARD_W / 4.0, -CARD_H / 3.0], [CARD_W / 4.0, 0], [np.nan, np.nan]],
             [[-CARD_W / 4.0, CARD_H / 4.0], [-CARD_W / 4.0, -CARD_H / 4.0],
              [CARD_W / 4.0, -CARD_H / 4.0], [CARD_W / 4.0, CARD_H / 4.0]]]

# #################

# Start the practice trials first.
# E = Experiment()
# E.output = open(data_file, 'w')

# Write the practice round column headers.
# E.output.write('TrialNum,Card,Rule,RespTime,Correct,')
# E.output.write('Card01Color,Card01Shape,Card01Count,')
# E.output.write('Card02Color,Card02Shape,Card02Count,')
# E.output.write('Card03Color,Card03Shape,Card03Count,')
# E.output.write('Card04Color,Card04Shape,Card04Count,')
# E.output.write('ResponseColor,Responsehape,ResponseCount\n')

# Run some practice trials.
# E.instruct(instructions + ' practice.', 'Starting the practice...')
# E.run(num_trials=12, streak_rule_change=3)

# Run the experiment session.
E = Experiment()
E.output = open(data_file, 'w')  # Change 'w' to 'a' for append mode.
E.output.write('TrialNum,Card,Rule,RespTime,')
E.output.write('StartX, StartY, EndX, EndY,MouseVeloc,Misclicks,Error,Streak,')
E.output.write('ResponseColor,ResponseShape,ResponseCount,')
E.output.write('CategoryComplete\n')
E.instruct(instructions, 'Starting the test...')
E.instruct_pause()

# Begin the experiment session with the given number of trials.
E.run(num_trials=48, streak_rule_change=6)

# Conclude the experiment session.
E.thank_you()
E.output.close()
E.window.close()
