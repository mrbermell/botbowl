# Reinforcement Learning I: OpenAI Gym Environment
This tutorial will introduce you to FFAI's implementations of the [Open AI Gym interface](https://arxiv.org/pdf/1606.01540.pdf) that will allow for easy integration of reinforcement learning algorithms. 

You can run [examples/gym.py](examples/gym.py) to se a random agent play Blood Bowl through the FFAI Gym environment. The rendering is simplified for faster execution and looks like this:
![FFAI Gym GUI](https://njustesen.github.io/ffai/img/gym.png?raw=true "FFAI Gym GUI FFAI-3")

[examples/gym.py](examples/gym.py) demonstrated how you can run multiple instance of the environment in parallel. Notice, that the render() function doesn't work across multiple processes. Instead a custom renderer is used in this example.

Agents receive numerical observations from the FFAI environment at every step and sends back and action with an action type and in some cases a position. Along with the observations, the environment also sends a scalar reward value to the agent. We will describe the structure of the three components: observations, actions, and rewards.

## Observations
An observation object is a dictionary containing four differet parts:
1. 'board': a list of two-dimensional feature leayers describing the board state.
2. 'state': a vector of normalized values (e.g. turn number, half, scores, etc.) describing the non-spatial game state.
3. 'procedures': a one-hot vector describing which of the 16 procedures the game is currently in. 
4. 'available-action-types': a one-hot vector describing which actions types that are available.

### Observation: 'board'
The default feature layers in obs['board'] are:

0. OccupiedLayer()
1. OwnPlayerLayer()
2. OppPlayerLayer()
3. OwnTackleZoneLayer()
4. OppTackleZoneLayer()
5. UpLayer()
6. StunnedLayer()
7. UsedLayer()
8. AvailablePlayerLayer()
9. AvailablePositionLayer()
10. RollProbabilityLayer()
11. BlockDiceLayer()
12. ActivePlayerLayer()
13. TargetPlayerLayer()
14. MALayer()
15. STLayer()
16. AGLayer()
17. AVLayer()
18. MovemenLeftLayer()
19. BallLayer()
20. OwnHalfLayer()
21. OwnTouchdownLayer()
22. OppTouchdownLayer()
23. SkillLayer(Skill.BLOCK)
24. SkillLayer(Skill.DODGE)
25. SkillLayer(Skill.SURE_HANDS)
26. SkillLayer(Skill.CATCH)
27. SkillLayer(Skill.PASS)

A layer is a 2-D array of scalars in [0,1] with the size of the board including __crowd__ padding. Some layers have binary values, e.g. indicating whether a square is occupied by player (```OccupiedLayer()```), a standing player (```UpLayer()```), or a player with the __Block__ skill (```SkillLayer(Skill.BLOCK)```). Other layers contain normalized values such as ```OwnTackleZoneLayer()``` that represents the number of frendly tackle zones squares are covered by divided by 8, or ```MALayer()``` where the values are equal to the movement allowence of players divided by 10.

FfAI environments have the above 45 layers by defaults. Custom layers can, however, be implemented by implementing the ```FeatureLayer```:

```python
from ffai.ai import FeatureLayer
class MyCustomLayer(FeatureLayer):

    def produce(self, game):
        out = np.zeros((game.arena.height, game.arena.width))
        for y in range(len(game.state.pitch.board)):
            for x in range(len(game.state.pitch.board[0])):
                player = game.state.pitch.board[y][x]
                out[y][x] = 1.0 if player is not None and player.role.cost > 80000 else 0.0
        return out

    def name(self):
        return "expensive players"
```

Layers can then be added to an environment like this this:

```python
env.layers.append(MyCustomLayer())
```

To visualize the feature layers, use the ```feature_layers``` option when calling ```render()```:

```python
env.render(feature_layers=True)
```

![FFAI Gym Feature Layers](img/gym_layers.png?raw=true "FFAI Gym Feature Layers")

### Observation: 'state'
The 'state' part of the observation contains normailized values for folliwng 50  features:

0. 'half'
1. 'round'
2. 'is sweltering heat'
3. 'is very sunny'
4. 'is nice'
5. 'is pouring rain'
6. 'is blizzard'
7. 'is own turn'available_positions
8. 'is kicking first half'
9. 'is kicking this drive'
10. 'own reserves'
11. 'own kods'
12. 'own casualites'
13. 'opp reserves'
14. 'opp kods'
15. 'opp casualties'
16. 'own score'
17. 'own turns'
18. 'own starting rerolls'
19. 'own rerolls left'
20. 'own ass coaches'
21. 'own cheerleaders'
22. 'own bribes'
23. 'own babes'
24. 'own apothecary available'
25. 'own reroll available'
26. 'own fame'
27. 'opp score'
28. 'opp turns'
29. 'opp starting rerolls'
30. 'opp rerolls left'
31. 'opp ass coaches'
32. 'opp cheerleaders'
33. 'opp bribes'
34. 'opp babes'
35. 'opp apothecary available'
36. 'opp reroll available'
37. 'opp fame'
38. 'is blitz available'
39. 'is pass available'
40. 'is handoff available'
41. 'is foul available'
42. 'is blitz'
43. 'is quick snap'
44. 'is move action'
45. 'is block action'
46. 'is blitz action'
47. 'is pass action'
48. 'is handoff action'
49. 'is foul action'

Some values are boolean, either 0 or 1, while others are normalized.

### Observation: 'procedure'
The 19 procedures represented in the one-hot vector obs['procedure'] are:

0. StartGame
1. CoinTossFlip
2. CoinTossKickReceive
3. Setup
4. PlaceBall
5. HighKick
6. Touchback
7. Turn
8. PlayerAction
9. Block
10. Push
11. FollowUp
12. Apothecary
13. PassAction
14. Catch
15. Interception
16. GFI
17. Dodge
18. Pickup

## Action Types
Actions consists of 31 action types. Some action types, denoted by `<position>` also requires an x and y-coordinate.

0. ActionType.START_GAME
1. ActionType.HEADS
2. ActionType.TAILS
3. ActionType.KICK
4. ActionType.RECEIVE
5. ActionType.END_PLAYER_TURN
6. ActionType.USE_REROLL
7. ActionType.DONT_USE_REROLL
8. ActionType.END_TURN
9. ActionType.STAND_UP
10. ActionType.SELECT_ATTACKER_DOWN
11. ActionType.SELECT_BOTH_DOWN
12. ActionType.SELECT_PUSH
13. ActionType.SELECT_DEFENDER_STUMBLES
14. ActionType.SELECT_DEFENDER_DOWN
15. ActionType.SELECT_NONE
16. ActionType.PLACE_PLAYER`<Position>`
17. ActionType.PLACE_BALL`<Position>`
18. ActionType.PUSH`<Position>`
19. ActionType.FOLLOW_UP`<Position>` 
20. ActionType.SELECT_PLAYER`<Position>` (position of the player)
21. ActionType.MOVE`<Position>`
22. ActionType.BLOCK`<Position>`
23. ActionType.PASS`<Position>`
24. ActionType.FOUL`<Position>`
25. ActionType.HANDOFF`<Position>`
24. ActionType.LEAP
25. ActionType.START_MOVE`<Position>` (position of the player)
26. ActionType.START_BLOCK`<Position>` (position of the player)
27. ActionType.START_BLITZ`<Position>` (position of the player)
28. ActionType.START_PASS`<Position>` (position of the player)
29. ActionType.START_FOUL`<Position>` (position of the player)
30. ActionType.START_HANDOFF`<Position>` (position of the player)
31. ActionType.USE_SKILL
32. ActionType.DONT_USE_SKILL
33. ActionType.SETUP_FORMATION_WEDGE
34. ActionType.SETUP_FORMATION_LINE
35. ActionType.SETUP_FORMATION_SPREAD
36. ActionType.SETUP_FORMATION_ZONE

### Observation: 'procedure'
The 'procedure' part of the observation contains a one-hot vector with 16 values representing which procedures the game is in:

0. ```StartGame```
1. ```CoinTossFlip```
2. ```CoinTossKickReceive```
3. ```Setup```
4. ```PlaceBall```
5. ```HighKick```
6. ```Touchback```
7. ```Turn```
8. ```PlayerAction```
9. ```Block```
10. ```Push```
11. ```FollowUp```
12. ```Apothecary```
13. ```PassAction```
14. ```Interception```
15. ```Reroll```

### Observation: 'available-action-types'
The 'available-action-types' part of the observation contains a one-hot vector describing which action types that are currently available.

0. ```ActionType.START_GAME```
1. ```ActionType.HEADS```
2. ```ActionType.TAILS```
3. ```ActionType.KICK```
4. ```ActionType.RECEIVE```
5. ```ActionType.END_PLAYER_TURN```
6. ```ActionType.USE_REROLL```
7. ```ActionType.DONT_USE_REROLL```
8. ```ActionType.END_TURN```
9. ```ActionType.STAND_UP```
10. ```ActionType.SELECT_ATTACKER_DOWN```
11. ```ActionType.SELECT_BOTH_DOWN```
12. ```ActionType.SELECT_PUSH```
13. ```ActionType.SELECT_DEFENDER_STUMBLES```
14. ```ActionType.SELECT_DEFENDER_DOWN```
15. ```ActionType.SELECT_NONE```
16. ```ActionType.PLACE_PLAYER <Position>```
17. ```ActionType.PLACE_BALL <Position>```
18. ```ActionType.PUSH <Position>```
19. ```ActionType.FOLLOW_UP <Position>```
20. ```ActionType.SELECT_PLAYER <Position>```
21. ```ActionType.MOVE <Position>```
22. ```ActionType.BLOCK <Position>```
23. ```ActionType.PASS <Position>```
24. ```ActionType.FOUL <Position>```
25. ```ActionType.HANDOFF`<Position>```
26. ```ActionType.LEAP`<Position>```
27. ```ActionType.STAB <Position>```
28. ```ActionType.START_MOVE <Position>```
29. ```ActionType.START_BLOCK <Position>```
30. ```ActionType.START_BLITZ <Position>```
31. ```ActionType.START_PASS <Position>```
32. ```ActionType.START_FOUL <Position>```
33. ```ActionType.START_HANDOFF <Position>```
34. ```ActionType.USE_SKILL```
35. ```ActionType.DONT_USE_SKILL```
36. ```ActionType.SETUP_FORMATION_WEDGE```
37. ```ActionType.SETUP_FORMATION_LINE```
38. ```ActionType.SETUP_FORMATION_SPREAD```
39. ```ActionType.SETUP_FORMATION_ZONE```

## Actions
To take an action, the step function must be called with an Action instance that contains an action type and a position if needed. See the list above whether an actions needs a position. Actions are instantiated and used like this:

```python
action = {
    'action-type': 26,
    'x': 8,
    'y': 6
}
obs, reward, done, info = env.step(action)
```

You can always check if an action type is available using ```env.available_action_types()``` and for positions ```available_positions(action_type)```. The same information is available through ```obs['available-action-types']``` and ```obs['board']['<action_type> positions']``` where ```<action_type>``` e.g. could be `move`.

## Rewards and Info
The default reward function only rewards for a win, draw or loss 1/0/-1.
However, the info object returned by the step function contains useful information for reward shaping:

```python
'cas_inflicted': {int},
'opp_cas_inflicted': {int},
'touchdowns': {int},
'opp_touchdowns': {int},
'half': {int},
'round': {int},
'ball_progression': {int}
```

These values are commulative, such that 'cas_inflicted' refers to the total number of casualties inflicted by the team in the game. Another way to detect events is looking at ```env.game.state.reports```.

## Environments
FFAI comes with five environments with various difficulty:

- **FFAI-v2:** 11 players on a 26x15 pitch (traditional size)
- **FFAI-7-v2:** 7 players on a 20x11 pitch
- **FFAI-5-v2:** 5 players on a 16x9 pitch
- **FFAI-3-v2:** 3 players on a 12x5 pitch
- **FFAI-1-v2:** 1 player on a 4x3 pitch

![A rendering of __FFAI-3-v2__.](img/gym_3.png?raw=true "A rendering of __FFAI-3-v2__.")

## Explore the Observation Space
Try running [examples/gym.py](examples/gym.py) while debugging in your favorite IDE (e.g. [PyCharm](https://www.jetbrains.com/pycharm/)). Set a break point in the line where the step function is called and investigate the obs object. If you run with the rendering enabled it is easier to analyze the values in the feature layers.

In the next tutorial, we will start developing a reinforcement learning agent.
