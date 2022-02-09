# tallon.py
# The code that defines the behaviour of Tallon. This is the place
# (the only place) where you should write code, using access methods
# from world.py, and using makeMove() to generate the next move.
# Written by: Simon Parsons
# Last Modified: 12/01/22
# Edited by: James Tombling
# Last Edited: 08/02/22
import numpy as np
import config
import mdptoolbox
from utils import Directions
class Tallon():
    def __init__(self, arena): # intiate
        self.gameWorld = arena
        self.maxX = config.worldLength
        self.maxY = config.worldBreadth
        self.mapsize = self.maxX*self.maxY
        # What moves are possible.
        self.moves = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]

    def makeMove(self): # define makeMove function
        mlocs = self.gameWorld.getMeanieLocation()  # meanie location
        plocs = self.gameWorld.getPitsLocation()  # pit location
        allBonuses = self.gameWorld.getBonusLocation() # all bonuses location
        myPosition = self.gameWorld.getTallonLocation() #
        Reward_Matrix = np.zeros((self.mapsize, 4)) # reward matrix
        f_prob = config.directionProbability  # forwards probability
        s_prob = (1 - config.directionProbability) / 2  # sideways probability
        Prob_Matrix = np.zeros((4, self.mapsize, self.mapsize)) # Probability matrix
        discount_factor = 0.5 # means how far it can see in the 'future', low = short , high = far in the future
        bad_value = -1  # negative value for pits and meanies in reward matrix
        good_value = 1  # positive value for bonuses in reward matrix
        edge_value = -0.5 # edge values for side and corners in reward matrix
        # north matrix = 0 , South matrix = 1 , east matrix = 2 , West matrix = 3
        # fill Probability matrix
        for j in range(4):  # iterate through each direction north, south, east, west
            for i in range(self.mapsize): # Prob matrix
                if i == 0:  # top left corner of map
                    if j == 0:    # north direction matrix when j = 0
                        Prob_Matrix[j, i, i] = f_prob + s_prob
                        Prob_Matrix[j, i, i + 1] = s_prob
                    elif j == 1:  # south direction matrix  when j = 1
                        Prob_Matrix[j, i, i + self.maxX] = f_prob
                        Prob_Matrix[j, i, i + 1] = s_prob
                        Prob_Matrix[j, i, i] = s_prob
                    elif j == 2:  # east direction matrix when j = 2
                        Prob_Matrix[j, i, i + self.maxX] = s_prob
                        Prob_Matrix[j, i, i + 1] = f_prob
                        Prob_Matrix[j, i, i] = s_prob
                    elif j == 3:  # west direction matrix when j = 3
                        Prob_Matrix[j, i, i] = f_prob + s_prob
                        Prob_Matrix[j, i, i + self.maxX] = s_prob
                        Reward_Matrix[i] = [edge_value, edge_value, edge_value, edge_value] # make the edges and corner negative value
                elif i == self.maxX - 1:  # Top right corner
                    if j == 0:    # north direction matrix when j = 0
                        Prob_Matrix[j, i, i] = f_prob + s_prob
                        Prob_Matrix[j, i, i - 1] = s_prob
                    elif j == 1:  # south direction matrix  when j = 1
                        Prob_Matrix[j, i, i] = s_prob
                        Prob_Matrix[j, i, i - 1] = s_prob
                        Prob_Matrix[j, i, i + self.maxX] = f_prob
                    elif j == 2:  # east direction matrix when j = 2
                        Prob_Matrix[j, i, i] = f_prob + s_prob
                        Prob_Matrix[j, i, i + self.maxX] = s_prob
                    elif j == 3:  # west direction matrix when j = 3
                        Prob_Matrix[j, i, i] = s_prob
                        Prob_Matrix[j, i, i - 1] = f_prob
                        Prob_Matrix[j, i, i + self.maxX] = s_prob
                        Reward_Matrix[i] = [edge_value, edge_value, edge_value, edge_value]
                elif i == (self.mapsize - 1) - (self.maxX - 1):  # bottom left corner of map
                    if j == 0:    # north direction matrix when j = 0
                        Prob_Matrix[j, i, i - self.maxX] = f_prob
                        Prob_Matrix[j, i, i + 1] = s_prob
                        Prob_Matrix[j, i, i] = s_prob
                    elif j == 1:  # south direction matrix  when j = 1
                        Prob_Matrix[j, i, i + 1] = s_prob
                        Prob_Matrix[j, i, i] = f_prob + s_prob
                    elif j == 2:  # east direction matrix when j = 2
                        Prob_Matrix[j, i, i - self.maxX] = s_prob
                        Prob_Matrix[j, i, i + 1] = f_prob
                        Prob_Matrix[j, i, i] = s_prob
                    elif j == 3:  # west direction matrix when j = 3
                        Prob_Matrix[j, i, i - self.maxX] = s_prob
                        Prob_Matrix[j, i, i] = f_prob + s_prob
                        Reward_Matrix[i] = [edge_value, edge_value, edge_value, edge_value]
                elif i == (self.mapsize - 1):  # bottom right corner of map
                    if j == 0:    # north direction matrix when j = 0
                        Prob_Matrix[j, i, i - self.maxX] = f_prob
                        Prob_Matrix[j, i, i - 1] = s_prob
                        Prob_Matrix[j, i, i] = s_prob
                    elif j == 1:  # south direction matrix  when j = 1
                        Prob_Matrix[j, i, i - 1] = s_prob
                        Prob_Matrix[j, i, i] = f_prob + s_prob
                    elif j == 2:  # east direction matrix when j = 2
                        Prob_Matrix[j, i, i - self.maxX] = s_prob
                        Prob_Matrix[j, i, i] = f_prob + s_prob
                    elif j == 3:  # west direction matrix when j = 3
                        Prob_Matrix[j, i, i - self.maxX] = s_prob
                        Prob_Matrix[j, i, i - 1] = f_prob
                        Prob_Matrix[j, i, i] = s_prob
                        Reward_Matrix[i] = [edge_value, edge_value, edge_value, edge_value]
                elif i < self.maxX - 1:  # north side of map
                    if j == 0:    # north direction  matrix when j = 0
                        Prob_Matrix[j, i, i] = f_prob
                        Prob_Matrix[j, i, i - 1] = s_prob
                        Prob_Matrix[j, i, i + 1] = s_prob
                    elif j == 1:  # south direction matrix  when j = 1
                        Prob_Matrix[j, i, i + self.maxX] = f_prob
                        Prob_Matrix[j, i, i - 1] = s_prob
                        Prob_Matrix[j, i, i + 1] = s_prob
                    elif j == 2:  # east direction matrix when j = 2
                        Prob_Matrix[j, i, i + self.maxX] = s_prob
                        Prob_Matrix[j, i, i] = s_prob
                        Prob_Matrix[j, i, i + 1] = f_prob
                    elif j == 3:  # west direction matrix when j = 3
                        Prob_Matrix[j, i, i + self.maxX] = s_prob
                        Prob_Matrix[j, i, i - 1] = f_prob
                        Prob_Matrix[j, i, i] = s_prob
                        Reward_Matrix[i] = [edge_value, edge_value, edge_value, edge_value]
                elif i % self.maxX == 0:  # West side of map
                    if j == 0:    # north direction matrix when j = 0
                        Prob_Matrix[j, i, i - self.maxX] = f_prob
                        Prob_Matrix[j, i, i] = s_prob
                        Prob_Matrix[j, i, i + 1] = s_prob
                    elif j == 1:  # south direction matrix  when j = 1
                        Prob_Matrix[j, i, i + self.maxX] = f_prob
                        Prob_Matrix[j, i, i] = s_prob
                        Prob_Matrix[j, i, i + 1] = s_prob
                    elif j == 2:  # east direction matrix when j = 2
                        Prob_Matrix[j, i, i - self.maxX] = s_prob
                        Prob_Matrix[j, i, i + self.maxX] = s_prob
                        Prob_Matrix[j, i, i + 1] = f_prob
                    elif j == 3:  # west direction matrix when j = 3
                        Prob_Matrix[j, i, i - self.maxX] = s_prob
                        Prob_Matrix[j, i, i] = f_prob
                        Prob_Matrix[j, i, i + self.maxX] = s_prob
                        Reward_Matrix[i] = [edge_value, edge_value, edge_value, edge_value]
                elif (i+1) % self.maxX == 0:  # East side of map
                    if j == 0:    # north direction matrix when j = 0
                        Prob_Matrix[j, i, i - self.maxX] = f_prob
                        Prob_Matrix[j, i, i] = s_prob
                        Prob_Matrix[j, i, i - 1] = s_prob
                    elif j == 1:  # south direction matrix  when j = 1
                        Prob_Matrix[j, i, i + self.maxX] = f_prob
                        Prob_Matrix[j, i, i] = s_prob
                        Prob_Matrix[j, i, i - 1] = s_prob
                    elif j == 2:  # east direction matrix when j = 2
                        Prob_Matrix[j, i, i - self.maxX] = s_prob
                        Prob_Matrix[j, i, i] = f_prob
                        Prob_Matrix[j, i, i + self.maxX] = s_prob
                    elif j == 3:  # west direction matrix when j = 3
                        Prob_Matrix[j, i, i - self.maxX] = s_prob
                        Prob_Matrix[j, i, i + self.maxX] = s_prob
                        Prob_Matrix[j, i, i - 1] = f_prob
                        Reward_Matrix[i] = [edge_value, edge_value, edge_value, edge_value]
                elif (self.mapsize - (self.maxX -1)) <= i <= (self.mapsize-1):  # south side of map
                    if j == 0:    # north direction matrix when j = 0
                        Prob_Matrix[j, i, i - self.maxX] = f_prob
                        Prob_Matrix[j, i, i + 1] = s_prob
                        Prob_Matrix[j, i, i - 1] = s_prob
                    elif j == 1:  # south direction matrix  when j = 1
                        Prob_Matrix[j, i, i] = f_prob
                        Prob_Matrix[j, i, i + 1] = s_prob
                        Prob_Matrix[j, i, i - 1] = s_prob
                    elif j == 2:  # east direction matrix when j = 2
                        Prob_Matrix[j, i, i - self.maxX] = s_prob
                        Prob_Matrix[j, i, i + 1] = f_prob
                        Prob_Matrix[j, i, i] = s_prob
                    elif j == 3:  # west direction matrix when j = 3
                        Prob_Matrix[j, i, i - self.maxX] = s_prob
                        Prob_Matrix[j, i, i] = s_prob
                        Prob_Matrix[j, i, i - 1] = f_prob
                        Reward_Matrix[i] = [edge_value, edge_value, edge_value, edge_value]
                else:  # rest of the map (anything but th edges)
                    if j == 0:    # north direction matrix when j = 0
                        Prob_Matrix[j, i, i - self.maxX] = f_prob
                        Prob_Matrix[j, i, i - 1] = s_prob
                        Prob_Matrix[j, i, i + 1] = s_prob
                    elif j == 1:  # south direction matrix  when j = 1
                        Prob_Matrix[j, i, i + self.maxX] = f_prob
                        Prob_Matrix[j, i, i - 1] = s_prob
                        Prob_Matrix[j, i, i + 1] = s_prob
                    elif j == 2:  # east direction matrix when j = 2
                        Prob_Matrix[j, i, i - self.maxX] = s_prob
                        Prob_Matrix[j, i, i + self.maxX] = s_prob
                        Prob_Matrix[j, i, i + 1] = f_prob
                    elif j == 3:  # west direction matrix when j = 3
                        Prob_Matrix[j, i, i - self.maxX] = s_prob
                        Prob_Matrix[j, i, i + self.maxX] = s_prob
                        Prob_Matrix[j, i, i - 1] = f_prob

        # fill reward matrix
        for i in range(len(allBonuses)): # fill the bonus rewards
            for j in range(4):
                bonusx = allBonuses[i].x
                bonusy = allBonuses[i].y
                bonus = (bonusx) + (bonusy * self.maxX)
                Reward_Matrix[bonus] = [good_value, good_value, good_value, good_value]
        for i in range(len(mlocs)): # create meanie zone in reward matrix
            for j in range(4):
                meaniex = mlocs[i].x
                meaniey = mlocs[i].y
                meanie = (meaniex) + (meaniey * self.maxX)
                Reward_Matrix[meanie] = [bad_value, bad_value, bad_value, bad_value]
        for i in range(len(plocs)): # create the pits zones in the reward matrix
            for j in range(4):
                pitx = plocs[i].x
                pity = plocs[i].y
                pit = (pitx) + (pity * self.maxX)
                Reward_Matrix[pit] = [bad_value, bad_value, bad_value, bad_value]

        # This is the function you need to define for the markov decision process
        # Policy iteration generation
        mdptoolbox.util.check(Prob_Matrix, Reward_Matrix)
        pi = mdptoolbox.mdp.PolicyIteration(Prob_Matrix, Reward_Matrix, discount_factor)
        pi.run()
        tallon = (myPosition.x) + ((myPosition.y) * self.maxX)
        return(self.moves[pi.policy[tallon]])

