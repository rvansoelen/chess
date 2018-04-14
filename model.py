#class file for the probability estimator network
#recieves features from board state and possible move and outputs the probability of that move (single bit)

import tensorflow as tf

class ProbNet: 

	def __init__(self):
		#define quantity and size of each hidden layer

		#define AdaDelta optimizer

		#create all layers of network including loss
		self.loss = createNetwork()

	def createNetwork():
		with tf.variable_scope('ProbNet'):
		#create input placeholders (should these be class variables?)
		#feed each group of variables into a separate hidden layer 
		#each hidden layer is fully connected with ReLU activation

			#global variables
			with tf.variable_scope('Global Variables'):
				side2move = #1 bit (everything is really just floats)
				castlingRights = #4 bits
				numOfEachPieceType = #array of 7 floats (should these be normalized? I think so)


				#global hidden layer


			#piece-centric variables
			with tf.variable_scope('Piece-Centric Variables'):
				pieceIsPresent = # array of 48 bits
				xyPosition = #array of 48 by 2 normalized floats (choose -1 if not present?)
				lowValAttacker = #array of 48 normalized floats (representing value of attacker??)
				lowValDefender = #array of 48 normalized floats (representing value of defender??)
				maxTravelDistance = #array of 14 floats (normalized)
				movePieceType = #one hot array of size 6
				pawnPromoType = #one hot array of size 7 (extra spot for when there is no pawn promotion, or should the vector be zero then?)

				#piece-centric hidden layer


			#square-centric variables
			with tf.variable_scope('Sqaure-Centric Variables'):

				#square-centric hidden layer

			#feed all three layers into a a singe hidden layer wtih ReLU activation
			with tf.variable_scope('Hidden Layer 2'):

			#repeat (if more than 2 hidden layers are used)

			#Output of last layer should be a single value, which is feed into a logistic actiation

			#Create target placeholder and feed output and target into cross-entropy loss

	def evaluate(self, boardState, move):
		#this function is used to evaluate the probability of a move given a board state
		#It is meant to only be used during evaluation, not training 

		#Two pass system

		#get all valid moves


		#for each move:
			#compute features of board assuming move is best move 
			#get network probability of each move

		#rank moves based on probability

		#run the network again on original move, this time with the rank set to its estimated value

		#return probability

		#