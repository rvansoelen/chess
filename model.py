#class file for the probability estimator network
#recieves features from board state and possible move and outputs the probability of that move (single bit)

import tensorflow as tf

class ProbNet: 

	def __init__(self):
		#define quantity and size of each hidden layer
		self.gHidden1OutputSize = 
		self.pHidden1OutputSize = 
		self.sHidden1OutputSize =
		self.hidden2OutputSize = 

		#define AdaDelta optimizer

		#create all layers of network including loss
		self.loss = createNetwork()

	def createWeight(self, name, shape='None'):
		initializer = tf.contrib.layers.variance_scaling_initializer()
		return tf.get_variable(name, shape, dtype='float32')

	def createNetwork():
		with tf.variable_scope('ProbNet'):
		#create input placeholders (should these be class variables?)
		#feed each group of variables into a separate hidden layer 
		#each hidden layer is fully connected with ReLU activation

			#global variables
			with tf.variable_scope('Global Variables'):
				self.side2move = tf.placeholder(tf.bool, shape=[None, 1], name='Side To Move')#1 bit (everything is really just floats)
				self.castlingRights = tf.placeholder(tf.bool, shape=[None, 4], name='Castling Rights')#4 bits
				self.numOfEachPieceType = tf.placeholder(tf.float32, shape=[None, 10], name = 'Number of Each Piece Type')#array of 10 floats, excluding kings (should these be normalized? I think so)

				globalFeats = tf.concat([self.side2move, self.castlingRights, self.numOfEachPieceType], 1, name='Global Features')
				
				#global hidden layer
				#gWeights1 =  self.createWeight('Weights 1')
				#gBais1 = self.createWeight('Bias 1')
				gHidden1 = tf.layers.dense(globalFeats, self.gHidden1OutputSize, name='Global Hidden Layer 1', activation='relu', kernel_initializer=tf.contrib.layers.variance_scaling_initializer())


			#piece-centric variables
			with tf.variable_scope('Piece-Centric Variables'):
				self.pieceIsPresent = # array of 48 bits
				self.xyPosition = #array of 48 by 2 normalized floats (choose -1 if not present?)
				self.pieceLowValAttacker = #array of 48 normalized floats (representing value of attacker??)
				self.pieceLowValDefender = #array of 48 normalized floats (representing value of defender??)
				self.maxTravelDistance = #array of 14 floats (normalized)
				self.movePieceType = #one hot array of size 6
				self.pawnPromoType = #one hot array of size 7 (extra spot for when there is no pawn promotion, or should the vector be zero then?)

				pieceFeats = tf.concat([self.pieceIsPresent, self.xyPosition, self.pieceLowValAttacker, 
					self.pieceLowValDefender, self.maxTravelDistance, self.maxTravelDistance, self.movePieceType, 
					self.pawnPromoType], 1, name='Piece-Centric Features')

				#piece-centric hidden layer
				#gWeights1 =  self.createWeight('Weights 1')
				#gBais1 = self.createWeight('Bias 1')
				pHidden1 = tf.layers.dense(pieceFeats, self.pHidden1OutputSize, name='Global Hidden Layer 1', activation='relu', kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

			#square-centric variables
			with tf.variable_scope('Sqaure-Centric Variables'):
				squareLowValAttacker = #12 by 12 array of lowest value attacker (float or one hot?)
				squareLowValDefender = #12 by 12 array of lowest value defender (float or one hot?)
				moveStartingSquare = #two floats (x and y)
				moveEndingSquare = #two floats (x and y)

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