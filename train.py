#Trains the chess network


#read the training data

#preprocess training data, feed into queue
	#to preprocess:
		#get each board state and the subsequent move (the target)
		#get all pseudo-valid moves for that board (including no move)
		#evaluate ranking of each move using stockfish (or something else)
		#create 1 sample for each board-move pairing: 
			#get all features (including move rank) into some encoding (this is the input)
			#target is binary value based on whether the move was the target move (from real data or stockfish or other peoples bots)

#intialize network (network is a class from another file)


#for each batch taken from queue:
	#train network

	#periodically save network

	#allow training to be canceled

	#integrate with tensorboard??

#save final result
