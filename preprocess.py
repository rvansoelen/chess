#this takes the raw data and preprocesses it into features

#open the raw data

#get the boards and moves from each board

#augement data: (does this bias the representation of states??)
	#perform a time limited search on board states from raw data
	#randomly sample from this search to get dataset
	#label board state with best move, either from stockfish, raw data, or other person's bot

#partition data into test, validation, and train