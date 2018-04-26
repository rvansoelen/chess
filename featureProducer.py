import numpy as np
import chess
import chess.uci

def getNormX(position):
    return (position%8.0)/7.0
def getNormY(position):
    return float(position/8)/7.0

def calcGlobalFeatures(board):
    #side2Move
    #THIS IS PASSED DIRECTLY TO SECOND LAYER
    side2Move = [float(board.turn)]
    
    #king positions (this is covered elsewhere, but in a different group)
    #THIS IS PASSED DIRECTLY TO SECOND LAYER
    kingPosition = [getNormX(board.king(chess.WHITE)), 
        getNormY(board.king(chess.WHITE)), 
        getNormX(board.king(chess.BLACK)),
        getNormY(board.king(chess.BLACK))]
    
    #castlingRights
    #THIS IS PASSED DIRECTLY TO SECOND LAYER (I think)
    castlingRights = [float(board.has_kingside_castling_rights(chess.WHITE)), 
        float(board.has_queenside_castling_rights(chess.WHITE)), 
        float(board.has_kingside_castling_rights(chess.BLACK)), 
        float(board.has_queenside_castling_rights(chess.BLACK))]
    
    #numOfEachPieceType 
    #THIS IS PASSED DIRECTLY TO SECOND LAYER
    numOfEachPieceType = [len(board.pieces(chess.PAWN, chess.WHITE))/8.0, 
        len(board.pieces(chess.KNIGHT, chess.WHITE))/2.0,
        len(board.pieces(chess.BISHOP, chess.WHITE))/2.0,
        len(board.pieces(chess.ROOK, chess.WHITE))/2.0,
        len(board.pieces(chess.QUEEN, chess.WHITE))/1.0,
        len(board.pieces(chess.PAWN, chess.BLACK))/8.0, 
        len(board.pieces(chess.KNIGHT, chess.BLACK))/2.0,
        len(board.pieces(chess.BISHOP, chess.BLACK))/2.0,
        len(board.pieces(chess.ROOK, chess.BLACK))/2.0,
        len(board.pieces(chess.QUEEN, chess.BLACK))/1.0]
    return np.array(side2Move+kingPosition+castlingRights+numOfEachPieceType, dtype='float32')

#see variable SEE::SEE_MAT for values (WK is white king i think)
#NormalizeCount(SEE::SEE_MAT[WK] + SEE::SEE_MAT[WK] / 2 - SEE::SEE_MAT[whitePt], SEE::SEE_MAT[WK] * 2)
'''
static const Score SEE_MAT[14] = {
    1500, // WK
    975, // WQ
    500, // WR
    325, // WN
    325, // WB
    100, // WP
}
'''

def getLowestValueAttackerScore(board, position, color):
    attackers = board.attackers(color, position)
    #determine the lowest value attacker
    if not bool(attackers): #no attackers
        value = 2250.0 #makes the score 0
    elif bool(attackers & board.pieces(chess.PAWN, color)):
        value = 100.0
    elif bool(attackers & board.pieces(chess.BISHOP, color)):
        value = 325.0
    elif bool(attackers & board.pieces(chess.KNIGHT, color)):
        value = 325.0
    elif bool(attackers & board.pieces(chess.ROOK, color)):
        value = 500.0
    elif bool(attackers & board.pieces(chess.QUEEN, color)):
        value = 975.0
    else: #Must be a king attacker
        value = 1500.0
    return (2250.0 - value)/3000.0

def calcSquareFeatures(board):
    feats = []
    for position in range(64):
        feats.append(getLowestValueAttackerScore(board, position, chess.WHITE))
        feats.append(getLowestValueAttackerScore(board, position, chess.BLACK))
    return np.array(feats, dtype='float32')

def getSinglePieceFeatures(board, position, exists=True):
    #return array of pieceExists, XYposition, and attackers and defenders
    if exists:
        whiteAttackerScore = getLowestValueAttackerScore(board, 
                                 position, chess.WHITE)
        blackAttackerScore = getLowestValueAttackerScore(board, 
                                 position, chess.BLACK)
        feats = [1.0, getNormX(position), getNormY(position), 
                    whiteAttackerScore, blackAttackerScore]
    else:
        feats = [0.0, 0.0, 0.0, 0.0, 0.0]
    return feats
        
def getPawnFeatures(board, color):
    pawns = board.pieces(chess.PAWN, color)
    #Assign each pawn to a slot based on X position
    slots = []
    unassignedSlots = []
    unassignedPieces = []
    for x in range(8):
        column = pawns & chess.BB_FILES[x]
        if len(column) == 1: #exactly one pawn in column
            position = column.pop()
            slots.append(getSinglePieceFeatures(board, position))
        elif len(column) == 0: #no pawns in column
            slots.append(None)
            unassignedSlots.append(x)
        else: #more than one pawn in column
            position = column.pop()
            slots.append(getSinglePieceFeatures(board, position))
            unassignedPieces.extend(list(column))
    #match unpaired slots to pawns 
    for x in unassignedSlots:
        #make sure there are still extra pawns
        if len(unassignedPieces) > 0:
            #find closest pawn position
            position = min(unassignedPieces, key=lambda pos : abs(x-pos%8))
            slots[x] = getSinglePieceFeatures(board, position)
            unassignedPieces.remove(position)
        else:
            slots[x] = getSinglePieceFeatures(None, None, exists=False)
       
    #any extra pawns after this don't have a slot and are ignored
    #combine slots into a single array and return it

    return [feature for slot in slots for feature in slot]

def getPairPieceFeatures(board, pieceType, color):
    #for rooks, bishops, and knights, there are only 2 slots
    pieces = board.pieces(pieceType, color)
    if len(pieces) == 0:
        feats = getSinglePieceFeatures(None, None, exists=False)*2
    elif len(pieces) == 1:
        position = pieces.pop()
        emptyFeatures = getSinglePieceFeatures(None, None, exists=False)
        existingFeatures = getSinglePieceFeatures(board, position)
        if position%8 <4:
            feats = existingFeatures + emptyFeatures
        else:
            feats = emptyFeatures + existingFeatures
    else:
        position1 = pieces.pop()
        position2 = pieces.pop()
        feats1 = getSinglePieceFeatures(board, position1)
        feats2 = getSinglePieceFeatures(board, position2)
        if position1%8 < position2%8:
            feats = feats1 + feats2
        else:
            feats = feats2 + feats1
    return feats
    
def getQueenFeatures(board, color):
    queens = board.pieces(chess.QUEEN, color)
    if len(queens) > 0:
        position = queens.pop()
        feats = getSinglePieceFeatures(board, position)
    else:
        feats = getSinglePieceFeatures(None, None, exists=False)
    return feats
    
def getKingFeatures(board, color): #maybe not needed
    kings = board.pieces(chess.KING, color)
    if len(kings) > 0:
        position = kings.pop()
        feats = getSinglePieceFeatures(board, position)
    else: #this should never happen
        feats = getSinglePieceFeatures(None, None, exists=False)
    return feats

#the main function for this cell:
def calcPieceFeatures(board):
    pairPieces = [chess.ROOK, chess.BISHOP, chess.KNIGHT]
    pairPiecesFeatures = []
    for pieceType in pairPieces:
        pairPiecesFeatures.extend(getPairPieceFeatures(board, pieceType, chess.WHITE))
        pairPiecesFeatures.extend(getPairPieceFeatures(board, pieceType, chess.BLACK))
    pieceFeatures = np.array(getPawnFeatures(board, chess.WHITE)
                     +getPawnFeatures(board, chess.BLACK)
                     +pairPiecesFeatures
                     +getQueenFeatures(board, chess.WHITE)
                     +getQueenFeatures(board, chess.BLACK)
                     +getKingFeatures(board, chess.WHITE)
                     +getKingFeatures(board, chess.BLACK), dtype='float32')
    
    return pieceFeatures

def checkRestartEngine(force=False):
    if('engine' not in globals()):
        global engine 
        engine = chess.uci.popen_engine("stockfish")
        engine.uci()
        #engine.debug(True)
        global infoHandler
        infoHandler = chess.uci.InfoHandler()
        engine.info_handlers.append(infoHandler)
        global computeTime
        computeTime = 50
    if force or not engine.is_alive():
        engine = chess.uci.popen_engine("stockfish")
        engine.uci()
        #engine.debug(True)
        infoHandler = chess.uci.InfoHandler()
        engine.info_handlers.append(infoHandler)
        
def getBestMove(board):
    checkRestartEngine()
    engine.ucinewgame()
    engine.position(board)
    possibleMoves = list(board.pseudo_legal_moves)
    possibleMoves.append(chess.Move(0, 0))
    try:
        bestMove = engine.go(searchmoves=possibleMoves, movetime=computeTime)[0]
        engine.stop()
        if bestMove == None: bestMove = chess.Move(0,0)
    except chess.uci.EngineTerminatedException as e:
        #print('Error processing board for best move:')
        #print(board.fen())
        #print(board)
        #print(infoHandler.info)
        #print(e)
        #print('')
        checkRestartEngine(force=True)
        bestMove = None #random.choice(possibleMoves)
    return bestMove

def calcLabel(board, move, bestMove):
    return np.array([bestMove == move], dtype='float32')

def calcMoveRankings(board):
    moves = list(board.pseudo_legal_moves)
    moves.append(chess.Move(0, 0))
    scores = {}
    for move in moves:
        checkRestartEngine()
        engine.ucinewgame()
        engine.position(board)
        try:
            engine.go(searchmoves=[move], movetime=computeTime)[0]
            engine.stop()
            scores[move] = infoHandler.info['score'][1].cp
        except chess.uci.EngineTerminatedException as e:
            #print('Error processing board and move ranking:')
            #print(board.fen())
            #print(board)
            #print(move)
            #print(infoHandler.info)
            #print(e)
            #print('')
            checkRestartEngine(force=True)
            scores[move] = 0
    moves.sort(key=lambda move: scores[move], reverse=True)
    rankings = {}
    for i, move in enumerate(moves):
            rankings[move] = i
    return rankings

def calcMoveFeatures(board, move):
    #from square
    fromPos = move.from_square
    fromX = getNormX(fromPos)
    fromY = getNormY(fromPos)
    #to square
    toPos = move.to_square
    toX = getNormX(toPos)
    toY = getNormY(toPos)
    #piece type
    pieceTypeLabel = board.piece_type_at(fromPos)
    pieceType = [0.0]*6
    if pieceTypeLabel != None:
        pieceType[pieceTypeLabel-1] = 1.0 
    #promotion type (if any)
    promotion = [0.0]*6
    if move.promotion != None:
        promotion[move.promotion-1] = 1.0
    return np.array([fromX, fromY, toX, toY]+pieceType
                    +promotion, 
                    dtype='float32')
def fen2Features(fen):
    checkRestartEngine()
    board = chess.Board(fen)
    nullMove = chess.Move(0, 0) #null move
    bestMove = getBestMove(board)
    if bestMove == None: #don't deal with bad boards
        return None
    moveRankings = calcMoveRankings(board)
    #consider all possible moves (including no move)
    globalFeatures = calcGlobalFeatures(board)
    pieceFeatures = calcPieceFeatures(board)
    squareFeatures = calcSquareFeatures(board)
    moveFeatures = calcMoveFeatures(board, nullMove)
    moveRanking = np.array([moveRankings[nullMove]/20.0], dtype='float32')
    label = calcLabel(board, nullMove, bestMove)
    
    features = [{'globalFeatures': globalFeatures, 
                 'pieceFeatures':pieceFeatures, 
                 'squareFeatures':np.concatenate((squareFeatures,moveFeatures)),
                 'moveRankings':moveRanking,
                 'labels':label}]
    possibleMoves = list(board.pseudo_legal_moves)
    possibleMoves.append(nullMove)
    possibleMoves = [random.choice(possibleMoves)]
    if possibleMoves[0] != bestMove: possibleMoves.append(bestMove)
    for move in possibleMoves:
        globalFeatures = calcGlobalFeatures(board)
        pieceFeatures = calcPieceFeatures(board)
        squareFeatures = calcSquareFeatures(board)
        moveFeatures = calcMoveFeatures(board, move)
        moveRanking = np.array([moveRankings[move]/20.0], dtype='float32')
        label = calcLabel(board, move, bestMove)
        features.append({'globalFeatures': globalFeatures, 
                         'pieceFeatures':pieceFeatures, 
                         'squareFeatures':np.concatenate((squareFeatures,moveFeatures)),
                         'moveRankings':moveRanking,
                         'labels':label, 
                         'fen', fen, 
                         'move', move})
    return features