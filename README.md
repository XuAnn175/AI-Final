# AI-Finals  
AI FINAL PROJECT  : Applying Ai to 3D Five in a row (將Ai運用於立體五子棋)  
Target : To build a application that allow people to game with AI  
Model : Mainly Adversarial Search (Minimax) , will also try other model to gain better performance.  
The main functions of this project are listed as follows.(details are to be implemented)  
```cpp
class ChessBoard{
	public:
	    int chessBoard[5][5][5];
	    void init(); //initialize the chessboard
	    void printBoard(); 
	    bool valid();
	    bool insertPices();
	    int dfs();   //search
	    int isWin(); //return who wins
	    int getSideScore();
	    int getPosValue();
	    int findWhiteScoreMaxValue(); //for alpha-beta pruning
	    int findWhiteScoreMinValue(); //for alpha-beta pruning
	    PicesPosList getAvailablePos();
};
```
