This is a readme file which containts instructions to run the code.

Only pre-requite is to have Jupyter notebook with python3 & scki-kit learn installed. Please have the ABAGAIL.jar added to your path incase you are running any *.java file inside `src/main/`

I have provided both the ABAGAIL.jar and ABAGAIL-master folder which can be used to generated the ABAGAIL.jar file.

The jar is compiled on java 1.8

Files of the dataset are:
1. UCI_Credit_Card.csv
	
	Source: https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset


Open Command line:

1. Switch to the directory where  *.ipynb file is located.
2. Run "jupyter notebook" or "jupyter notebook Data.ipynb"
3. Run all cells, this will generated test_credit.csv & train_credit.csv in data folder.
4. Run each of the following files individually to get the outputs of Each optimization problem:
   Random hill climbing on Neural network: CreditRandomHillClimb.java
   Simulated Annealing on Neural network: CreditSimulatedAnnealing.java
   Genetic algorithm on Neural network: CreditGeneticAlgorithm.java
   Continuous Peaks Problem: ContinuousPeaksTest.java
   Knapsack Problem: KnapsackTest.java
5. The output from all the above java files are pasted into csv file in generated_data folder and "Assignment 2 - Randomized Search.xlsx" as well. Prefer to look into the excel sheet as it has column names which will make more meaning for the data.

Code Credits:

1. Sci-kit Documentation: http://scikit-learn.org/stable/modules/tree.html
2. Post-Pruning decision tree: https://stackoverflow.com/questions/49428469/pruning-decision-trees
3. Generating plots: http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
4. https://github.com/pushkar/ABAGAIL/