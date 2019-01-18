package edu.packt.neuralnet.examples.chapter07;

import java.util.Arrays;
import java.util.Scanner;

import edu.packt.neuralnet.NeuralException;
import edu.packt.neuralnet.data.DataNormalization;
import edu.packt.neuralnet.data.DataNormalization.NormalizationTypes;
import edu.packt.neuralnet.data.DataSet;
import edu.packt.neuralnet.data.NeuralDataSet;
import edu.packt.neuralnet.data.NeuralOutputData;
import edu.packt.neuralnet.init.UniformInitialization;
import edu.packt.neuralnet.learn.LearningAlgorithm;
import edu.packt.neuralnet.math.ArrayOperations;
import edu.packt.neuralnet.math.RandomNumberGenerator;
import edu.packt.neuralnet.som.CompetitiveLearning;
import edu.packt.neuralnet.som.Kohonen;

/**
 *
 * ClusterExamples This class performs use cases described in
 * chapter 7 of the book: Credit Analysis for Customer Profiling (card)
 * and Product Profiling . Data are loaded, data normalization is done, 
 * neural net is created using parameters defined, SOM approach is used 
 * to make neural net learn and charts are plotted
 * 
 * @authors Alan de Souza, Fábio Soares
 * @version 0.1
 * 
 */
public class ClusterExamples {

	public static void main(String[] args) {

		RandomNumberGenerator.setSeed(7);
		
		String CHOSEN_OPTION = "";
		int numberOfInputs  = 0;
		int numberOfNeurons = 0;
		int[] inputColumns  = null;
		int[] outputColumns = null; //special case
		
		DataSet dataSet = new DataSet();
		DataSet dataSetTestInputs  = null;
		DataSet dataSetTestOutputs = null;
		
		DataNormalization dn = null;
		
		Scanner sc = new Scanner(System.in);
		
		int experiment = 0;
		
		while(true) {
			boolean flagNeural = true;
			
			experiment++;
			System.out.println("*** EXPERIMENT #"+experiment+" ***");
			
			System.out.println("Number of epochs:");
			int typedEpochs = sc.nextInt();
			
			System.out.println("Number of neurons to cluster:");
			int typedNumNeurons = sc.nextInt();
			
			System.out.println("Learning rate:");
			double typedLearningRate = sc.nextDouble();
			
			System.out.println("Normalization type:");
			System.out.println("1) MIN_MAX [-1; 1]");
			System.out.println("2) Z_SCORE");
			int typedNormalization = sc.nextInt();
			
			System.out.println("Which dataset would you like to use?");
			System.out.println("1) Card");
			System.out.println("2) Product profiling");
			System.out.println("99) Quit");
			int option = sc.nextInt();
			
			switch (option) {
				case 1:
					CHOSEN_OPTION   = "Card";
					numberOfInputs  = 10;
					
					// load data
					dataSetTestInputs  = new DataSet();
					dataSetTestOutputs = new DataSet();
					
					dataSet = new DataSet("data", "card_inputs_training.txt");
					dataSetTestInputs  = new DataSet("data", "card_inputs_test.txt");
					dataSetTestOutputs = new DataSet("data", "card_output_test.txt");
					inputColumns  = new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
					outputColumns = new int[] { 0 };
					
					break;
				case 2:
					CHOSEN_OPTION   = "Product profiling";
					numberOfInputs  = 27;
					
					// load data
					dataSet = new DataSet("data", "product_profiling_training.txt");
					inputColumns  = new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
												15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26 };
					//outputColumns = new int[] { 8, 9 };
					
					break;
				case 99:
					System.out.println("Successful exit!");
					System.exit(0);
					break;
				default:
					System.out.println("Invalid option for dataset...");
					flagNeural = false;
			}
			//sc.close();
			
			switch (typedNormalization) {
				case 1:
					dn = new DataNormalization(-1.0, 1.0);
					break;
				case 2:
					dn = new DataNormalization(NormalizationTypes.ZSCORE);
					break;
				default:
					System.out.println("Invalid option for normalization...");
					flagNeural = false;
					break;
			}
			
			if(flagNeural) {
				
				numberOfNeurons = typedNumNeurons;
			
				System.out.println("Creating Neural Network...");
				Kohonen kn1 = new Kohonen(numberOfInputs, numberOfNeurons, new UniformInitialization(0.0,1000.0),1);
				System.out.println("Neural Network created!");
				//nn.print();
				
				// load data
				double[][] _neuralDataSet = dataSet.getData();
				
				double[][] _neuralDataSetTestInputs = null;
				if (dataSetTestInputs != null) {
					_neuralDataSetTestInputs = dataSetTestInputs.getData();
				}
				double[][] _neuralDataSetTestOutput = null;
				if (dataSetTestOutputs != null) {
					_neuralDataSetTestOutput = dataSetTestOutputs.getData();
				}
			
				// normalize data
				double[][] dataNormalized = new double[_neuralDataSet.length][_neuralDataSet[0].length];
				dataNormalized = dn.normalize(_neuralDataSet);
				
				double[][] dataNormalizedToTrain = dataNormalized;
				
				// normalize data to train ANN:
				NeuralDataSet neuralDataSetToTrain = new NeuralDataSet(dataNormalizedToTrain, inputColumns.length);
				
				// normalize data to test ANN:
				NeuralDataSet neuralDataSetToTest = null;
				if(outputColumns != null) {
					double[][] dataNormalizedToTestInputs = new double[_neuralDataSetTestInputs.length][_neuralDataSetTestInputs[0].length];
					dataNormalizedToTestInputs = dn.normalize(_neuralDataSetTestInputs);
				
					neuralDataSetToTest = new NeuralDataSet(dataNormalizedToTestInputs, inputColumns.length);
				}
				
				// create ANN and define parameters to TRAIN: 
				CompetitiveLearning cl = new CompetitiveLearning(kn1, neuralDataSetToTrain, LearningAlgorithm.LearningMode.ONLINE);
		        cl.show2DData=false;
		        cl.printTraining=false;
		        cl.setLearningRate( typedLearningRate );
		        cl.setMaxEpochs( typedEpochs );
		        cl.setReferenceEpoch( 100 );
		        if(outputColumns != null) {
		        	cl.setTestingDataSet(neuralDataSetToTest);
		        }
		        
		        // train ANN
		        try {
		        	System.out.println("Training neural net... Please, wait...");
		        	
		        	cl.train();
		        	
		        	System.out.println("Winner neurons (clustering result) [TRAIN]:");
		        	System.out.println( Arrays.toString( cl.getIndexWinnerNeuronTrain() ) );
		        	
				} catch (NeuralException ne) {
					ne.printStackTrace();
				}
		        
		        // test ANN
		        if (outputColumns != null) {
			        try {
			        	
			        	cl.test();
			        	
			        	System.out.println("\n\n### EXPERIMENT " + experiment +" ###");
			        	System.out.println("### Results for "+ CHOSEN_OPTION +" ###");
			        	
			        	// confusion matrix after test:
			        	NeuralOutputData nout = new NeuralOutputData();
			        	
			        	double[] realData = ArrayOperations.getColumn(_neuralDataSetTestOutput, 0); 
			        	double[][] confusionMatrix = nout.calculateConfusionMatrix(cl.getIndexWinnerNeuronTest(), realData);
			        	
						System.out.println("\n### Confusion Matrix ###");
						System.out.println( Arrays.deepToString(confusionMatrix).replaceAll("],", "]\n") );
			        	
			        	nout.calculatePerformanceMeasures(confusionMatrix);
			        	
			        	System.out.println("\n\n\n");
			        	
					} catch (NeuralException ne) {
						ne.printStackTrace();
					}
		        }
				
			} //end if flagNeural
			
		} //end while loop 

	}


}
