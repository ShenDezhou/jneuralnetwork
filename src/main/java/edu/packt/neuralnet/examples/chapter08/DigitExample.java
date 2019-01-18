package edu.packt.neuralnet.examples.chapter08;

import java.util.Arrays;
import java.util.Scanner;

import edu.packt.neuralnet.NeuralException;
import edu.packt.neuralnet.NeuralNet;
import edu.packt.neuralnet.data.DataNormalization;
import edu.packt.neuralnet.data.DataNormalization.NormalizationTypes;
import edu.packt.neuralnet.data.DataSet;
import edu.packt.neuralnet.data.NeuralDataSet;
import edu.packt.neuralnet.init.UniformInitialization;
import edu.packt.neuralnet.learn.Backpropagation;
import edu.packt.neuralnet.learn.LearningAlgorithm;
import edu.packt.neuralnet.math.ArrayOperations;
import edu.packt.neuralnet.math.IActivationFunction;
import edu.packt.neuralnet.math.Linear;
import edu.packt.neuralnet.math.RandomNumberGenerator;
import edu.packt.neuralnet.math.Sigmoid;
import edu.packt.neuralnet.som.CompetitiveLearning;
import edu.packt.neuralnet.som.Kohonen;

/**
 *
 * DigitExamples This class performs use cases described in
 * chapter 8 of the book. Data are loaded, data normalization is done, 
 * neural net is created using parameters defined, Backpropagation approach  
 * is used to make neural net learn and charts are plotted
 * 
 * @authors Alan de Souza, Fábio Soares
 * @version 0.1
 * 
 */
public class DigitExample {

	public static void main(String[] args) {

		RandomNumberGenerator.setSeed(7);
		
		String CHOOSEN_DATASET = "";
		int numberOfInputs  = 0;
		int numberOfOutputs = 0;
		int[] inputColumns  = null;
		int[] outputColumns = null;
		
		DataSet dataSet = new DataSet();
		
		Scanner sc = new Scanner(System.in);
		
		int experiment = 0;
		
		while(true) {
			boolean flagNeural = true;
			
			experiment++;
			System.out.println("*** EXPERIMENT #"+experiment+" ***");
			
			System.out.println("Number of epochs:");
			int typedEpochs = sc.nextInt();
			
			System.out.println("Number of neurons in hidden layer:");
			int typedNumHdnLayer = sc.nextInt();
			
			System.out.println("Learning rate:");
			double typedLearningRate = sc.nextDouble();
			
			System.out.println("Which dataset would you like to use?");
			System.out.println("1) OCR 10X10");
			System.out.println("2) OCR 20X20");
			System.out.println("99) Quit");
			int option = sc.nextInt();
			
			numberOfOutputs = 10;
			
			switch (option) {
				case 1:
					CHOOSEN_DATASET   = "OCR 10X10";
					numberOfInputs  = 100;
					
					// load data
					dataSet = new DataSet("data", "digits_ocr_10x10.txt");
					
					break;
				case 2:
					CHOOSEN_DATASET   = "OCR 20X20";
					numberOfInputs  = 400;
					
					// load data
					dataSet = new DataSet("data", "digits_ocr_20x20.txt");
					
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
			
			inputColumns  = buildInputColumns(0, numberOfInputs);
			outputColumns = buildInputColumns(numberOfInputs, numberOfOutputs+numberOfInputs);
			
			if(flagNeural) {
			
				int[] numberOfHiddenNeurons = { typedNumHdnLayer };
			
				//HyperTan hl0Fnc = new HyperTan(1.0);
				//HyperTan outputAcFnc = new HyperTan(1.0);
				Sigmoid hl0Fnc = new Sigmoid(1.0);
				//Sigmoid outputAcFnc = new Sigmoid(1.0);
				Linear outputAcFnc = new Linear(1.0);
			
				IActivationFunction[] hiddenAcFnc = { hl0Fnc };
			
				System.out.println("Creating Neural Network...");
				NeuralNet nn = new NeuralNet(numberOfInputs, numberOfOutputs, numberOfHiddenNeurons, hiddenAcFnc, outputAcFnc,
						new UniformInitialization(-1.0, 1.0));
				System.out.println("Neural Network created!");
				//nn.print();
				
				// load data
				double[][] inputDataSet = dataSet.getData();
			
				// normalize data
				DataNormalization dn = new DataNormalization(NormalizationTypes.ZSCORE);
				double[][] inputNormalized = new double[inputDataSet.length][inputDataSet[0].length];
				inputNormalized = dn.normalize(inputDataSet);
				
				double[][] dataNormalizedToTrain = Arrays.copyOfRange(inputNormalized, 0, 40);
				double[][] dataNormalizedToTest  = Arrays.copyOfRange(inputNormalized, 40, 50);
				
				// normalized data to train ANN:
				NeuralDataSet neuralDataSetToTrain = new NeuralDataSet(dataNormalizedToTrain, inputColumns, outputColumns);
				
				// normalized data to test ANN:
				NeuralDataSet neuralDataSetToTest  = new NeuralDataSet(dataNormalizedToTest, inputColumns, outputColumns);
			
				//System.out.println("Dataset to train created");
				//neuralDataSetToTrain.printInput();
				//neuralDataSetToTrain.printTargetOutput();
			
				System.out.println("Getting the first output of the neural network");
			
				// create ANN and define parameters to TRAIN: 
				Backpropagation backprop = new Backpropagation(nn, neuralDataSetToTrain, LearningAlgorithm.LearningMode.BATCH);
				backprop.setLearningRate( typedLearningRate );
				backprop.setMaxEpochs( typedEpochs );
				backprop.setGeneralErrorMeasurement(Backpropagation.ErrorMeasurement.SimpleError);
				backprop.setOverallErrorMeasurement(Backpropagation.ErrorMeasurement.MSE);
				backprop.setMinOverallError(0.001);
				backprop.setMomentumRate(0.7);
				backprop.setTestingDataSet(neuralDataSetToTest);
				backprop.printTraining = true;
				backprop.showPlotError = true;
				
				// train ANN:
				try {
					backprop.forward();
					//neuralDataSetToTrain.printNeuralOutput();
				
					backprop.train();
					System.out.println("End of training");
					if (backprop.getMinOverallError() >= backprop.getOverallGeneralError()) {
						System.out.println("Training successful!");
					} else {
						System.out.println("Training was unsuccessful");
					}
					System.out.println("Overall Error:" + String.valueOf(backprop.getOverallGeneralError()));
					System.out.println("Min Overall Error:" + String.valueOf(backprop.getMinOverallError()));
					System.out.println("Epochs of training:" + String.valueOf(backprop.getEpoch()));
				
					//System.out.println("Target Outputs (TRAIN):");
					//neuralDataSetToTrain.printTargetOutput();
				
					//System.out.println("Neural Output after training:");
					//backprop.forward();
					//neuralDataSetToTrain.printNeuralOutput();
				} catch (NeuralException ne) {
					ne.printStackTrace();
				}
			
				//System.out.println("Dataset to test created");
				//neuralDataSetToTest.printInput();
				//neuralDataSetToTest.printTargetOutput();
				
				try {
					
					backprop.test();// forward();
					
					//neuralDataSetToTest.printNeuralOutput();
					
					//System.out.println("Target Outputs (TEST):");
					//neuralDataSetToTest.printTargetOutput();
					
					double[][] targetDataSet = dataSet.getData(outputColumns);
					
					double[][] estimatedDataSetNorm = ArrayOperations.arrayListToDoubleMatrix( neuralDataSetToTest.getArrayNeuralOutputData() );
					int[] comparisonTargetVsEstimated = new int[estimatedDataSetNorm.length];
					
					double[][] m = new double[estimatedDataSetNorm.length][estimatedDataSetNorm[0].length];
					
					for (int i = 0; i < estimatedDataSetNorm.length; i++) {
						int el = outputColumns[i];
						double[] vNormEstimated = ArrayOperations.getColumn( estimatedDataSetNorm, i );
						double[] vDenormEstimated = dn.denormalize( vNormEstimated, el );
						double[] vTarget = ArrayOperations.getColumn( targetDataSet, i );
						
						for (int j = 0; j < vDenormEstimated.length; j++) {
							m[i][j] = vDenormEstimated[j];
						}
						
						int indexMaxTarget = ArrayOperations.indexmax(vTarget);
						int indexMaxEstimated = ArrayOperations.indexmax(vDenormEstimated);
						
						if ( indexMaxTarget == indexMaxEstimated ) {
							comparisonTargetVsEstimated[i] = 1;
						}
						
					}
					
					System.out.println("Comparison Target x Estimated");
					System.out.println( Arrays.toString(comparisonTargetVsEstimated).replaceAll(", ", "\n") );
					
					System.out.println("REAL MATRIX");
					System.out.println( Arrays.deepToString(targetDataSet).replaceAll("],", "]\n") );
					System.out.println("ESTIMATED MATRIX");
					System.out.println( Arrays.deepToString(m).replaceAll("],", "]\n") );
					
				} catch (Exception e) {
					e.printStackTrace();
				}
		
			} //end if flagNeural
			
		} //end while loop 

	}
	
	private static int[] buildInputColumns(int begin, int end) {
		int length = end-begin;
		int[] inputColumns = new int[length];
		for (int i = 0; i < length; i++) {
			inputColumns[i] = begin;
			begin++;
		}
		return inputColumns;
	}


}
