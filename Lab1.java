/**
 *
 *  Author: Gautam Singh                      gautamsingh@cs.wisc.edu
 *  Copyright 2017
 *  Perceptron implementation in java         CS838 Deep Neural Networks - Lab1
 *
 *
 */

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

import java.util.ArrayList;
import java.util.Random;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.Enumeration;


public class  Lab1{

    public static void main(String[] args) {
	
	    Perceptron perceptron;
	    boolean debug = false;
	    double learningRate = 0.05;

	    int ITER_MAX = 5000;
	
	    double bestWeights[];
	
	    if (args.length != 3) {
	        System.err.println("Enter filename:  <train> <tune> <test>");
	        System.exit(1);
	    }

	    Dataset train_data = readData(args[0]);
        Dataset tune_data  = readData(args[1]);
        Dataset test_data  = readData(args[2]);

	 
	    int [] predictedValues = new int [ train_data.numExamples];

	    perceptron = new Perceptron( train_data.numFeatures, learningRate);
	
	     bestWeights = new double [ train_data.numFeatures +1 ];
	    double bestError = Double.MAX_VALUE;
	    int iteration = 0;
	    int count = 0;
	 
	    do{

	        iteration ++;

	        double  trainError = perceptron.train(train_data);
	        double   tuneError = perceptron.tune(tune_data);
	     
	        if(tuneError < bestError){
		        bestError = tuneError;
		        System.arraycopy(perceptron.getWeights(), 0, bestWeights, 0, bestWeights.length);
		        count = 0;
	        }else{
		        count ++;
		        if(count >= 100)
		        break;
	        }
	     
	        if(debug){
		        double testError = perceptron.percentageError(test_data);
		        System.out.println(" Train Error: " +trainError + " Tune Error: " + tuneError);
		        System.out.println(" Percentage error on test set : " + testError);
	        }


	    }while(iteration < ITER_MAX);
	 
	    perceptron.setWeights(bestWeights);

	    perceptron.testOut(test_data );

	    // printPredicted(predictedValues);
	 
	    if(debug)
	    System.out.println(" Best errorRate : " +bestError);

    }

    /*******************************************************************************************************************
     *
     * @param file
     * @return
     */

    public static Dataset readData( String file){

        Scanner fileScanner = null;
        try {
            fileScanner = new Scanner(new File(file));
        } catch(FileNotFoundException e) {
            System.err.println("Could not find file '" + file + "'.");
            System.exit(1);
        }

        // Iterate through each line in the file.
        int lineCount =  -1;
        boolean skipLine = true;
        int numFeatures = 0;
        int numExamples = 0;
        boolean debugParser = false;
        double countLabels = 0;

        Dataset data = new Dataset();

        while(fileScanner.hasNext()) {
            String line = fileScanner.nextLine().trim();

            // Skip blank lines
            if(line.length() == 0  || line.startsWith("//")) {
                continue;
            }


            // read number of features as int from dataset
            if(lineCount < 0){
                numFeatures = Integer.parseInt(line);
                data.numFeatures = numFeatures;
                if(debugParser)
                    System.out.println("Number of features : " +numFeatures);
                lineCount++;
                continue;
            }


            Scanner tempScanner = new Scanner(line);

            if(lineCount == 0){

                String  featureName = tempScanner.next(); // Useless
                String  temp = tempScanner.next(); // again useless

                double key = 0;
                while(tempScanner.hasNext()){
                    data.features.put(tempScanner.next(),key);  // add all feature values with
                    key += 1;
                }

                lineCount ++;
                continue;
            }
            // skip feature descriptions, two possible labels and read num of examples

            if( skipLine && lineCount <= (numFeatures+2)){

                if(lineCount < (numFeatures)){
                    lineCount++;
                    continue;
                }else if(lineCount == (numFeatures+2)) {

                    numExamples = Integer.parseInt(line);
                    data.numExamples = numExamples;
                    if (debugParser)
                        System.out.println("Number of examples : " + numExamples);
                    skipLine = false;
                    lineCount = 0;
                    break;
                }else if(lineCount>=numFeatures  && tempScanner.hasNext()){

                    data.labels.put(tempScanner.next(),countLabels);
                    countLabels += 1.0;
                    lineCount++;
                }
            }
        }


        // Read each example
        while(fileScanner.hasNext()) {
            String line = fileScanner.nextLine().trim();

            // Skip blank lines.
            if(line.length() == 0  || line.startsWith("//")) {
                continue;
            }

            if(debugParser)
                System.out.println("Line " + lineCount + "\n=======");
            //lineCount++;
            // Use another scanner to parse each word from the line and print it.
            Scanner lineScanner = new Scanner(line);
            Instance instance = new Instance(numFeatures);
            int wordCount = -2;
            while(lineScanner.hasNext()) {
                String word = lineScanner.next();
                if(debugParser)
                    System.out.println("Word " + wordCount + ": " +word);
                wordCount++;

                // Skip example name
                if(wordCount >=0){

                    if(0 == wordCount){  // Read label
                        instance.setLabel(word);
                    }else{
                        // Read features
                        double val = ((Double)data.features.get(word)).doubleValue();
                        instance.features.add(val );
                    }
                }
            }
            data.add(instance);
            lineCount++;
            if(debugParser)
                System.out.println();
        }

        if(debugParser) {

            System.out.println(" Number of example: " + data.numExamples);
            System.out.println(" Number of features: " + data.numFeatures);


            Enumeration feature = data.features.keys();

            while (feature.hasMoreElements()) {
                String str = (String) feature.nextElement();
                System.out.println(str + ": " + data.features.get(str));
            }


            for (int i = 0; i < data.numExamples; i++) {

                for (int j = 0; j < data.instances.get(0).size(); j++)
                    System.out.print(data.instances.get(i).get(j) + " ");

                System.out.println("");
            }
        }


        return data;

    }

}


/***********************************************************************************************************************
 *
 *
 *        Perceptron class
 *
 **********************************************************************************************************************/


class Perceptron{

    double weights[];
    double learningRate;
    int numInputs,numOutputs;

    boolean debug;
    /*
      Constructor method
      
      @param numInputs : Number of input units
      @param learningRate: learning rate for Perceptron

    */
    public Perceptron(int numInputs, double learningRate){

	    this.numInputs = numInputs;
	    this.learningRate = learningRate;
	    this.numOutputs = 1; // only 1 output unit
	
	    this.weights = new double[numInputs + 1] ; // including bias
	
	    debug = false; //  true: print all debug statements

	    Random rand = new Random();
	
	    for(int i=0; i <=numInputs; i++){

	        int sign = rand.nextInt() % 2;
	        weights[i] = rand.nextDouble();

	        if( 0 == sign)
		        weights[i] *= -1;

	        if(debug)
	            System.out.println("Weight[ " +i +"] : " + weights[i]);
	    }

    }
    
    /*******************************************************************************************************************
    calculate weighted sum based on given input
    
       @param input: input features
       @return       weighted sum
     */
    public double weightedSum(Instance input) {

	    double sum = weights[0]; // bias
	
	    for(int i = 0; i< input.features.size(); i++) {
	    
	        sum += input.features.get(i) * weights[i];

	        //if(debug)
		    //    System.out.println(" Input[ " +i+" ] : " +input[i]);
	    }
	    //sum += weights[input.length];  // bias
	
	    if(debug)
	        System.out.println("Weighted sum : " + sum);
	    
	    return sum;
    }

    /*******************************************************************************************************************
      Adjust weights based on calculated error 
      suggested loss function: Use squared error (plus the sum of squared weights if you do weight decay)  
      
     */

    public void backpropagate(Instance input, double output, double actual){
	
	    double error = output - actual;
	
	    for( int i =1; i<= input.features.size(); i++) {
	        weights[i] +=  error* learningRate * input.features.get(i-1);
	    }
	    weights[0] += error*learningRate;
	
	    for(int i =0; debug && i<= numInputs; i++ ) {
	        System.out.println(weights[i] + " ");
	    }
	    if(debug)
	        System.out.println("Output: " +output +" Actual: " + actual + " Error: " + error + " ");
    }

    public double output( double weightSum){

	    return (weightSum < 1) ? 0 : 1;

    }

    public double train(Dataset data){

        double globalError = 0;

	    for( int i = 0; i <data.numExamples; i++){
	
	        double weightSum = weightedSum(data.instances.get(i));
	        double out = output( weightSum);
	        String label = data.instances.get(i).label;
           // System.out.println(" Bug:  Label is null " + label);
            Hashtable ht = data.labels;
           // System.out.println(label + ": " + ht.get(label));

            double expected = ((Double) ht.get(label)).doubleValue();
	        double localError =  expected-out;

	        this.backpropagate(data.instances.get(i),expected,out);
	        globalError += localError * localError;
	    }
	    if(debug){
	        System.out.println("Squared error in this training epoch : " + globalError);
	    }
	    return globalError;
    }

    public double tune(Dataset data){
	
	    double globalError = 0;

        for( int i = 0; i <data.numExamples; i++){

            double weightSum = weightedSum(data.instances.get(i));

	        double out = output( weightSum);
            String label = data.instances.get(i).label;

            //System.out.println(" Bug:  Label is null " + label);
            Hashtable ht = data.labels;
            //System.out.println(label + ": " + ht.get(label));

            double expected = ((Double) ht.get(label)).doubleValue();

            double localError =expected -out;

            globalError+= localError *localError;
        }
	    if(debug){
            System.out.println("Squared error in this tuning epoch : "+ globalError);
        }
        return globalError;
    }
    
    public double percentageError(Dataset data ){
	
	    double globalError = 0;
	    double errorPercent = 0;
	    double errorCount = 0;
	    for(int i =0; i<data.numExamples; i++) {
	    
	        double weightSum = weightedSum(data.instances.get(i));
            double out = output( weightSum);
            double expected = (Double)data.features.get(data.instances.get(i).label );
            double localError =expected -out;
            globalError+= localError *localError;
	    
	        errorCount += (globalError > 0.5)? 1: 0;
	    }
	    errorPercent = (errorCount/ data.numExamples) * 100;
	    if(true){
            System.out.println("Squared error in this test epoch : "+ globalError);
	    System.out.println("Percent error in this test epoch : "+ errorPercent);
	    }
	    return errorPercent;
    }

    public void testOut(Dataset data) {

	    double globalError = 0;
        //double errorCount = 0;
        
	    int count = 0;
        for(int i =0; i<data.numExamples; i++) {

            double weightSum = weightedSum(data.instances.get(i));
            double predicted = output( weightSum);

            String label = data.instances.get(i).label;

            Hashtable ht = data.labels;


            double expected = ((Double) ht.get(label)).doubleValue();

            double localError = expected -predicted;

            globalError+= localError *localError;

	        // errorCount += (globalError > 0.5)? 1: 0;
	        int predictedValue = (int) predicted;
	    
	        if(expected == predicted)
		    count ++;

            Enumeration lt = data.labels.keys();

            while(lt.hasMoreElements()) {
                String str = (String) lt.nextElement();
                if((double)data.labels.get(str) == predicted) {
                    System.out.println(str);
                    break;
                }
            }


        }
        if(debug){                                                                                                                                                             
            System.out.println("Squared error in this test epoch : "+ globalError);
	        System.out.println(" Number of correct examples : "+ count);
	    }
	    double accuracy = (count * 100.00) / data.numExamples;
	    System.out.println("Accuracy in percent: " + accuracy);

    }

    public double [] getWeights(){

	    return this.weights;

    }

    public void setWeights(double inputs[]) {

	    for(int i =0;i < inputs.length; i++)
	         weights[i] = inputs[i];

    }
}

/***********************************************************************************************************************
 *
 *     Dataset instance class
 *
 **********************************************************************************************************************/

class Instance{
    ArrayList<Double> features;
    String label;

    public Instance(int numFeatures){

        features = new ArrayList<Double>();
    }
    public void setLabel(String label){
        this.label=label;
    }
    public int size(){
        return features.size();
    }
    public Double get(int pos){
        return features.get(pos);
    }

    public String getLabel(){
        return label;
    }
}
/**********************************************************************************************************************
*
*       Class for Dataset
*
 **********************************************************************************************************************/

class Dataset {
    int numFeatures;
    int numExamples;
    ArrayList<Instance>          instances;
    Hashtable                    labels;
    Hashtable                    features;

    public Dataset() {
        this.instances = new ArrayList<Instance>();
        this.features  = new Hashtable();
        this.labels    = new Hashtable();
    }

    public Instance get(int pos){

        return instances.get(pos);
    }

    public int getSize() {
        return instances.size();
    }

    public void add(Instance inst) {
        instances.add(inst);
    }
}