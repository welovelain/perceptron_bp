package perceptronbp.neuralnetwork;

import perceptronbp.matrix.SimpleMatrixSolver;
import perceptronbp.neuralnetwork.activationfunctions.ActivationFunction;
import perceptronbp.neuralnetwork.activationfunctions.SigmoidActivationFunction;
import perceptronbp.neuralnetwork.activationfunctions.TangensoidActivationFunction;

import java.util.ArrayList;
import java.util.Arrays;

public class Perceptron {

    // these are global weights. e.g. _weights.get(0) - get weight matrix for the 1st layer.
    private ArrayList<double[][]> globalWeights = new ArrayList<>();

    // this is accumulating all outputs of neural network. Updated every epoch.
    private ArrayList<double[]> globalOutputs = new ArrayList<>();

    // this is accumulating errors of each neuron. Updated every epoch
    private ArrayList<double[]> globalErrors = new ArrayList<>();

    // amount of layers and amount of neurons in each layer
    private int[] layers;

    private double meanSquaredError;

    private double learningCoefficient;

    private ActivationFunction activationFunction;

    private Perceptron(Builder builder) {
        this.layers = builder.layers;
        this.activationFunction = builder.activationFunction;
        this.learningCoefficient = builder.learningCoefficient;

        // allocate all
        double[] e = {};
        double[][] ee = {{}};
        for (int i = 0; i < this.layers.length; ++i){
            globalOutputs.add(e);
            globalWeights.add(ee);
            globalErrors.add(e);
        }
        System.out.println ("Space for global weights, errors and outputs allocated");

        initWeights(builder.n_inputs);
        System.out.println ("Weights initialized");
    }

    public static class Builder {

        private final double DEFAULT_LAMBDA = 1.0d;
        private ActivationFunction DEFAULT_ACTIVATION_FUNCTION = new TangensoidActivationFunction(DEFAULT_LAMBDA);
        private double DEFAULT_LEARNING_COEFFICIENT = 0.1d;

        private int[] layers;
        private int n_inputs;

        double learningCoefficient = DEFAULT_LEARNING_COEFFICIENT;
        private ActivationFunction activationFunction = DEFAULT_ACTIVATION_FUNCTION;

        public Builder(int n_inputs, int[] layers) {
            if (layers.length <= 0) {
                throw new IllegalArgumentException("Layers shouldn't be 0 length");
            }
            if (n_inputs <= 0) {
                throw new IllegalArgumentException("Number of inputs should be > 0 length.");
            }
            this.layers = layers;
            this.n_inputs = n_inputs;
        }

        public Builder withLearningCoefficient(double learningCoefficient) {

            if (learningCoefficient <= 0) {
                throw new IllegalArgumentException("Learning coefficient should be > 0");
            }
            this.learningCoefficient = learningCoefficient;
            return this;
        }

        public Builder withActivationFunction(ActivationFunction activationFunction) {
            if (activationFunction == null) {
                throw new IllegalArgumentException("Activation function can't be null");
            }
            this.activationFunction = activationFunction;
            return this;
        }

        public Perceptron build() {
            return new Perceptron(this);
        }

    }

    
    //Add 1 to inputs at the end
    private double[] addBias(double [] inputs){
        
        double[] outputs = Arrays.copyOf(inputs, inputs.length+1);
        outputs[outputs.length - 1] = 1;
        
        return outputs;
    }
    
    // initialize weigts with random numbers (-0.5, 0.5) and put it into _weights
    // e.g. _weigts.get(0) - get weight matrix of 1st layer
    private void initWeights (int n_inputs) {      
        
        int n_prev_neurons; 
        int n_neurons;
        
        for (int i = 0; i < layers.length; ++i){
            
            n_prev_neurons = (i == 0) ?  n_inputs + 1 : layers[i-1] + 1; // +1 is for bias
            n_neurons = layers[i];
            
            // Random weight matrix for layer(i) with weights from -0.5 to 0.5
            double[][] weightMatrix = SimpleMatrixSolver.random(n_neurons, n_prev_neurons);
                       
            // Adding weightMatrix to weights field
            globalWeights.set(i, weightMatrix);
            
        }        
    }
    
    public void learn(ArrayList<double[]> trainData, ArrayList<double[]> dOutputs, int maxEpochs) {
        
        double [] inputVector;
        double [] dOutputVector;
        
        for (int epoch = 0; epoch < maxEpochs; ++epoch) {
            
            // set all necessary vars to 0
            meanSquaredError = 0;
            //_weightDeltas = 0 & _errors = empty
            double[] e = {};
            for (int i = 0; i < layers.length; ++i){
                globalErrors.add(e);
            }
                          
            // we go input by input
            for (int i = 0; i < trainData.size(); ++i ) {
                
                inputVector = trainData.get(i); // vector for one input from batch
                dOutputVector = dOutputs.get(i); // vector of desired output for this input 
                
                calcOutputs(inputVector);                
                calcMSE (dOutputVector);                
                calcNeuronErrors(dOutputVector);
                calcNewWeights(inputVector);
               
            }

            // shuffle trainData and dOutputs here

           System.out.println("epoch: " + epoch + ", MSE: " + meanSquaredError);
                     
        }
        
    }
    
    // calculate outputs and put it into _outputs; 
    // e.g. _outputs.get(0) - get output vector of 1st layer    
    private void calcOutputs (double[] inputs) {
        
        double [] outputs;
        
        for (int l = 0; l < layers.length; ++l){
            
            inputs = addBias(inputs);
            
            // count Net matrix
            double[][] w = globalWeights.get(l);
                        
            outputs = SimpleMatrixSolver.multiply(w, inputs);

            // pass it through activation function
            outputs = activationFunction.activate(outputs);
            
            // add all outputs to _outputs (global outputs store)
            globalOutputs.set(l, outputs);
           
            inputs = outputs;            
        }
        
    }
    
    // MSE = 1/n * SUM [(d_o - y_o)^2]
    private void calcMSE (double[] dOutputs) {
        
           
            double[] outputs = globalOutputs.get(layers.length - 1);
            
            
          
            
            for (int i = 0; i < outputs.length; ++ i ) {
                meanSquaredError += (Math.pow((dOutputs[i] - outputs[i]), 2)) / outputs.length;
            }
            //_mse /= outputs.length;
    }
    
    // calculate Errors of each neuron
    private void calcNeuronErrors(double[] dOutputVector) {
        
        double error;
        int n_layers= layers.length ;
                
        // back-propagating
        for (int layer = n_layers - 1; layer >= 0; --layer) {
            
            double[] errorsVector = new double[globalOutputs.get(layer).length];
            double[] outputsVector = globalOutputs.get(layer);

            for (int i = 0; i < outputsVector.length; ++i ) {
                
                if ( layer == n_layers - 1){ // first count errors of outputs
                     
                    double y = outputsVector[i];
                    double d = dOutputVector[i];
                    error = activationFunction.getDerivative(y) * (d - y); // y * (1-y) * (d-y)
                     
                } else {
                    
                    double sum = 0;    
                    double[] nextLayerErrorsVector = globalErrors.get(layer + 1); // vector of errors of next layer
                    
                    for (int j = 0; j < nextLayerErrorsVector.length; ++j ){
                        
                        double connectedW = globalWeights.get(layer + 1)[j][i];
                        sum += nextLayerErrorsVector[j] * connectedW;
                        
                    }
                    
                    error = activationFunction.getDerivative(outputsVector[i]) * sum;
                    
                }
               
                errorsVector[i] = error;
            }


            globalErrors.set(layer, errorsVector);
           
        }
        
    }
    
     // deltaW = q*d*z
    private void calcNewWeights(double[] inputVector) {
        
        double q = learningCoefficient;
        double[] errorsVector;
        double[] neuronWeightDeltaVector;
       
        
        
        for (int layer = 0; layer < layers.length; ++layer ) {
            
           errorsVector = globalErrors.get(layer);
           double[][] newWeightDelta = globalWeights.get(layer);
            
           for (int i = 0; i < errorsVector.length; ++i ) {
               
               double d = globalErrors.get(layer)[i];
               double deltaW = q * d;
               
               if (layer  == 0 ) 
                   neuronWeightDeltaVector = SimpleMatrixSolver.multiply(inputVector, deltaW);
               else {
                   inputVector = globalOutputs.get(layer - 1);
                   neuronWeightDeltaVector = SimpleMatrixSolver.multiply(inputVector, deltaW);
               }
                              
               for (int j = 0; j < neuronWeightDeltaVector.length; ++j){
                   newWeightDelta[i][j] += neuronWeightDeltaVector[j];
               }
               
           } // neuron

            globalWeights.set(layer, newWeightDelta);
            
        } //layer
        
    }

    
    public void test(ArrayList<double[]> testTrainData, ArrayList<double[]> testDOutputs) { 
    
        double [] inputVector;
        double [] dOutputVector;
             
            // we go input by input
            for (int i = 0; i < testTrainData.size(); ++i ) {
                inputVector = testTrainData.get(i); // vector for one input from batch
                dOutputVector = testDOutputs.get(i); // vector of desired output for this input 
                
                calcOutputs(inputVector);      
               
                double[] outputVector = globalOutputs.get(layers.length - 1);
                
                printInputLetter(inputVector);
                
                System.out.print("Expected result is: ");
                printOutputLetter(dOutputVector);
                
                System.out.print("\r\nPerceptron thinks that the result is: ");
                printOutputLetter(outputVector);
                System.out.println(); 
                System.out.println("------------------------");
            }
        
    }
    
    //----------STATIC METHODS-----------
    
    private static void printInputLetter (double[] input) {
        
        String result="";
        int count = 0;
        for (int i = 0; i < 6; ++i){
            for (int j = 0; j < 4; ++j) {
               
                if (input[count] == 1) result += "1";
                else result += " ";
               
               ++count;
            }
            result+="\r\n";
           
        }
        
        System.out.println(result);
        
    }
    
    private static void printOutputLetter (double[] input) {
        
        double maxValue = input[0];
        int maxValueIndex = 0;
        
        for (int i = 1; i < input.length; ++i) {
            if (input[i] > maxValue) {
                maxValue = input[i];
                maxValueIndex = i;
            }
        }
        
        char result =  'a';
        result += maxValueIndex;
        
        System.out.print(result);
        
    }
    
//    public static ArrayList<double[]> createTrainDataXOR() {
//
//          ArrayList<double[]> trainData = new ArrayList<>();
//
//          trainData.add(new double[]{0,0});
//          trainData.add(new double[]{0,1});
//          trainData.add(new double[]{1,0});
//          trainData.add(new double[]{1,1});
//
//          return trainData;
//    }
//
//
//    public static ArrayList<double[]> createDesiredOutputDataXOR() {
//
//          ArrayList<double[]> dOutputs = new ArrayList<>();
//
//
//            dOutputs.add(new double[]{0});
//            dOutputs.add(new double[]{1});
//            dOutputs.add(new double[]{1});
//            dOutputs.add(new double[]{0});
//
//          return dOutputs;
//    }
    
    






}
