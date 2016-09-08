package perceptronbp;

import java.util.ArrayList;
import java.util.Arrays;

public class PerceptronBP {
    
    // main constructor
    public PerceptronBP (int n_inputs, int[] layers) {
        _layers = layers;
        
        // allocate all
        double[] e = {};
        double[][] ee = {{}};
        for (int i = 0; i < _layers.length; ++i){
            _outputs.add(e);
            _weights.add(ee);
            _errors.add(e);
        }
        System.out.println ("Space for global weights, errors and outputs allocated");
        
        initWeights(n_inputs);
        System.out.println ("Weights initialized");
    }
    
    
    
    // 1 / (1+e^(-lambda*x))
    private double[] sigmoidActivate (double [] inputs){
        
        double[] outputs = new double[inputs.length];
        double result; 
        
        for (int i = 0; i < inputs.length; ++i ){
            result = 1 / (1 + Math.exp(-1 * _lambda * inputs[i]));
            outputs[i] = result;
        }
        
        return outputs;
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
        
        for (int i = 0; i < _layers.length; ++i){
            
            n_prev_neurons = (i == 0) ?  n_inputs + 1 : _layers[i-1] + 1; // +1 is for bias
            n_neurons = _layers[i];
            
            // Random weight matrix for layer(i) with weights from -0.5 to 0.5
            double[][] weightMatrix = Matrix.random(n_neurons, n_prev_neurons);
                       
            // Adding weightMatrix to weights field
            _weights.set(i, weightMatrix);
            
        }        
    }
    
    private void learn(ArrayList<double[]> trainData, ArrayList<double[]> dOutputs, int maxEpochs) {
        
        double [] inputVector;
        double [] dOutputVector;
        
        for (int epoch = 0; epoch < maxEpochs; ++epoch) {
            
            // set all necessary vars to 0
            _mse = 0;
            //_weightDeltas = 0 & _errors = empty
            double[] e = {};
            for (int i = 0; i < _layers.length; ++i){
                _errors.add(e);
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
           
           System.out.println("epoch: " + epoch + ", MSE: " + _mse);
                     
        }
        
    }
    
    // calculate outputs and put it into _outputs; 
    // e.g. _outputs.get(0) - get output vector of 1st layer    
    private void calcOutputs (double[] inputs) {
        
        double [] outputs;
        
        for (int l = 0; l < _layers.length; ++l){
            
            inputs = addBias(inputs);
            
            // count Net matrix
            double[][] w = _weights.get(l);
                        
            outputs = Matrix.multiply(w, inputs);            
            // pass it through activation function
            outputs = sigmoidActivate(outputs);
            
            // add all outputs to _outputs (global outputs store)
            _outputs.set(l, outputs);
           
            inputs = outputs;            
        }
        
    }
    
    // MSE = 1/n * SUM [(d_o - y_o)^2]
    private void calcMSE (double[] dOutputs) {
        
           
            double[] outputs = _outputs.get(_layers.length - 1);  
            
            
          
            
            for (int i = 0; i < outputs.length; ++ i ) 
                _mse += Math.pow((dOutputs[i] -  outputs[i]), 2);
                     
            //_mse /= outputs.length;
    }
    
    // calculate Errors of each neuron
    private void calcNeuronErrors(double[] dOutputVector) {
        
        double error;
        int n_layers= _layers.length ;
                
        // back-propagating
        for (int layer = n_layers - 1; layer >= 0; --layer) {
            
            double[] errorsVector = new double[_outputs.get(layer).length];
            double[] outputsVector = _outputs.get(layer);

            for (int i = 0; i < outputsVector.length; ++i ) {
                
                if ( layer == n_layers - 1){ // first count errors of outputs
                     
                    double y = outputsVector[i];
                    double d = dOutputVector[i];
                    error = getSigmoidDerivative(y) * (d - y); // y * (1-y) * (d-y)
                     
                } else {
                    
                    double sum = 0;    
                    double[] nextLayerErrorsVector = _errors.get(layer + 1); // vector of errors of next layer
                    
                    for (int j = 0; j < nextLayerErrorsVector.length; ++j ){
                        
                        double connectedW = _weights.get(layer + 1)[j][i];
                        sum += nextLayerErrorsVector[j] * connectedW;
                        
                    }
                    
                    error = getSigmoidDerivative(outputsVector[i]) * sum;
                    
                }
               
                errorsVector[i] = error;
            }
            
            
            _errors.set(layer, errorsVector);
           
        }
        
    }
    
     // deltaW = q*d*z
    private void calcNewWeights(double[] inputVector) {
        
        double q = _learningCoefficient;
        double[] errorsVector;
        double[] neuronWeightDeltaVector;
       
        
        
        for (int layer = 0; layer < _layers.length; ++layer ) {
            
           errorsVector = _errors.get(layer);
           double[][] newWeightDelta = _weights.get(layer);
            
           for (int i = 0; i < errorsVector.length; ++i ) {
               
               double d = _errors.get(layer)[i];               
               double deltaW = q * d;
               
               if (layer  == 0 ) 
                   neuronWeightDeltaVector = Matrix.multiply(inputVector, deltaW);
               else {
                   inputVector = _outputs.get(layer - 1);
                   neuronWeightDeltaVector = Matrix.multiply(inputVector, deltaW);
               }
                              
               for (int j = 0; j < neuronWeightDeltaVector.length; ++j){
                   newWeightDelta[i][j] += neuronWeightDeltaVector[j];
               }
               
           } // neuron
           
           _weights.set(layer, newWeightDelta);
            
        } //layer
        
    }
    
    
    private double getSigmoidDerivative (double input) {
        return input * (1 - input);
    }
    
    
    
    public void test(ArrayList<double[]> testTrainData, ArrayList<double[]> testDOutputs) { 
    
        double [] inputVector;
        double [] dOutputVector;
             
            // we go input by input
            for (int i = 0; i < testTrainData.size(); ++i ) {
                inputVector = testTrainData.get(i); // vector for one input from batch
                dOutputVector = testDOutputs.get(i); // vector of desired output for this input 
                
                calcOutputs(inputVector);      
               
                double[] outputVector = _outputs.get(_layers.length - 1);
                
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
    
    public static ArrayList<double[]> createTrainDataXOR() {
          
          ArrayList<double[]> trainData = new ArrayList<>();
          
          trainData.add(new double[]{0,0});
          trainData.add(new double[]{0,1});
          trainData.add(new double[]{1,0});
          trainData.add(new double[]{1,1});
        
          return trainData;
    }
    
    
    public static ArrayList<double[]> createDesiredOutputDataXOR() {
          
          ArrayList<double[]> dOutputs = new ArrayList<>();
           
          
            dOutputs.add(new double[]{0});
            dOutputs.add(new double[]{1});
            dOutputs.add(new double[]{1});
            dOutputs.add(new double[]{0});
        
          return dOutputs;
    }
    
    
    public static void main(String[] args) {
        
        
        /* 
        // implementing XOR function
        //2 inputs, 2 hidden layers, 1 output layer. Can store any number of hidden layers
        int n_inputs = 2; 
        int[] layers = {5, 1};  
        PerceptronBP perceptron = new PerceptronBP(n_inputs, layers);
        
        // preparing data                  
        ArrayList<double[]> trainData = createTrainDataXOR();
        ArrayList<double[]> dOutputs = createDesiredOutputDataXOR();
        
        //let's learn
        int maxEpochs = 5000;
        perceptron.learn(trainData, dOutputs, maxEpochs);
        perceptron.test(trainData, dOutputs);
        */
        
        // prepare trainData
        ArrayList<double[]> trainData = new ArrayList<>();
        ArrayList<double[]> dOutputs = new ArrayList<>();
        
      
        //neurons: {A, B, C, D, E}. e.g. desired output for neuron B is {0, 1, 0, 0, 0}
        {
            trainData.add(GetImageInputs.get("img/a1.bmp")); dOutputs.add(new double[]{1, 0, 0, 0, 0});
            trainData.add(GetImageInputs.get("img/a2.bmp")); dOutputs.add(new double[]{1, 0, 0, 0, 0});
            trainData.add(GetImageInputs.get("img/a3.bmp")); dOutputs.add(new double[]{1, 0, 0, 0, 0});
            trainData.add(GetImageInputs.get("img/a4.bmp")); dOutputs.add(new double[]{1, 0, 0, 0, 0});
            trainData.add(GetImageInputs.get("img/a5.bmp")); dOutputs.add(new double[]{1, 0, 0, 0, 0});
            
            trainData.add(GetImageInputs.get("img/b1.bmp")); dOutputs.add(new double[]{0, 1, 0, 0, 0});
            trainData.add(GetImageInputs.get("img/b2.bmp")); dOutputs.add(new double[]{0, 1, 0, 0, 0});
            trainData.add(GetImageInputs.get("img/b3.bmp")); dOutputs.add(new double[]{0, 1, 0, 0, 0});
            trainData.add(GetImageInputs.get("img/b4.bmp")); dOutputs.add(new double[]{0, 1, 0, 0, 0});
            
            trainData.add(GetImageInputs.get("img/c1.bmp")); dOutputs.add(new double[]{0, 0, 1, 0, 0});
            trainData.add(GetImageInputs.get("img/c2.bmp")); dOutputs.add(new double[]{0, 0, 1, 0, 0});
            trainData.add(GetImageInputs.get("img/c3.bmp")); dOutputs.add(new double[]{0, 0, 1, 0, 0});
            trainData.add(GetImageInputs.get("img/c4.bmp")); dOutputs.add(new double[]{0, 0, 1, 0, 0});
            
            trainData.add(GetImageInputs.get("img/d1.bmp")); dOutputs.add(new double[]{0, 0, 0, 1, 0});
            trainData.add(GetImageInputs.get("img/d2.bmp")); dOutputs.add(new double[]{0, 0, 0, 1, 0});
            trainData.add(GetImageInputs.get("img/d3.bmp")); dOutputs.add(new double[]{0, 0, 0, 1, 0});
            trainData.add(GetImageInputs.get("img/d4.bmp")); dOutputs.add(new double[]{0, 0, 0, 1, 0});
            
            trainData.add(GetImageInputs.get("img/e1.bmp")); dOutputs.add(new double[]{0, 0, 0, 0, 1});
            trainData.add(GetImageInputs.get("img/e2.bmp")); dOutputs.add(new double[]{0, 0, 0, 0, 1});
            trainData.add(GetImageInputs.get("img/e3.bmp")); dOutputs.add(new double[]{0, 0, 0, 0, 1});
            trainData.add(GetImageInputs.get("img/e4.bmp")); dOutputs.add(new double[]{0, 0, 0, 0, 1});
        }
        
        
        int n_inputs = 24; 
        int[] layers = {10, 5};  
        PerceptronBP perceptron = new PerceptronBP(n_inputs, layers);
        
        //let's learn
        int maxEpochs = 10000;
        perceptron.learn(trainData, dOutputs, maxEpochs);
        
        
        System.out.println("NOW TESTING!");
        ArrayList<double[]> testTrainData = new ArrayList<>();
        ArrayList<double[]> testDOutputs = new ArrayList<>();
        
        // test data
         {
            testTrainData.add(GetImageInputs.get("img/a6.bmp")); testDOutputs.add(new double[]{1, 0, 0, 0, 0});   
            testTrainData.add(GetImageInputs.get("img/b5.bmp")); testDOutputs.add(new double[]{0, 1, 0, 0, 0});            
            testTrainData.add(GetImageInputs.get("img/c5.bmp")); testDOutputs.add(new double[]{0, 0, 1, 0, 0});            
            testTrainData.add(GetImageInputs.get("img/d5.bmp")); testDOutputs.add(new double[]{0, 0, 0, 1, 0});            
            testTrainData.add(GetImageInputs.get("img/e5.bmp")); testDOutputs.add(new double[]{0, 0, 0, 0, 1});
        }
        perceptron.test(testTrainData, testDOutputs);
    }
    
    
    
    //--------FIELDS-------
    
    // these are global weights. e.g. _weights.get(0) - get weight matrix for the 1st layer.
    ArrayList<double[][]> _weights = new ArrayList<>();     
    
    
    // this is accumulating all outputs of neural network. Updated every epoch.
    ArrayList<double[]> _outputs = new ArrayList<>();
    
    // this is accumulating errors of each neuron. Updated every epoch
    ArrayList<double[]> _errors = new ArrayList<>();
    
    // amount of layers and amount of neurons in each layer
    private int[] _layers;
    
    private double _mse;
    
    final static double _learningCoefficient = 0.1;
    final static double _lambda = 1;
    
}
