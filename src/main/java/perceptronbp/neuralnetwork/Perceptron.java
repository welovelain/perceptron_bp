package perceptronbp.neuralnetwork;

import perceptronbp.matrix.SimpleMatrixSolver;
import perceptronbp.neuralnetwork.layers.Layer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.ListIterator;

public class Perceptron {

    private List<Layer> layerList;
    private double meanSquaredError = 0;
    private double learningCoefficient;

    private Perceptron(Builder builder) {
        this.layerList = builder.layerList;
        this.learningCoefficient = builder.learningCoefficient;
    }

    public static class Builder {

        private double DEFAULT_LEARNING_COEFFICIENT = 0.1d;
        private List<Layer> layerList;
        private double learningCoefficient = DEFAULT_LEARNING_COEFFICIENT;

        public Builder(List<Layer> layerList) {
            if (layerList.isEmpty() || layerList == null) {
                throw new IllegalArgumentException("Layers shouldn't be 0 length");
            }
            this.layerList = layerList;
        }

        public Builder withLearningCoefficient(double learningCoefficient) {

            if (learningCoefficient <= 0) {
                throw new IllegalArgumentException("Learning coefficient should be > 0");
            }
            this.learningCoefficient = learningCoefficient;
            return this;
        }

        public Perceptron build() {
            return new Perceptron(this);
        }
    }

    public void learn(ArrayList<double[]> trainData, ArrayList<double[]> dOutputs, int maxEpochs) {
        double [] inputVector;
        double [] dOutputVector;

        for (int epoch = 0; epoch < maxEpochs; ++epoch) {
            meanSquaredError = 0;

            // we go input by input
            for (int i = 0; i < trainData.size(); ++i ) {
                inputVector = trainData.get(i); // vector for one input from batch
                dOutputVector = dOutputs.get(i); // vector of desired output for this input

                calculateOutput(inputVector);
                calcMSE(dOutputVector);
                calcNeuronErrors(dOutputVector);
                calcNewWeights(inputVector);
            }

            // shuffle trainData and dOutputs here
           System.out.println("epoch: " + epoch + ", MSE: " + meanSquaredError);
        }
    }

    public void calculateOutput(double[] initialInput) {
        double[] input = initialInput;
        for (Layer layer: layerList) {
            layer.calculateOutput(input);
            input = layer.getCurrentOutput();
        }
    }

    // MSE = 1/n * SUM [(d_o - y_o)^2]
    private void calcMSE (double[] dOutputs) {

        Layer finalLayer = layerList.get(layerList.size() - 1);
        double[] finalOutput = finalLayer.getCurrentOutput();
        for (int i = 0; i < finalOutput.length; ++ i ) {
            meanSquaredError += (Math.pow((dOutputs[i] - finalOutput[i]), 2));
        }
        meanSquaredError /= finalOutput.length;
    }

    // calculate Errors of each neuron
    public void calcNeuronErrors(double[] desiredOutputVector) {

        // back-propagating from last layer to the first
        ListIterator<Layer> li = layerList.listIterator(layerList.size());
        Layer lastLayer = li.previous();
        lastLayer.calculateError(desiredOutputVector);

        while (li.hasPrevious()) {
            Layer currentLayer = li.previous();
            currentLayer.calculateError(desiredOutputVector, lastLayer);
            lastLayer = currentLayer;
        }
    }

     // deltaW = lCoef * d * z
    private void calcNewWeights(double[] inputVector) {
        double[] errorVector;

        for (Layer layer: layerList) {
            inputVector = addBias(inputVector);

            double[][] oldWeight = layer.getWeights();
            errorVector = layer.getErrorVector();

            double[][] weightDelta = new double[errorVector.length][inputVector.length];

            for (int i = 0; i < errorVector.length - 1; ++i) {
                for (int j = 0; j < inputVector.length; ++j) {
                    weightDelta[i][j] = errorVector[i] * inputVector[j] * learningCoefficient;
                }
            }

            double[][] newWeights = SimpleMatrixSolver.add(oldWeight, weightDelta);
            layer.setWeights(newWeights);
            inputVector = layer.getCurrentOutput();
        }
    }

    public void test(ArrayList<double[]> testTrainData, ArrayList<double[]> testDOutputs) {

        double [] inputVector;
        double [] dOutputVector;
        Layer lastLayer = layerList.get(layerList.size() - 1);

        // we go input by input
        for (int i = 0; i < testTrainData.size(); ++i ) {
            inputVector = testTrainData.get(i); // vector for one input from batch
            dOutputVector = testDOutputs.get(i); // vector of desired output for this input

            calculateOutput(inputVector);

            double[] outputVector = lastLayer.getCurrentOutput();

            printInputLetter(inputVector);

            System.out.print("Expected result is: ");
            printOutputLetter(dOutputVector);

            System.out.print("\r\nPerceptron thinks that the result is: ");
            printOutputLetter(outputVector);
            System.out.println();
            System.out.println("------------------------");
        }
    }

    public void testXor(ArrayList<double[]> testTrainData, ArrayList<double[]> testDOutputs) {

        double [] inputVector;
        double [] dOutputVector;
        Layer lastLayer = layerList.get(layerList.size() - 1);

        // we go input by input
        for (int i = 0; i < testTrainData.size(); ++i ) {
            inputVector = testTrainData.get(i); // vector for one input from batch
            dOutputVector = testDOutputs.get(i); // vector of desired output for this input

            calculateOutput(inputVector);

            double[] outputVector = lastLayer.getCurrentOutput();

            System.out.println(Arrays.toString(inputVector));

            System.out.print("Expected result is: ");
            System.out.println(Arrays.toString(dOutputVector));

            System.out.print("\r\nPerceptron thinks that the result is: ");
            System.out.println(Arrays.toString(outputVector));
            System.out.println();
            System.out.println("------------------------");
        }

    }

    //----------STATIC METHODS-----------

    //Add 1 to inputs at the end
    public static double[] addBias(double[] inputs){
        double[] outputs = Arrays.copyOf(inputs, inputs.length+1);
        outputs[outputs.length - 1] = 1;
        return outputs;
    }

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

}
