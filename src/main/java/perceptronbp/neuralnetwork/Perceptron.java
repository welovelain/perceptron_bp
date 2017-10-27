package perceptronbp.neuralnetwork;

import perceptronbp.matrix.Matrix;
import perceptronbp.neuralnetwork.layers.Layer;
import perceptronbp.neuralnetwork.traindata.TrainData;

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

    public void learn(TrainData trainData, int maxEpochs) {
        double [] inputVector;
        double [] dOutputVector;

        for (int epoch = 0; epoch < maxEpochs; ++epoch) {
            meanSquaredError = 0;

            // we go input by input
            for (int i = 0; i < trainData.size(); ++i ) {
                inputVector = trainData.getInput(i); // vector for one input from batch
                dOutputVector = trainData.getDesiredOutput(i); // vector of desired output for this input

                calculateOutput(inputVector);
                calcMSE(dOutputVector);
                calcNeuronErrors(dOutputVector);
                calcNewWeights(inputVector);
            }

            // shuffle trainData and dOutputs here
            trainData.shuffle();
            System.out.println("epoch: " + epoch + ", MSE: " + meanSquaredError);
        }
    }

    public double[] calculateOutput(double[] initialInput) {
        double[] input = initialInput;
        for (Layer layer: layerList) {
            layer.calculateOutput(input);
            input = layer.getCurrentOutput();
        }
        return input;
    }

    // MSE = 1/n * SUM [(d_o - y_o)^2]
    private void calcMSE (double[] dOutputs) {

        Layer finalLayer = layerList.get(layerList.size() - 1);
        double[] finalOutput = finalLayer.getCurrentOutput();
        for (int i = 0; i < finalOutput.length; ++ i ) {
            meanSquaredError += (Math.pow((dOutputs[i] - finalOutput[i]), 2));
        }
//        meanSquaredError /= finalOutput.length;
    }

    // calculate Errors of each neuron. Back-propagating from last layer to the first
    public void calcNeuronErrors(double[] desiredOutputVector) {
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

            // TODO -> Multiply by matrix
            for (int i = 0; i < errorVector.length - 1; ++i) {
                for (int j = 0; j < inputVector.length; ++j) {
                    weightDelta[i][j] = errorVector[i] * inputVector[j] * learningCoefficient;
                }
            }

            double[][] newWeights = Matrix.add(oldWeight, weightDelta);
            layer.setWeights(newWeights);
            inputVector = layer.getCurrentOutput();
        }
    }

    /* Static methods */

    //Add 1 to inputs at the end
    public static double[] addBias(double[] inputs){
        double[] outputs = Arrays.copyOf(inputs, inputs.length+1);
        outputs[outputs.length - 1] = 1;
        return outputs;
    }
}
