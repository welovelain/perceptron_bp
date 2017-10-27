package perceptronbp.neuralnetwork.layers;

import perceptronbp.matrix.SimpleMatrixSolver;
import perceptronbp.neuralnetwork.Perceptron;
import perceptronbp.neuralnetwork.activationfunctions.ActivationFunction;
import perceptronbp.neuralnetwork.activationfunctions.TangensoidActivationFunction;

import java.util.Arrays;

public class Layer {

    private int counter;
    private double[][] weights;
    private int amountOfNeurons;

    private ActivationFunction DEFAULT_ACTIVATION_FUNCTION = new TangensoidActivationFunction(1d);
    private ActivationFunction activationFunction = DEFAULT_ACTIVATION_FUNCTION;

    private double[] currentOutput;
    private double[] errorVector;

    public Layer(int counter, int amountOfNeurons, int amountOfPreviousLayerOutputs) {
        this.counter = counter;
        this.amountOfNeurons = amountOfNeurons;

        // init weights
        weights = SimpleMatrixSolver.random(amountOfNeurons, amountOfPreviousLayerOutputs);
    }

    public double[][] getWeights() {
        return weights;
    }

    public void setWeights(double[][] weights) {
        this.weights = weights;
    }

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    public void setActivationFunction(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    public int getAmountOfNeurons() {
        return amountOfNeurons;
    }

    public int getCounter() {
        return counter;
    }

    public void setCounter(int counter) {
        this.counter = counter;
    }

    public double[] getCurrentOutput() {
        return currentOutput;
    }

    public double[] getErrorVector() {
        return errorVector;
    }

    /**
     * Calculate output of the layer : acticationFunction(Weights * X);
     * @param input input vector X
     * @return result of activation function with the Weights * X input.
     */
    public void calculateOutput(double[] input) {
        input = Perceptron.addBias(input);
        double[] net = SimpleMatrixSolver.multiply(weights, input);
        currentOutput = activationFunction.activate(net);
    }

    /**
     * Use this method if the layer is the last layer.
     */
    public void calculateError(double[] desiredOutputVector) {
        errorVector = new double[currentOutput.length];

        double error;
        for (int i = 0; i < currentOutput.length; ++i) {
            double y = currentOutput[i];
            double d = desiredOutputVector[i];
            error = activationFunction.getDerivative(y) * (d - y);
            errorVector[i] = error;
        }
    }

    /**
     * Use this method for calculating errors for hidden vectors.
     * @param desiredOutputVector
     * @param nextLayer
     */
    public void calculateError(double[] desiredOutputVector, Layer nextLayer) {
        double error;
        double sum;
        errorVector = new double[currentOutput.length];
        double[] nextLayerErrorVector = nextLayer.getErrorVector();

        for (int i = 0; i < currentOutput.length; ++i) {
            sum = 0d;
            for (int j = 0; j < nextLayerErrorVector.length; ++j) {
                double connectedWeights = nextLayer.getWeights()[j][i];
                sum += nextLayerErrorVector[j] * connectedWeights;
            }
            error = activationFunction.getDerivative(currentOutput[i]) * sum;
            errorVector[i] = error;
        }
    }
}
