package perceptronbp.neuralnetwork.layers;

import perceptronbp.matrix.Matrix;
import perceptronbp.neuralnetwork.Perceptron;
import perceptronbp.neuralnetwork.activationfunctions.ActivationFunction;
import perceptronbp.neuralnetwork.activationfunctions.TangensoidActivationFunction;

public class Layer {

    private int counter;
    private float[][] weights;
    private int amountOfNeurons;

    private ActivationFunction DEFAULT_ACTIVATION_FUNCTION = new TangensoidActivationFunction(1d);
    private ActivationFunction activationFunction = DEFAULT_ACTIVATION_FUNCTION;

    private float[] currentOutput;
    private float[] errorVector;

    public Layer(int counter, int amountOfNeurons, int amountOfPreviousLayerOutputs) {
        this.counter = counter;
        this.amountOfNeurons = amountOfNeurons;

        // init weights
        weights = Matrix.random(amountOfNeurons, amountOfPreviousLayerOutputs);
    }

    public float[][] getWeights() {
        return weights;
    }

    public void setWeights(float[][] weights) {
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

    public float[] getCurrentOutput() {
        return currentOutput;
    }

    public float[] getErrorVector() {
        return errorVector;
    }

    /**
     * Calculate output of the layer : acticationFunction(Weights * X);
     * @param input input vector X
     * @return result of activation function with the Weights * X input.
     */
    public void calculateOutput(float[] input) {
        input = Perceptron.addBias(input);
        float[] net = Matrix.multiply(weights, input);
        currentOutput = activationFunction.activate(net);
    }

    /**
     * Use this method if the layer is the last layer.
     */
    public void calculateError(float[] desiredOutputVector) {
        errorVector = new float[currentOutput.length];

        float error;
        for (int i = 0; i < currentOutput.length; ++i) {
            float y = currentOutput[i];
            float d = desiredOutputVector[i];
            error = activationFunction.getDerivative(y) * (d - y);
            errorVector[i] = error;
        }
    }

    /**
     * Use this method for calculating errors for hidden vectors.
     * @param desiredOutputVector
     * @param nextLayer
     */
    public void calculateError(float[] desiredOutputVector, Layer nextLayer) {
        float error;
        float sum;
        errorVector = new float[currentOutput.length];
        float[] nextLayerErrorVector = nextLayer.getErrorVector();

        for (int i = 0; i < currentOutput.length; ++i) {
            sum = 0f;
            for (int j = 0; j < nextLayerErrorVector.length; ++j) {
                float connectedWeights = nextLayer.getWeights()[j][i];
                sum += nextLayerErrorVector[j] * connectedWeights;
            }
            error = activationFunction.getDerivative(currentOutput[i]) * sum;
            errorVector[i] = error;
        }
    }
}
