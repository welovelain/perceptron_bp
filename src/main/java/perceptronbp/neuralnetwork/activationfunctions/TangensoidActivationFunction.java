package perceptronbp.neuralnetwork.activationfunctions;

import perceptronbp.neuralnetwork.activationfunctions.ActivationFunction;

public class TangensoidActivationFunction implements ActivationFunction {

    private double lambda;

    public TangensoidActivationFunction(double lambda) {
        if (lambda <= 0) {
            throw new IllegalArgumentException("Lambda must be > 0");
        }
        this.lambda = lambda;
    }

    /**
     * Tangensoid Activation Function:
     *  2 / (1+e^(-lambda*n)) - 1
     */
    @Override
    public float[] activate(float [] inputs) {
        float[] outputs = new float[inputs.length];
        float result;

        for (int i = 0; i < inputs.length; ++i ){
            result = (float)((2 / (1 + Math.exp(-lambda * inputs[i]))) - 1);
            outputs[i] = result;
        }

        return outputs;
    }

    @Override
    public float getDerivative(float input) {
        return (float)(1 - Math.pow(input, 2));
    }
}
