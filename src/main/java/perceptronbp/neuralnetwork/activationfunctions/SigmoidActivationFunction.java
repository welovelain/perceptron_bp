package perceptronbp.neuralnetwork.activationfunctions;

import perceptronbp.neuralnetwork.activationfunctions.ActivationFunction;

public class SigmoidActivationFunction implements ActivationFunction {

    private double lambda;

    public SigmoidActivationFunction(double lambda) {
        if (lambda <= 0) {
            throw new IllegalArgumentException("Lambda must be > 0");
        }
        this.lambda = lambda;
    }
    /**
     * Sigmoid Activation Function:
     *  1 / (1+e^(-lambda*n))
     */
    @Override
    public double[] activate(double [] inputs) {
        double[] outputs = new double[inputs.length];
        double result;

        for (int i = 0; i < inputs.length; ++i ){
            result = 1 / (1 + Math.exp(-lambda * inputs[i]));
            outputs[i] = result;
        }

        return outputs;
    }

    @Override
    public double getDerivative(double input) {
        return input * (1 - input);
    }
}
