package perceptronbp.neuralnetwork.activationfunctions;

import perceptronbp.neuralnetwork.activationfunctions.ActivationFunction;

public class SigmoidActivationFunction implements ActivationFunction {

    private float lambda;

    public SigmoidActivationFunction(float lambda) {
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
    public float[] activate(float [] inputs) {
        float[] outputs = new float[inputs.length];
        float result;

        for (int i = 0; i < inputs.length; ++i ){
            result = (float)(1 / (1 + Math.exp(-lambda * inputs[i])));
            outputs[i] = result;
        }

        return outputs;
    }

    @Override
    public float getDerivative(float input) {
        return (float)(input * (1 - input));
    }
}
