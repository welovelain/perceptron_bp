package perceptronbp.neuralnetwork.activationfunctions;

public class SigmoidActivationFunction implements ActivationFunction {

    /**
     * Sigmoid Activation Function:
     *  1 / (1+e^(-lambda*x))
     */
    @Override
    public double[] activate(double [] inputs, double lambda) {
        double[] outputs = new double[inputs.length];
        double result;

        for (int i = 0; i < inputs.length; ++i ){
            result = 1 / (1 + Math.exp(-1 * lambda * inputs[i]));
            outputs[i] = result;
        }

        return outputs;
    }

    @Override
    public double getDerivative(double input) {
        return input * (1 - input);
    }
}
