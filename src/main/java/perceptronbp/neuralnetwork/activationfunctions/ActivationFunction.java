package perceptronbp.neuralnetwork.activationfunctions;

public interface ActivationFunction {
    double[] activate(double [] inputs);
    double getDerivative(double input);
}
