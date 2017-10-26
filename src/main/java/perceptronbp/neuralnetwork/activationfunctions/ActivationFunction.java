package perceptronbp.neuralnetwork.activationfunctions;

public interface ActivationFunction {
    double[] activate(double [] inputs, double lambda);
    double getDerivative(double input);
}
