package perceptronbp.neuralnetwork.activationfunctions;

public interface ActivationFunction {
    float[] activate(float [] inputs);
    float getDerivative(float input);
}
