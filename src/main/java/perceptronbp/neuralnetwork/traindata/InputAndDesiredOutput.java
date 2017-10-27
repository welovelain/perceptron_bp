package perceptronbp.neuralnetwork.traindata;

public class InputAndDesiredOutput {
    // input vector
    private double[] input;

    // desired output vector
    private double[] desiredOutput;

    public InputAndDesiredOutput(double[] input, double[] desiredOutput) {
        this.input = input;
        this.desiredOutput = desiredOutput;
    }

    public double[] getInput() {
        return input;
    }

    public void setInput(double[] input) {
        this.input = input;
    }

    public double[] getDesiredOutput() {
        return desiredOutput;
    }

    public void setDesiredOutput(double[] desiredOutput) {
        this.desiredOutput = desiredOutput;
    }
}
