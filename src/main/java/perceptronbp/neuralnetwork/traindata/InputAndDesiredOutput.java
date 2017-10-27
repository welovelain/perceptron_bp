package perceptronbp.neuralnetwork.traindata;

public class InputAndDesiredOutput {
    // input vector
    private float[] input;

    // desired output vector
    private float[] desiredOutput;

    public InputAndDesiredOutput(float[] input, float[] desiredOutput) {
        this.input = input;
        this.desiredOutput = desiredOutput;
    }

    public float[] getInput() {
        return input;
    }

    public void setInput(float[] input) {
        this.input = input;
    }

    public float[] getDesiredOutput() {
        return desiredOutput;
    }

    public void setDesiredOutput(float[] desiredOutput) {
        this.desiredOutput = desiredOutput;
    }
}
