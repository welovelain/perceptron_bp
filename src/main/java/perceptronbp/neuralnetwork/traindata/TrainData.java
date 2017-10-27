package perceptronbp.neuralnetwork.traindata;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class TrainData {
    private List<InputAndDesiredOutput> inputAndDesiredOutputList;

    public TrainData() {
        inputAndDesiredOutputList = new ArrayList<>(0);
    }

    public TrainData(List<InputAndDesiredOutput> inputAndDesiredOutputList) {
        this.inputAndDesiredOutputList = inputAndDesiredOutputList;
    }

    public void setInputAndDesiredOutputList(List<InputAndDesiredOutput> inputAndDesiredOutputList) {
        this.inputAndDesiredOutputList = inputAndDesiredOutputList;
    }

    public void add(float[] input, float[] output) {

        if (!inputAndDesiredOutputList.isEmpty()) {
            InputAndDesiredOutput previous = inputAndDesiredOutputList.get(inputAndDesiredOutputList.size() - 1);

            int previousInputLength = previous.getInput().length;
            int previousDesiredOutputLength = previous.getDesiredOutput().length;

            if (previousInputLength != input.length) {
                throw new IllegalArgumentException
                        (String.format("Previous input has size %s, but new data has input size %s. Sizes can't be different.",
                                previousInputLength, input.length));
            }
            if (previousDesiredOutputLength != output.length) {
                throw new IllegalArgumentException
                        (String.format("Previous desired output has size %s, but new data has desired output size %s. Sizes can't be different.",
                                previousDesiredOutputLength, output.length));
            }
        }


        inputAndDesiredOutputList.add(new InputAndDesiredOutput(input, output));
    }

    public int size() {
        return inputAndDesiredOutputList.size();
    }

    public float[] getInput(int i) {
        if (i > inputAndDesiredOutputList.size()) {
            throw new IllegalArgumentException("Can't return result: outbound of traindata");
        }
        return inputAndDesiredOutputList.get(i).getInput();
    }

    public float[] getDesiredOutput(int i) {
        if (i > inputAndDesiredOutputList.size()) {
            throw new IllegalArgumentException("Can't return result: outbound of traindata");
        }
        return inputAndDesiredOutputList.get(i).getDesiredOutput();
    }

    public void shuffle() {
        Collections.shuffle(inputAndDesiredOutputList);
    }


}
