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
