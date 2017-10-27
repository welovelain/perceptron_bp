package perceptronbp.neuralnetwork.traindata;

import java.util.Collections;
import java.util.List;

public class TrainData {
    private List<InputAndDesiredOutput> inputAndDesiredOutputList;
    private int size;

    public TrainData(List<InputAndDesiredOutput> inputAndDesiredOutputList) {
        this.inputAndDesiredOutputList = inputAndDesiredOutputList;
        size = inputAndDesiredOutputList.size();
    }

    public void setInputAndDesiredOutputList(List<InputAndDesiredOutput> inputAndDesiredOutputList) {
        this.inputAndDesiredOutputList = inputAndDesiredOutputList;
    }

    public int size() {
        return size;
    }

    public double[] getInput(int i) {
        if (i > size) {
            throw new IllegalArgumentException("Can't return result: outbound of traindata");
        }
        return inputAndDesiredOutputList.get(i).getInput();
    }

    public double[] getDesiredOutput(int i) {
        if (i > size) {
            throw new IllegalArgumentException("Can't return result: outbound of traindata");
        }
        return inputAndDesiredOutputList.get(i).getDesiredOutput();
    }

    public void shuffle() {
        Collections.shuffle(inputAndDesiredOutputList);
    }


}
