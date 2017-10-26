package perceptronbp;

import perceptronbp.input.ImageInputExtractor;
import perceptronbp.neuralnetwork.Perceptron;

import java.util.ArrayList;

/**
 * Main class to run the Perceptron;
 */
public class Runner {

    public static void main(String[] args) throws Exception {

        int n_inputs = 24;
        int[] layers = {10, 5};
        Perceptron perceptron = new Perceptron(n_inputs, layers);


        // prepare trainData
        ArrayList<double[]> trainData = new ArrayList<>();
        ArrayList<double[]> dOutputs = new ArrayList<>();

        int filledColorCode = -14503604;
        int pixelSize = 36;
        int height = 6;
        int width = 4;
        int border = 1;
        ImageInputExtractor inputExtractor = new ImageInputExtractor(filledColorCode, pixelSize, height, width, border);

        //neurons: {A, B, C, D, E}. e.g. desired output for neuron B is {0, 1, 0, 0, 0}

        // A
        trainData.add(inputExtractor.get("img/a1.bmp")); dOutputs.add(new double[]{1, 0, 0, 0, 0});
        trainData.add(inputExtractor.get("img/a2.bmp")); dOutputs.add(new double[]{1, 0, 0, 0, 0});
        trainData.add(inputExtractor.get("img/a3.bmp")); dOutputs.add(new double[]{1, 0, 0, 0, 0});
        trainData.add(inputExtractor.get("img/a4.bmp")); dOutputs.add(new double[]{1, 0, 0, 0, 0});
        trainData.add(inputExtractor.get("img/a5.bmp")); dOutputs.add(new double[]{1, 0, 0, 0, 0});

        // B
        trainData.add(inputExtractor.get("img/b1.bmp")); dOutputs.add(new double[]{0, 1, 0, 0, 0});
        trainData.add(inputExtractor.get("img/b2.bmp")); dOutputs.add(new double[]{0, 1, 0, 0, 0});
        trainData.add(inputExtractor.get("img/b3.bmp")); dOutputs.add(new double[]{0, 1, 0, 0, 0});
        trainData.add(inputExtractor.get("img/b4.bmp")); dOutputs.add(new double[]{0, 1, 0, 0, 0});

        // C
        trainData.add(inputExtractor.get("img/c1.bmp")); dOutputs.add(new double[]{0, 0, 1, 0, 0});
        trainData.add(inputExtractor.get("img/c2.bmp")); dOutputs.add(new double[]{0, 0, 1, 0, 0});
        trainData.add(inputExtractor.get("img/c3.bmp")); dOutputs.add(new double[]{0, 0, 1, 0, 0});
        trainData.add(inputExtractor.get("img/c4.bmp")); dOutputs.add(new double[]{0, 0, 1, 0, 0});

        // D
        trainData.add(inputExtractor.get("img/d1.bmp")); dOutputs.add(new double[]{0, 0, 0, 1, 0});
        trainData.add(inputExtractor.get("img/d2.bmp")); dOutputs.add(new double[]{0, 0, 0, 1, 0});
        trainData.add(inputExtractor.get("img/d3.bmp")); dOutputs.add(new double[]{0, 0, 0, 1, 0});
        trainData.add(inputExtractor.get("img/d4.bmp")); dOutputs.add(new double[]{0, 0, 0, 1, 0});

        // E
        trainData.add(inputExtractor.get("img/e1.bmp")); dOutputs.add(new double[]{0, 0, 0, 0, 1});
        trainData.add(inputExtractor.get("img/e2.bmp")); dOutputs.add(new double[]{0, 0, 0, 0, 1});
        trainData.add(inputExtractor.get("img/e3.bmp")); dOutputs.add(new double[]{0, 0, 0, 0, 1});
        trainData.add(inputExtractor.get("img/e4.bmp")); dOutputs.add(new double[]{0, 0, 0, 0, 1});

        //let's learn
        int maxEpochs = 1000;
        perceptron.learn(trainData, dOutputs, maxEpochs);


        System.out.println("NOW TESTING!");
        ArrayList<double[]> testTrainData = new ArrayList<>();
        ArrayList<double[]> testDOutputs = new ArrayList<>();

        // test data
        testTrainData.add(inputExtractor.get("img/a6.bmp")); testDOutputs.add(new double[]{1, 0, 0, 0, 0});
        testTrainData.add(inputExtractor.get("img/b5.bmp")); testDOutputs.add(new double[]{0, 1, 0, 0, 0});
        testTrainData.add(inputExtractor.get("img/c5.bmp")); testDOutputs.add(new double[]{0, 0, 1, 0, 0});
        testTrainData.add(inputExtractor.get("img/d5.bmp")); testDOutputs.add(new double[]{0, 0, 0, 1, 0});
        testTrainData.add(inputExtractor.get("img/e5.bmp")); testDOutputs.add(new double[]{0, 0, 0, 0, 1});
        perceptron.test(testTrainData, testDOutputs);
    }








      /*
        // implementing XOR function
        //2 inputs, 2 hidden layers, 1 output layer. Can store any number of hidden layers
        int n_inputs = 2;
        int[] layers = {5, 1};
        Perceptron perceptron = new Perceptron(n_inputs, layers);

        // preparing data
        ArrayList<double[]> trainData = createTrainDataXOR();
        ArrayList<double[]> dOutputs = createDesiredOutputDataXOR();

        // learning
        int maxEpochs = 5000;
        perceptron.learn(trainData, dOutputs, maxEpochs);

        // testing
        perceptron.test(trainData, dOutputs);
        */
}
