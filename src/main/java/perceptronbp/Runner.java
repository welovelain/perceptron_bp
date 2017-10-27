package perceptronbp;

import perceptronbp.input.ImageInputExtractor;
import perceptronbp.neuralnetwork.Perceptron;
import perceptronbp.neuralnetwork.activationfunctions.SigmoidActivationFunction;
import perceptronbp.neuralnetwork.activationfunctions.TangensoidActivationFunction;
import perceptronbp.neuralnetwork.layers.Layer;
import perceptronbp.neuralnetwork.layers.LayerFactory;
import perceptronbp.neuralnetwork.traindata.InputAndDesiredOutput;
import perceptronbp.neuralnetwork.traindata.TrainData;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Main class to run and test the Perceptron;
 */
public class Runner {

    public static void main(String[] args) throws Exception {
        runLetterRecognizeSuite();
//        runXorSuite();
    }

    public static void runLetterRecognizeSuite() throws IOException {
        long startTimer = System.nanoTime();

        float lambda = 1f;

        List<Layer> layers = new LayerFactory(24)
                .addLayer(20, new TangensoidActivationFunction(lambda))
                .addLayer(5, new SigmoidActivationFunction(lambda))
                .getLayers();

        Perceptron perceptron = new Perceptron.Builder(layers)
                .withLearningCoefficient(0.1f)
                .build();

        // prepare trainData
        ArrayList<float[]> trainData = new ArrayList<>();
        ArrayList<float[]> dOutputs = new ArrayList<>();

        int filledColorCode = -14503604;
        int pixelSize = 36;
        int height = 6;
        int width = 4;
        int border = 1;
        ImageInputExtractor inputExtractor = new ImageInputExtractor(filledColorCode, pixelSize, height, width, border);

        //let's learn
        int maxEpochs = 30000;
        perceptron.learn(createLetterRecognizeData(), maxEpochs);

        System.out.println("NOW TESTING!");

        // test data
        testLetter(perceptron, inputExtractor.get("img/a6.bmp"), new float[]{1, 0, 0, 0, 0});
        testLetter(perceptron, inputExtractor.get("img/b5.bmp"), new float[]{0, 1, 0, 0, 0});
        testLetter(perceptron, inputExtractor.get("img/c5.bmp"), new float[]{0, 0, 1, 0, 0});
        testLetter(perceptron, inputExtractor.get("img/d5.bmp"), new float[]{0, 0, 0, 1, 0});
        testLetter(perceptron, inputExtractor.get("img/e5.bmp"), new float[]{0, 0, 0, 0, 1});

        float passed = ((float)(System.nanoTime() - startTimer)/1000000000);
        System.out.println("Time passed: " + passed);
    }

    public static void testLetter(Perceptron perceptron, float[] input, float[] desiredOutput) {
        printInputLetter(input);
        System.out.println("\r\nI think it's:");
        printOutputLetter(perceptron.calculateOutput(input));
        System.out.println("\r\n\r\n\r\n");
    }

    public static TrainData createLetterRecognizeData() throws IOException {
        int filledColorCode = -14503604;
        int pixelSize = 36;
        int height = 6;
        int width = 4;
        int border = 1;
        ImageInputExtractor inputExtractor = new ImageInputExtractor(filledColorCode, pixelSize, height, width, border);

        TrainData trainData = new TrainData();
        //neurons: {A, B, C, D, E}. e.g. desired output for neuron B is {0, 1, 0, 0, 0}
        // A
        trainData.add(inputExtractor.get("img/a1.bmp"), new float[]{1, 0, 0, 0, 0});
        trainData.add(inputExtractor.get("img/a2.bmp"), new float[]{1, 0, 0, 0, 0});
        trainData.add(inputExtractor.get("img/a3.bmp"), new float[]{1, 0, 0, 0, 0});
        trainData.add(inputExtractor.get("img/a4.bmp"), new float[]{1, 0, 0, 0, 0});
        trainData.add(inputExtractor.get("img/a5.bmp"), new float[]{1, 0, 0, 0, 0});

        // B
        trainData.add(inputExtractor.get("img/b1.bmp"), new float[]{0, 1, 0, 0, 0});
        trainData.add(inputExtractor.get("img/b2.bmp"), new float[]{0, 1, 0, 0, 0});
        trainData.add(inputExtractor.get("img/b3.bmp"), new float[]{0, 1, 0, 0, 0});
        trainData.add(inputExtractor.get("img/b4.bmp"), new float[]{0, 1, 0, 0, 0});

        // C
        trainData.add(inputExtractor.get("img/c1.bmp"), new float[]{0, 0, 1, 0, 0});
        trainData.add(inputExtractor.get("img/c2.bmp"), new float[]{0, 0, 1, 0, 0});
        trainData.add(inputExtractor.get("img/c3.bmp"), new float[]{0, 0, 1, 0, 0});
        trainData.add(inputExtractor.get("img/c4.bmp"), new float[]{0, 0, 1, 0, 0});

        // D
        trainData.add(inputExtractor.get("img/d1.bmp"), new float[]{0, 0, 0, 1, 0});
        trainData.add(inputExtractor.get("img/d2.bmp"), new float[]{0, 0, 0, 1, 0});
        trainData.add(inputExtractor.get("img/d3.bmp"), new float[]{0, 0, 0, 1, 0});
        trainData.add(inputExtractor.get("img/d4.bmp"), new float[]{0, 0, 0, 1, 0});

        // E
        trainData.add(inputExtractor.get("img/e1.bmp"), new float[]{0, 0, 0, 0, 1});
        trainData.add(inputExtractor.get("img/e2.bmp"), new float[]{0, 0, 0, 0, 1});
        trainData.add(inputExtractor.get("img/e3.bmp"), new float[]{0, 0, 0, 0, 1});
        trainData.add(inputExtractor.get("img/e4.bmp"), new float[]{0, 0, 0, 0, 1});

        return trainData;
    }

    private static void printInputLetter (float[] input) {
        String result="";
        int count = 0;
        for (int i = 0; i < 6; ++i){
            for (int j = 0; j < 4; ++j) {

                if (input[count] == 1) result += "1";
                else result += " ";

                ++count;
            }
            result+="\r\n";
        }
        System.out.println(result);
    }

    private static void printOutputLetter (float[] input) {
        float maxValue = input[0];
        int maxValueIndex = 0;

        for (int i = 1; i < input.length; ++i) {
            if (input[i] > maxValue) {
                maxValue = input[i];
                maxValueIndex = i;
            }
        }

        char result =  'a';
        result += maxValueIndex;
        System.out.print(result);
    }

    public static void runXorSuite() {
        // implementing XOR function
        //2 inputs, 2 hidden layers, 1 output layer. Can store any number of hidden layers
        float lambda = 1f;

        List<Layer> layers2 = new LayerFactory(2)
                .addLayer(20, new TangensoidActivationFunction(lambda))
                .addLayer(18, new TangensoidActivationFunction(lambda))
                .addLayer(1, new TangensoidActivationFunction(lambda))
                .getLayers();

        Perceptron perceptron2 =  new Perceptron.Builder(layers2)
                .withLearningCoefficient(0.01f)
                .build();

        // learning
        int maxEpochs = 50000;
        perceptron2.learn(createTrainDataXOR(), maxEpochs);

        // testing
        System.out.println("Should be false: " + testXor(perceptron2.calculateOutput(new float[]{0,0}), 0.1f));
        System.out.println("Should be true: " + testXor(perceptron2.calculateOutput(new float[]{0,1}), 0.1f));
        System.out.println("Should be true: " + testXor(perceptron2.calculateOutput(new float[]{1,0}), 0.1f));
        System.out.println("Should be false: " + testXor(perceptron2.calculateOutput(new float[]{0,0}), 0.1f));

    }
    public static boolean testXor(float[] value, float threshold) {
        return value[0] > threshold;
    }

    public static TrainData createTrainDataXOR() {
        List<InputAndDesiredOutput> inputAndDesiredOutputList = new ArrayList<>(4);
        inputAndDesiredOutputList.add(new InputAndDesiredOutput(new float[]{0,0}, new float[]{0}));
        inputAndDesiredOutputList.add(new InputAndDesiredOutput(new float[]{0,1}, new float[]{1}));
        inputAndDesiredOutputList.add(new InputAndDesiredOutput(new float[]{1,0}, new float[]{1}));
        inputAndDesiredOutputList.add(new InputAndDesiredOutput(new float[]{1,1}, new float[]{0}));

        TrainData trainData = new TrainData(inputAndDesiredOutputList);

        return trainData;
    }
}
