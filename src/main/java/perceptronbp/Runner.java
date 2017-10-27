package perceptronbp;

import perceptronbp.input.ImageInputExtractor;
import perceptronbp.neuralnetwork.Perceptron;
import perceptronbp.neuralnetwork.activationfunctions.SigmoidActivationFunction;
import perceptronbp.neuralnetwork.activationfunctions.TangensoidActivationFunction;
import perceptronbp.neuralnetwork.layers.Layer;
import perceptronbp.neuralnetwork.layers.LayerFactory;
import perceptronbp.neuralnetwork.traindata.InputAndDesiredOutput;
import perceptronbp.neuralnetwork.traindata.TrainData;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Main class to run the Perceptron;
 */
public class Runner {

    public static void main(String[] args) throws Exception {

//        double lambda = 1d;

//        List<Layer> layers = new LayerFactory(24)
//                .addLayer(10, new TangensoidActivationFunction(lambda))
//                .addLayer(5, new TangensoidActivationFunction(lambda))
//                .getLayers();
//
//
//        Perceptron perceptron = new Perceptron.Builder(layers)
//                .withLearningCoefficient(0.1d)
//                .build();
//
//        // prepare trainData
//        ArrayList<double[]> trainData = new ArrayList<>();
//        ArrayList<double[]> dOutputs = new ArrayList<>();
//
//        int filledColorCode = -14503604;
//        int pixelSize = 36;
//        int height = 6;
//        int width = 4;
//        int border = 1;
//        ImageInputExtractor inputExtractor = new ImageInputExtractor(filledColorCode, pixelSize, height, width, border);
//
//        //neurons: {A, B, C, D, E}. e.g. desired output for neuron B is {0, 1, 0, 0, 0}
//
//        // A
//        trainData.add(inputExtractor.get("img/a1.bmp")); dOutputs.add(new double[]{1, 0, 0, 0, 0});
//        trainData.add(inputExtractor.get("img/a2.bmp")); dOutputs.add(new double[]{1, 0, 0, 0, 0});
//        trainData.add(inputExtractor.get("img/a3.bmp")); dOutputs.add(new double[]{1, 0, 0, 0, 0});
//        trainData.add(inputExtractor.get("img/a4.bmp")); dOutputs.add(new double[]{1, 0, 0, 0, 0});
//        trainData.add(inputExtractor.get("img/a5.bmp")); dOutputs.add(new double[]{1, 0, 0, 0, 0});
//
//        // B
//        trainData.add(inputExtractor.get("img/b1.bmp")); dOutputs.add(new double[]{0, 1, 0, 0, 0});
//        trainData.add(inputExtractor.get("img/b2.bmp")); dOutputs.add(new double[]{0, 1, 0, 0, 0});
//        trainData.add(inputExtractor.get("img/b3.bmp")); dOutputs.add(new double[]{0, 1, 0, 0, 0});
//        trainData.add(inputExtractor.get("img/b4.bmp")); dOutputs.add(new double[]{0, 1, 0, 0, 0});
//
//        // C
//        trainData.add(inputExtractor.get("img/c1.bmp")); dOutputs.add(new double[]{0, 0, 1, 0, 0});
//        trainData.add(inputExtractor.get("img/c2.bmp")); dOutputs.add(new double[]{0, 0, 1, 0, 0});
//        trainData.add(inputExtractor.get("img/c3.bmp")); dOutputs.add(new double[]{0, 0, 1, 0, 0});
//        trainData.add(inputExtractor.get("img/c4.bmp")); dOutputs.add(new double[]{0, 0, 1, 0, 0});
//
//        // D
//        trainData.add(inputExtractor.get("img/d1.bmp")); dOutputs.add(new double[]{0, 0, 0, 1, 0});
//        trainData.add(inputExtractor.get("img/d2.bmp")); dOutputs.add(new double[]{0, 0, 0, 1, 0});
//        trainData.add(inputExtractor.get("img/d3.bmp")); dOutputs.add(new double[]{0, 0, 0, 1, 0});
//        trainData.add(inputExtractor.get("img/d4.bmp")); dOutputs.add(new double[]{0, 0, 0, 1, 0});
//
//        // E
//        trainData.add(inputExtractor.get("img/e1.bmp")); dOutputs.add(new double[]{0, 0, 0, 0, 1});
//        trainData.add(inputExtractor.get("img/e2.bmp")); dOutputs.add(new double[]{0, 0, 0, 0, 1});
//        trainData.add(inputExtractor.get("img/e3.bmp")); dOutputs.add(new double[]{0, 0, 0, 0, 1});
//        trainData.add(inputExtractor.get("img/e4.bmp")); dOutputs.add(new double[]{0, 0, 0, 0, 1});
//
//        //let's learn
//        int maxEpochs = 30000;
//        perceptron.learn(trainData, dOutputs, maxEpochs);
//
//
//        System.out.println("NOW TESTING!");
//        ArrayList<double[]> testTrainData = new ArrayList<>();
//        ArrayList<double[]> testDOutputs = new ArrayList<>();
//
//        // test data
//        testTrainData.add(inputExtractor.get("img/a6.bmp")); testDOutputs.add(new double[]{1, 0, 0, 0, 0});
//        testTrainData.add(inputExtractor.get("img/b5.bmp")); testDOutputs.add(new double[]{0, 1, 0, 0, 0});
//        testTrainData.add(inputExtractor.get("img/c5.bmp")); testDOutputs.add(new double[]{0, 0, 1, 0, 0});
//        testTrainData.add(inputExtractor.get("img/d5.bmp")); testDOutputs.add(new double[]{0, 0, 0, 1, 0});
//        testTrainData.add(inputExtractor.get("img/e5.bmp")); testDOutputs.add(new double[]{0, 0, 0, 0, 1});
//        perceptron.test(testTrainData, testDOutputs);
//




        // implementing XOR function
        //2 inputs, 2 hidden layers, 1 output layer. Can store any number of hidden layers

        double lambda = 1d;

        List<Layer> layers2 = new LayerFactory(2)
                .addLayer(20, new TangensoidActivationFunction(lambda))
                .addLayer(18, new TangensoidActivationFunction(lambda))
                .addLayer(1, new TangensoidActivationFunction(lambda))
                .getLayers();

        Perceptron perceptron2 =  new Perceptron.Builder(layers2)
                .withLearningCoefficient(0.01d)
                .build();

        // learning
        int maxEpochs = 50000;
        perceptron2.learn(createTrainDataXOR(), maxEpochs);

        // testing
        double[] output = perceptron2.calculateOutput(new double[]{0,0});
        System.out.println(Arrays.toString(output));
//        perceptron2.testXor(createTrainDataXOR(), createDesiredOutputDataXOR());
    }




    public static TrainData createTrainDataXOR() {

        List<InputAndDesiredOutput> inputAndDesiredOutputList = new ArrayList<>(4);
        inputAndDesiredOutputList.add(new InputAndDesiredOutput(new double[]{0,0}, new double[]{0}));
        inputAndDesiredOutputList.add(new InputAndDesiredOutput(new double[]{0,1}, new double[]{1}));
        inputAndDesiredOutputList.add(new InputAndDesiredOutput(new double[]{1,0}, new double[]{1}));
        inputAndDesiredOutputList.add(new InputAndDesiredOutput(new double[]{1,1}, new double[]{0}));

        TrainData trainData = new TrainData(inputAndDesiredOutputList);

        return trainData;
    }


    public static ArrayList<double[]> createDesiredOutputDataXOR() {

          ArrayList<double[]> dOutputs = new ArrayList<>();


            dOutputs.add(new double[]{0});
            dOutputs.add(new double[]{1});
            dOutputs.add(new double[]{1});
            dOutputs.add(new double[]{0});

          return dOutputs;
    }

}
