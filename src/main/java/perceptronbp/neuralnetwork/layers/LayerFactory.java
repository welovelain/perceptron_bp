package perceptronbp.neuralnetwork.layers;

import perceptronbp.neuralnetwork.activationfunctions.ActivationFunction;

import java.util.ArrayList;
import java.util.List;

public class LayerFactory {
    private int n_inputs;
    private List<Layer> layers = new ArrayList<>();

    public LayerFactory(int n_inputs) {
        if (n_inputs <= 0) {
            throw new IllegalArgumentException("Number of inputs should be > 0");
        }
        this.n_inputs = n_inputs;
    }

    public LayerFactory addLayer(int n_neurons) {

        Layer previous = null;
        if (!layers.isEmpty()) {
            previous = layers.get(layers.size() - 1);
        }

        Layer newLayer;
        if (previous == null) {
            newLayer = new Layer(0, n_neurons, n_inputs + 1); // add 1 for bias.
        } else {
            newLayer = new Layer(layers.size(), n_neurons, previous.getAmountOfNeurons() + 1); // add 1 for bias.
        }
        layers.add(newLayer);

        return this;
    }

    public LayerFactory addLayer(int n_neurons, ActivationFunction activationFunction) {
        Layer previous = null;
        if (!layers.isEmpty()) {
            previous = layers.get(layers.size() - 1);
        }

        Layer newLayer;
        if (previous == null) {
            newLayer = new Layer(0, n_neurons, n_inputs + 1);
        } else {
            newLayer = new Layer(layers.size(), n_neurons, previous.getAmountOfNeurons() + 1); // add 1 for bias.
        }
        newLayer.setActivationFunction(activationFunction);
        layers.add(newLayer);

        return this;
    }

    public List<Layer> getLayers() {
        return layers;
    }

}
