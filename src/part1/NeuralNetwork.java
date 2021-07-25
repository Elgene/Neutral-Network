package part1;

import java.util.Arrays;

public class NeuralNetwork {
    public final double[][] hidden_layer_weights;
    public final double[][] output_layer_weights;
    private final int num_inputs;
    private final int num_hidden;
    private final int num_outputs;
    private final double learning_rate;private final double[][] bias;
   //public NeuralNetwork(int num_inputs, int num_hidden, int num_outputs, double[][] initial_hidden_layer_weights, double[][] initial_output_layer_weights, double learning_rate) { //without bias
    public NeuralNetwork(int num_inputs, int num_hidden, int num_outputs, double[][] initial_hidden_layer_weights, double[][] initial_output_layer_weights, double learning_rate, double[][] bias) {
        //Initialise the network
        this.num_inputs = num_inputs;
        this.num_hidden = num_hidden;
        this.num_outputs = num_outputs;

        this.hidden_layer_weights = initial_hidden_layer_weights;
        this.output_layer_weights = initial_output_layer_weights;

        this.learning_rate = learning_rate;
        this.bias = bias;
    }


    //Calculate neuron activation for an input
    public double sigmoid(double input) {
        double output = Double.NaN; //TODO!
        output= 1/(1+Math.exp(-1*input));
        return output;
    }

    //Feed forward pass input to a network output
    public double[][] forward_pass(double[] inputs) {
        double[] hidden_layer_outputs = new double[num_hidden];
        for (int i = 0; i < num_hidden; i++) {
            // TODO! Calculate the weighted sum, and then compute the final output.
            double weighted_sum = 0;
            double output = 0;
            for (int j = 0; j < num_inputs; j++) {
                weighted_sum+=inputs[j]*hidden_layer_weights[j][i];

            }
            weighted_sum+=bias[0][i];
            output=sigmoid(weighted_sum);
            hidden_layer_outputs[i] = output;
        }

        double[] output_layer_outputs = new double[num_outputs];
        for (int i = 0; i < num_outputs; i++) {
            // TODO! Calculate the weighted sum, and then compute the final output.
            double weighted_sum = 0;
            double output = 0;
            for (int j = 0; j < num_hidden; j++) {
                weighted_sum+=hidden_layer_outputs[j]*output_layer_weights[j][i];

            }
            weighted_sum+=bias[1][i];
            output=sigmoid(weighted_sum);
            output_layer_outputs[i] = output;
        }
        return new double[][]{hidden_layer_outputs, output_layer_outputs};
    }

    public double[][][] backward_propagate_error(double[] inputs, double[] hidden_layer_outputs,
                                                 double[] output_layer_outputs, int desired_outputs) {

        double[] output_layer_betas = new double[num_outputs];
        // TODO! Calculate output layer betas.
        int[] desiredArray = new int[num_outputs];
        for (int i = 0; i < num_outputs; i++) {
            desiredArray[i]=0;
        }
        desiredArray[desired_outputs]=1;
        for (int i = 0; i < num_outputs; i++) {
            output_layer_betas[i]=desiredArray[i]-output_layer_outputs[i];
        }
       // System.out.println("OL betas: " + Arrays.toString(output_layer_betas));

        double[] hidden_layer_betas = new double[num_hidden];
        // TODO! Calculate hidden layer betas.
        for (int i = 0; i < num_hidden; i++) {
            double sum = 0;
            for (int j = 0; j < num_outputs; j++) {
                sum+=output_layer_weights[i][j]*output_layer_outputs[j]*(1-output_layer_outputs[j])*output_layer_betas[j];
            }
            hidden_layer_betas[i]+=sum;
        }

        //System.out.println("HL betas: " + Arrays.toString(hidden_layer_betas));

        // This is a HxO array (H hidden nodes, O outputs)
        double[][] delta_output_layer_weights = new double[num_hidden][num_outputs];
        // TODO! Calculate output layer weight changes.
        for (int i = 0; i < num_hidden; i++) {
            for (int j = 0; j < num_outputs; j++) {
                delta_output_layer_weights[i][j]=learning_rate*hidden_layer_outputs[i]*output_layer_outputs[j]*(1-output_layer_outputs[j])*output_layer_betas[j];
            }

        }

        // This is a IxH array (I inputs, H hidden nodes)
        double[][] delta_hidden_layer_weights = new double[num_inputs][num_hidden];
        // TODO! Calculate hidden layer weight changes.
        for (int i = 0; i < num_inputs; i++) {
            for (int j = 0; j < num_hidden; j++) {
                delta_hidden_layer_weights[i][j]=learning_rate*inputs[i]*hidden_layer_outputs[j]*(1-hidden_layer_outputs[j])*hidden_layer_betas[j];
            }

        }
        double[][] delta_biases = new double[2][num_outputs];
        if(!(bias[1][1] == 0)) {
            for (int i = 0; i < num_hidden; i++) {
                delta_biases[0][i] = learning_rate*hidden_layer_outputs[i]*(1-hidden_layer_outputs[i])*hidden_layer_betas[i];
            }

            for (int i = 0; i < num_outputs; i++) {
                delta_biases[1][i] = learning_rate*output_layer_outputs[i]*(1-output_layer_outputs[i])*output_layer_betas[i];
            }
        }

        // Return the weights we calculated, so they can be used to update all the weights.
        return new double[][][]{delta_output_layer_weights, delta_hidden_layer_weights,delta_biases};
        //return new double[][][]{delta_output_layer_weights, delta_hidden_layer_weights}; //without bias

    }
    //public void update_weights(double[][] delta_output_layer_weights, double[][] delta_hidden_layer_weights) { //without bias

    public void update_weights(double[][] delta_output_layer_weights, double[][] delta_hidden_layer_weights, double[][] delta_biases) {
        // TODO! Update the weights
        for (int i = 0; i < num_hidden; i++) {
            for (int j = 0; j < num_outputs; j++) {
                output_layer_weights[i][j] += delta_output_layer_weights[i][j];
            }
        }

        for (int i = 0; i < num_inputs; i++) {
            for (int j = 0; j < num_hidden; j++) {
                hidden_layer_weights[i][j] += delta_hidden_layer_weights[i][j];
            }
        }

        for (int i = 0; i < num_hidden; i++) {
            bias[0][i] += delta_biases[0][i];
        }

        for (int i = 0; i < num_outputs; i++) {
            bias[1][i] += delta_biases[1][i];
        }
       // System.out.println("Placeholder");
    }

    public void train(double[][] instances, int[] desired_outputs, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
           System.out.println("epoch = " + epoch);
            int[] predictions = new int[instances.length];
            for (int i = 0; i < instances.length; i++) {
                double[] instance = instances[i];
                double[][] outputs = forward_pass(instance);
                double[][][] delta_weights = backward_propagate_error(instance, outputs[0], outputs[1], desired_outputs[i]);
                int predicted_class = -1; // TODO!

                double maxVal = Double.MIN_VALUE;
                for (int j = 0; j < num_outputs; j++) {
                    if(outputs[1][j] > maxVal) {
                        maxVal = outputs[1][j];
                        predicted_class = j;
                    }
                }
                predictions[i] = predicted_class;

                //We use online learning, i.e. update the weights after every instance.
                //update_weights(delta_weights[0], delta_weights[1]); without bias
                update_weights(delta_weights[0], delta_weights[1],delta_weights[2]);

            }

            // Print new weights
            System.out.println("Hidden layer weights \n" + Arrays.deepToString(hidden_layer_weights));
            System.out.println("Output layer weights  \n" + Arrays.deepToString(output_layer_weights));

            // TODO: Print accuracy achieved over this epoch
            double acc = 0;
            for (int i = 0; i < predictions.length; i++) {
                if(predictions[i] == desired_outputs[i]) { acc++; }
            }
            acc = acc/(double)predictions.length;
            System.out.printf("acc = %.2f%%\n", acc*100);
        }
    }

    public int[] predict(double[][] instances) {
        int[] predictions = new int[instances.length];
        for (int i = 0; i < instances.length; i++) {
            double[] instance = instances[i];
            double[][] outputs = forward_pass(instance);
            if (i == 0) {
                System.out.println("Report the first instance output: " + Arrays.toString(outputs[1]));

            }
            int predicted_class = -1;  // TODO !Should be 0, 1, or 2.
            double maxVal = Double.MIN_VALUE;
            for (int j = 0; j < num_outputs; j++) {
                if(outputs[1][j] > maxVal) {
                    maxVal = outputs[1][j];
                    predicted_class = j;
                }
            }
            predictions[i] = predicted_class;
        }
        return predictions;
    }

}
