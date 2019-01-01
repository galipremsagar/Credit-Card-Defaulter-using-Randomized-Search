import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import opt.OptimizationAlgorithm;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import shared.DataSet;
import shared.ErrorMeasure;
import shared.Instance;
import shared.SumOfSquaresError;
import util.linalg.Vector;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.text.DecimalFormat;
import java.util.Scanner;

public class CreditSimulatedAnnealing {

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, Integer trainingIterations, Instance[] instances, ErrorMeasure measure) {

        for (int i = 0; i < trainingIterations; i++) {
            oa.train();

            double error = 0;
            for (int j = 0; j < instances.length; j++) {
                network.setInputValues(instances[j].getData());
                network.run();

                Instance output = instances[j].getLabel(), example = new Instance(network
                        .getOutputValues());
                example.setLabel(new Instance(network.getOutputValues()));
                error += measure.value(output, example);
            }
//            System.out.println(df.format(error));
        }
    }

    public static Instance[] initializeInstances(String fileName, Integer numInstances) {

        double[][][] attributes = new double[numInstances][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File(fileName)));
            for (int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[23];
                attributes[i][1] = new double[2];

                for (int j = 0; j < 23; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());

                for (int j = 0; j < 2; j++) {
                    attributes[i][1][j] = Double.parseDouble(scan.next());

                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for (int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            instances[i].setLabel(new Instance(attributes[i][1]));
        }

        return instances;
    }

    public static void runNeuralNetwork(OptimizationAlgorithm oa, Integer inputLayer, Integer hiddenLayer, Integer outputLayer, DataSet trainSet, ErrorMeasure measure, Instance[] trainInstances, Instance[] testInstances, Integer numIters, Integer skips) {
        BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
        BackPropagationNetwork network = factory.createClassificationNetwork(
                new int[]{inputLayer, hiddenLayer, outputLayer});

        for (int iter = 1; iter <= numIters; iter++) {
            Integer num_iter = iter * skips;
            double start_testing, end_testing, trainingTime, testingTime, correct_test = 0, wrong_test = 0;
            train(oa, network, num_iter, trainInstances, measure);

            Instance optimalInstance = oa.getOptimal();
            network.setWeights(optimalInstance.getData());


            double start_training = System.nanoTime();
            double correct_train = 0, wrong_train = 0;
            for (int j = 0; j < trainInstances.length; j++) {
                network.setInputValues(trainInstances[j].getData());
                network.run();

                Vector actual_instance = trainInstances[j].getLabel().getData();
                Vector predicted_instance = network.getOutputValues();
                int temp_index = predicted_instance.argMax();

                if (actual_instance.get(temp_index) == 1.0) {
                    correct_train++;
                } else {
                    wrong_train++;
                }
            }
            double end_training = System.nanoTime();
            trainingTime = end_training - start_training;
            trainingTime /= Math.pow(10, 9);


            start_testing = System.nanoTime();
            for (int j = 0; j < testInstances.length; j++) {
                network.setInputValues(testInstances[j].getData());
                network.run();

                Vector actual_instance = testInstances[j].getLabel().getData();
                Vector predicted_instance = network.getOutputValues();
                int temp_index = predicted_instance.argMax();

                if (actual_instance.get(temp_index) == 1.0) {
                    correct_test++;
                } else {
                    wrong_test++;
                }
            }
            end_testing = System.nanoTime();
            testingTime = end_testing - start_testing;
            testingTime /= Math.pow(10, 9);

            System.out.println(num_iter + "," +
                    correct_test + "," +
                    wrong_test + "," +
                    df.format(correct_test / (correct_test + wrong_test) * 100) + "," +
                    correct_train + "," +
                    wrong_train + "," +
                    df.format(correct_train / (correct_train + wrong_train) * 100) + "," +
                    df.format(trainingTime) + "," +
                    df.format(testingTime));
        }
    }


    public static void main(String[] args) {

        Instance[] trainInstances = initializeInstances("data/train_credit.csv", 1400);
        Instance[] testInstances = initializeInstances("data/test_credit.csv", 600);

        BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
        BackPropagationNetwork network = factory.createClassificationNetwork(
                new int[]{23, 19, 2});
        DataSet set = new DataSet(trainInstances);
        ErrorMeasure measure = new SumOfSquaresError();

        NeuralNetworkOptimizationProblem nnop = new NeuralNetworkOptimizationProblem(set, network, measure);
        OptimizationAlgorithm oa = new SimulatedAnnealing(1E12, 0.75, nnop);
        runNeuralNetwork(oa, 23, 19, 2, set, measure, trainInstances, testInstances, 30, 1000);
    }
}