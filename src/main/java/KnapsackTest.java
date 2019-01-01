import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import opt.*;
import opt.example.KnapsackEvaluationFunction;
import opt.ga.*;
import shared.FixedIterationTrainer;

import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.Random;

/**
 * A test of the knapsack problem
 * <p>
 * Given a set of items, each with a weight and a value, determine the number of each item to include in a
 * collection so that the total weight is less than or equal to a given limit and the total value is as
 * large as possible.
 * https://en.wikipedia.org/wiki/Knapsack_problem
 *
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class KnapsackTest {
    /**
     * Random number generator
     */
    private static final Random random = new Random();
    /**
     * The number of items
     */
    private static final int NUM_ITEMS = 50;
    /**
     * The number of copies each
     */
    private static final int COPIES_EACH = 4;
    /**
     * The maximum value for a single element
     */
    private static final double MAX_VALUE = 50;
    /**
     * The maximum weight for a single element
     */
    private static final double MAX_WEIGHT = 50;
    /**
     * The maximum weight for the knapsack
     */
    private static final double MAX_KNAPSACK_WEIGHT =
            MAX_WEIGHT * NUM_ITEMS * COPIES_EACH * .4;

    /**
     * The test main
     *
     * @param args ignored
     */
    public static void main(String[] args) {

        DecimalFormat decimalFormat = new DecimalFormat("0.0000");

        int[] copies = new int[NUM_ITEMS];
        Arrays.fill(copies, COPIES_EACH);
        double[] values = new double[NUM_ITEMS];
        double[] weights = new double[NUM_ITEMS];
        for (int i = 0; i < NUM_ITEMS; i++) {
            values[i] = random.nextDouble() * MAX_VALUE;
            weights[i] = random.nextDouble() * MAX_WEIGHT;
        }
        int[] ranges = new int[NUM_ITEMS];
        Arrays.fill(ranges, COPIES_EACH + 1);


        for (int iter = 1; iter < 16; iter++) {

            int num_iter = ((Double) Math.pow(2, iter)).intValue();
            EvaluationFunction ef = new KnapsackEvaluationFunction(values, weights, MAX_KNAPSACK_WEIGHT, copies);

            Distribution odd = new DiscreteUniformDistribution(ranges);
            NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);

            MutationFunction mf = new DiscreteChangeOneMutation(ranges);
            CrossoverFunction cf = new UniformCrossOver();
            Distribution df = new DiscreteDependencyTree(.1, ranges);

            HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
            GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);

            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
            FixedIterationTrainer fit = new FixedIterationTrainer(rhc, num_iter);
            Long start_rhc = System.currentTimeMillis();
            fit.train();
            Long time_rhc = System.currentTimeMillis() - start_rhc;
            Double rhc_fitness = ef.value(rhc.getOptimal());


            SimulatedAnnealing sa = new SimulatedAnnealing(1E12, 0.70, hcp);
            fit = new FixedIterationTrainer(sa, num_iter);
            Long start_sa = System.currentTimeMillis();
            fit.train();
            Long time_sa = System.currentTimeMillis() - start_sa;
            Double sa_fitness = ef.value(sa.getOptimal());


            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 20, gap);
            fit = new FixedIterationTrainer(ga, num_iter);
            Long start_ga = System.currentTimeMillis();
            fit.train();
            Long time_ga = System.currentTimeMillis() - start_ga;
            Double ga_fitness = ef.value(ga.getOptimal());

            System.out.println(num_iter + "," +
                    decimalFormat.format(rhc_fitness) + "," +

                    time_rhc + "," +
                    decimalFormat.format(sa_fitness) + "," +

                    time_sa + "," +
                    decimalFormat.format(ga_fitness) + "," +

                    time_ga + ","
            );
        }
    }

}
