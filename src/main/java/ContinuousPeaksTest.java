
import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import opt.*;
import opt.example.ContinuousPeaksEvaluationFunction;
import opt.ga.*;
import shared.FixedIterationTrainer;

import java.text.DecimalFormat;
import java.util.Arrays;

/**
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class ContinuousPeaksTest {
    /**
     * The n value
     */
    private static final int N = 60;
    /**
     * The t value
     */
    private static final int T = N / 10;

    public static void main(String[] args) {
        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);

        DecimalFormat decimalFormat = new DecimalFormat("0.0000");

        for (int iter = 1; iter < 15; iter++) {

            int num_iter = ((Double) Math.pow(2, iter)).intValue();

            EvaluationFunction ef = new ContinuousPeaksEvaluationFunction(T);
            Distribution odd = new DiscreteUniformDistribution(ranges);
            NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
            MutationFunction mf = new DiscreteChangeOneMutation(ranges);
            CrossoverFunction cf = new SingleCrossOver();
            Distribution df = new DiscreteDependencyTree(.1, ranges);
            HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
            GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);


            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
            FixedIterationTrainer fit = new FixedIterationTrainer(rhc, num_iter);
            Long start_rhc = System.currentTimeMillis();
            fit.train();
            Long rhc_time = System.currentTimeMillis() - start_rhc;
            Double rhc_fitness = ef.value(rhc.getOptimal());


            SimulatedAnnealing sa = new SimulatedAnnealing(1E12, 0.95, hcp);
            fit = new FixedIterationTrainer(sa, num_iter);
            Long start_sa = System.currentTimeMillis();
            fit.train();
            Long sa_time = System.currentTimeMillis() - start_sa;
            Double sa_fitness = ef.value(sa.getOptimal());


            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 100, gap);
            fit = new FixedIterationTrainer(ga, num_iter);
            Long start_ga = System.currentTimeMillis();
            fit.train();
            Long ga_time = System.currentTimeMillis() - start_ga;
            Double ga_fitness = ef.value(ga.getOptimal());


            System.out.println(num_iter + "," +
                    decimalFormat.format(rhc_fitness) + "," +
                    rhc_time + "," +
                    decimalFormat.format(sa_fitness) + "," +
                    sa_time + "," +
                    decimalFormat.format(ga_fitness) + "," +
                    ga_time
            );
        }
    }
}
