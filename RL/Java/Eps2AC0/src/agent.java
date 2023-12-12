import java.util.Random;
import java.util.Arrays;

public class agent {
    private double[][] actorAlphaWeights; // weights for the alpha parameter
    private double[][] actorBetaWeights; // weights for the beta parameter
    private double[] stateValues; // Value Funcction estimation
    private Random random; // Random number generator
    private double learningRate; // learning rate for updates
    private int stateSize; // size of state vector
    private final int output=1; // Output size, 1 because we only need to sample one action
                                // and get the other by subtraction

    // Constructor
    public agent(int stateSize, int numStates, double learningRate) {
        this.stateSize = stateSize;
        this.learningRate = learningRate;
        this.actorAlphaWeights = new double[stateSize][output];
        this.actorBetaWeights = new double[stateSize][output];
        this.stateValues = new double[numStates];
        this.random = new Random();

        initializeWeights();
    }

    // Initialize weights for alpha and beta to small random values
    private void initializeWeights() {
        for (int i = 0; i< stateSize; i++) {
            for (int j = 0; j < output; j++) {
                actorAlphaWeights[i][j] = random.nextDouble() * 1;
                actorBetaWeights[i][j] = random.nextDouble() * 1;
            }
        }
    }

    // Method to choose an action based on the current state
    public double[] chooseAction(double[] state) {
        // Compute alpha and beta for the Beta distribution
        double alpha = positiveTransformation(matrixMultiply(state, actorAlphaWeights)[0]);
        double beta = positiveTransformation(matrixMultiply(state, actorBetaWeights)[0]);

        // Sample one value from beta distribution
        double actionProbability = sampleFromBeta(alpha,beta);

        // Return action probabilities (action, 1-action)
        return new double[]{actionProbability, 1 - actionProbability};
    }

    // Ensure the transformation of weights output to positive values (alpha and beta > 0)
    private double positiveTransformation(double value) {
        // Applying exponential function to ensure positivity
        return Math.exp(value/5.0);
    }

    // Matrix multiplication to get raw output from the actor network
    private double[] matrixMultiply(double[] state, double[][] weights) {
        double[] result = new double[output];
        for (int i = 0; i < stateSize; i++){
            for (int j = 0; j < output; j++) {
                result[j] += state[i] * weights[i][j];
            }
        }
        return result;
    }
    // Sample from Beta distribution given alpha and beta
    // Sample from Beta distribution given alpha and beta
    private double sampleFromBeta(double alpha, double beta) {
        // Generate two gamma-distributed values
        double gammaSampleForAlpha = sampleFromGamma(alpha);
        double gammaSampleForBeta = sampleFromGamma(beta);

        // Beta sample is gammaSampleForAlpha divided by the sum of both samples
        return gammaSampleForAlpha / (gammaSampleForAlpha + gammaSampleForBeta);
    }

    // Generate a sample from a Gamma distribution (simplified approximation)
    private double sampleFromGamma(double shape) {
        // Using Java's Random class to generate a normally distributed value
        double standardNormal = random.nextGaussian();
        // Transforming the standard normal value
        return Math.pow(standardNormal, 2) * shape;
    }

    // Method to update actor and critic based on the experience
    public void learn(double[] state, double[] action, double reward,double[] nextState, boolean done) {
        // Calculate td error for critic
        int stateIndex = convertStateToIndex(state);
        double valueNextState = done ? 0 : stateValues[convertStateToIndex(nextState)];
        double tdError = reward + valueNextState - stateValues[stateIndex];
        stateValues[stateIndex] += learningRate * tdError;
        System.out.println("Learning: Current State Index = " + stateIndex + ", Next State Index = " + convertStateToIndex(nextState));


        // Update Actor
        // Compute alpha and beta for the current state
        double alpha = positiveTransformation(matrixMultiply(state, actorAlphaWeights)[0]);
        double beta = positiveTransformation(matrixMultiply(state, actorBetaWeights)[0]);
        double sampledAction = action[0]; // Assume action[0] is the sampled action probability

        // Compute gradients for alpha and beta weights
        for (int i = 0; i < stateSize; i++) {
            final double EPSILON = 1e-1; // Small constant for numerical stability
            double gradAlpha = (Math.log(sampledAction + EPSILON) - digamma(alpha) + digamma(alpha + beta)) * state[i];
            double gradBeta = (Math.log(1 - sampledAction + EPSILON) - digamma(beta) + digamma(alpha + beta)) * state[i];

            actorAlphaWeights[i][0] += learningRate * tdError * gradAlpha;
            actorBetaWeights[i][0] += learningRate * tdError * gradBeta;
        }

    }

    // Numerically approximate the digamma function
    private double digamma(double x) {
        double delta = 1e-1;  // A small number for finite difference
        return (Math.log(gamma(x + delta)) - Math.log(gamma(x))) / delta;
    }

    // Gamma function approximation (could use Stirling's approximation or Lanczos approximation)
    private double gamma(double x) {
        // Stirling's approximation for gamma function
        double stirling = Math.sqrt(2 * Math.PI / x) * Math.pow(x / Math.E, x);
        return stirling; // This is a rough approximation, especially for small x
    }

    // Method to convert a state array to an index based on specific criteria
    private int convertStateToIndex(double[] state) {
        // Validate the state length
        if (state.length != 6) {
            throw new IllegalArgumentException("State array must have a length of 6.");
        }
        int index = -1; // Default to an invalid index
        // Check for each specific state and return the corresponding index
        // Check for each specific state and set the corresponding index
        if (state[0] == 0 && state[1] == 0 && state[2] == 0 && state[3] == 0 &&
                state[4] == 1 && state[5] == 0)  index = 0; // State (0,0,0,0,1,0)
        if (state[0] == 0 && state[1] == 0 && state[2] == 0 && state[3] == 0 &&
                state[4] == 0 && state[5] == 1) index = 1; // State (0,0,0,0,0,1)
        if (state[0] == 1 && state[5] == 1) index = 2; // State (1,0,0,0,0,1)
        if (state[1] == 1 && state[5] == 1) index = 3; // State (0,1,0,0,0,1)
        if (state[2] == 1 && state[5] == 1) index = 4; // State (0,0,1,0,0,1)
        if (state[3] == 1 && state[5] == 1) index = 5; // State (0,0,0,1,0,1)

        // Debugging: Print the state and its corresponding index
        System.out.println("State: " + Arrays.toString(state) + " -> Index: " + index);

        // Check if a valid index was found
        if (index == -1) {
            throw new IllegalArgumentException("Invalid state array: " + Arrays.toString(state));
        }

        return index;

    }

    // Helper method to compare doubles with a tolerance
    private boolean isApproximatelyEqual(double a, double b) {
        final double TOLERANCE = 1E-6;
        return Math.abs(a - b) < TOLERANCE;
    }

    // Method to get the value function estimate for a given state index
    public double getStateValue(int stateIndex) {
        if (stateIndex < 0 || stateIndex >= stateValues.length) {
            throw new IllegalArgumentException("Invalid state index.");
        }
        return stateValues[stateIndex];
    }

    // Method to print the actor weights
    public void printWeights() {
        System.out.println("Actor Alpha Weights:");
        printWeightMatrix(actorAlphaWeights);
        System.out.println("Actor Beta Weights:");
        printWeightMatrix(actorBetaWeights);
    }

    private void printWeightMatrix(double[][] weights) {
        for (double[] row : weights) {
            for (double weight : row) {
                System.out.print(weight + " ");
            }
            System.out.println();
        }
    }

}