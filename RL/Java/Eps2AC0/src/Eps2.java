import java.util.Random;

public class Eps2{
    // Possible states
    private double[] state;
    private Random random;
    private int cueIndex;

    // Constructor initializes the environment to the t1 state
    public Eps2() {
        random = new Random();
        // Initial state at t1: visual cue shown
        state = new double[]{0,0,0,0,1,0};
        initializeRandomCue(); // Initialize the cue
    }

    // Add this method to get the current state
    public double[] getState() {
        return state;
    }

    private void initializeRandomCue() {
        // Randomly select one of four cues
        cueIndex = random.nextInt(4);
        state[cueIndex] = 1; // Set selected cue to 1
    }

    // Method to receive an action and update the state
    public double takeAction(double nu1, double nu2, int timeStep) {
        // Ensure nu1 and nu2 sum up to 1 and are within range [0,1]
        if (nu1 + nu2 != 1.0 || nu1 < 0 || nu1 > 1 || nu2 < 0 || nu2 > 1) {
            throw new IllegalArgumentException("Invalid action values. nu1 and nu2 should sum to 1 and be within [0,1]");
        }

        if (timeStep == 1) {
            // First timestep logic (as previously implemented)
            handleFirstTimeStep(nu1, nu2);
            return 0.0; // No reward for the first time step
        } else if (timeStep == 2) {
            // Second timestep logic
            return handleSecondTimeStep(nu1, nu2);
        } else {
            throw new IllegalArgumentException("Invalid time step. Only steps 1 and 2 are supported.");
        }
    }
    private void handleFirstTimeStep(double nu1, double nu2) {
        // Ensure nu1 and nu2 sum up to 1 and are within range [0,1]
        if (nu1 + nu2 != 1.0 || nu1 < 0 || nu1 > 1 || nu2 < 0 || nu2 > 1) {
            throw new IllegalArgumentException("Invalid action values. nu1 and nu2 should sum to 1 and be within [0,1]");
        }

        // Check if the visual cue is observed based on the attention allocation and cue type
        boolean isLeftCueObserved = (state[0] == 1 || state[1] == 1) && nu1 >= 0.3;
        boolean isRightCueObserved = (state[2] == 1 || state[3] == 1) && nu2 >= 0.3;

        // Update the state only if the cue is not observed
        if (!isLeftCueObserved && !isRightCueObserved) {
            // If the cue is not observed, adjust the state
            state = new double[]{0, 0, 0, 0, 0, 1}; // Cue not observed
        }
        // Otherwise, leave the state as it is (cue observed)
    }
    private double handleSecondTimeStep(double nu1, double nu2) {
        // Logic for the second time step
        // Determine the reward based on the state, cue type, and attention allocation
        return calculateReward(nu1, nu2); // return reward
    }
    private double calculateReward(double nu1, double nu2) {
        // Reward calculation logic based on the current state and agent's action
        double rewardProbability; // Probability for the Bernoulli distribution
        if (cueIndex == 0) { // Cue left high observed
            rewardProbability = nu1; // Probability is nu1
        } else if (cueIndex == 1) { // Cue left low observed
            rewardProbability = nu2; // Probability is nu2
        } else if (cueIndex == 2) { // Cue right high observed
            rewardProbability = nu2; // Probability is nu2
        } else if (cueIndex == 3) { // Cue right low observed
            rewardProbability = nu1; // Probability is nu1
        } else {
            return 0; // No reward if no cue is observed
        }

        // Sample the reward as a Bernoulli random variable
        return random.nextDouble() < rewardProbability ? 1.0 : 0.0;
    }
}