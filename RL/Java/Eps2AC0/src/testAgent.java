import java.util.Random;
import java.util.Arrays;


public class testAgent {
    public static void main(String[] args) {
        // Initialize the ActorCriticAgent
        int stateSize = 6; // Size of the state vector
        int numStates = 6; // Number of distinct states
        double learningRate = 0.01;
        agent agentAttention = new agent(stateSize, numStates, learningRate);

        // Define a simple environment
        double[][] states = {
                {0, 0, 0, 0, 1, 0},
                {0, 0, 0, 0, 0, 1},
                {1, 0, 0, 0, 0, 1},
                {0, 1, 0, 0, 0, 1},
                {0, 0, 1, 0, 0, 1},
                {0, 0, 0, 1, 0, 1}
        };

        // Simulate a sequence of actions and state transitions
        for (int i = 0; i < 100; i++) {
            int stateIndex = i % states.length;
            double[] currentState = states[stateIndex];
            double[] nextState = states[(stateIndex + 1) % states.length];

            // Agent chooses an action based on current state
            double[] action = agentAttention.chooseAction(currentState);

            // Assume a simple reward function (for testing)
            double reward = calculateReward(currentState, action);

            // Agent learns from the experience
            boolean done = (stateIndex == states.length - 1);
            agentAttention.learn(currentState, action, reward, nextState, done);

            // Optionally, print out results for inspection
            System.out.println("Step " + i + ", State: " + Arrays.toString(currentState) +
                    ", Action: " + Arrays.toString(action) +
                    ", Reward: " + reward);
        }
    }

    // Placeholder for a reward calculation function
    private static double calculateReward(double[] state, double[] action) {
        // Define how the reward is calculated based on the state and action
        // This is just a placeholder for a proper reward function
        return Math.random(); // Random reward for testing
    }
}
