import java.util.Random;
import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        // Initialize the ActorCriticAgent
        int stateSize = 6; // Size of the state vector
        int numStates = 6; // Number of distinct states
        double learningRate = 0.01;
        agent agentAttention = new agent(stateSize, numStates, learningRate);

        // For tracking rewards
        double totalReward;
        double meanReward;

        for (int trial = 1; trial <= 50000; trial++) {
            totalReward = 0.0; // Reset total reward for this tria

            // Initialize environment
            Eps2 env = new Eps2();

            System.out.println("Trial " + trial + ":");
            System.out.println("Initial State: " +  Arrays.toString(env.getState()));

            double[] currentState = Arrays.copyOf(env.getState(), env.getState().length); // Create an independent copy of the current state

            // Time Step 1
            // First attention action to observe the cue
            double[] action_t1 = agentAttention.chooseAction(env.getState());
            double reward_t1 = env.takeAction(action_t1[0],  action_t1[1], 1);
//            env.takeAction(action_t1[0], action_t1[1], 1);
            System.out.println("Time Step 1 - Attention Allocation: nu1 = " + action_t1[0] + ", nu2 = " + action_t1[1]);
            System.out.println("State after Time Step 1: " + Arrays.toString(env.getState()));

            double[] nextState = Arrays.copyOf(env.getState(), env.getState().length); // Get the next state after the action

            System.out.println("Current State T1: " +  Arrays.toString(currentState));

            // Calculate reward and update agent
//            double reward_t1 = env.calculateReward(action_t1[0], action_t1[1]);
            agentAttention.learn(currentState, action_t1, reward_t1, nextState, false); // Assuming non-terminal state

            // Time Step 2
            double[] currentStateT2 = env.getState(); // Get the current state for time step 2

            // Agent applies attention to spot cue
            double[] action_t2 = agentAttention.chooseAction(env.getState());
            double reward_t2 = env.takeAction(action_t2[0], action_t2[1], 2);
            System.out.println("Time Step 2 - Attention Allocation: nu1 = " + action_t2[0] + ", nu2 = " + action_t2[1]);
            System.out.println("Reward: " + reward_t2);
            System.out.println("State after Time Step 2: " + Arrays.toString(env.getState()));
            double[] nextStateT2 = env.getState(); // Get the next state after the action

            // Update agent for the second time step
            agentAttention.learn(currentStateT2, action_t2, reward_t2, nextStateT2, true); // Assuming terminal state

            // Accumulate total rewards
            totalReward += reward_t1 + reward_t2;

            // Calculate and print mean reward for this trial
            meanReward = totalReward / 2; // Assuming 2 time steps per trial
            System.out.println("Mean Reward for Trial " + trial + ": " + meanReward);

            System.out.println("------------------------------------------------------");
        }

        // Print the Q-Table (value function estimates) after all trials
        System.out.println("Q-Table (Value Function Estimates) after all trials:");
        for (int i = 0; i < numStates; i++) {
            System.out.println("State " + i + ": " + agentAttention.getStateValue(i));
        }

        // After all trials, print the actor weights
        agentAttention.printWeights();

    }

    // ... arrayToString method ...
}
