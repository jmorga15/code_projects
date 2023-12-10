// If your Environment class is in a package, include the package line here.
// For example:
// package com.yourdomain.yourproject;

public class Eps2Test {
    public static void main(String[] args) {
        for (int trial = 1; trial <= 3; trial++) {
            // Initialize environment
            Eps2 env = new Eps2();

            // Randomly generate attention allocations for two time steps
            double nu1_t1 = Math.random();
            double nu2_t1 = 1.0 - nu1_t1;

            double nu1_t2 = Math.random();
            double nu2_t2 = 1.0 - nu1_t2;

            System.out.println("Trial " + trial + ":");
            System.out.println("Initial State: " + arrayToString(env.getState()));
            System.out.println("Time Step 1 - Attention Allocation: nu1 = " + nu1_t1 + ", nu2 = " + nu2_t1);

            // Take action in time step 1
            env.takeAction(nu1_t1, nu2_t1, 1);
            System.out.println("State after Time Step 1: " + arrayToString(env.getState()));

            System.out.println("Time Step 2 - Attention Allocation: nu1 = " + nu1_t2 + ", nu2 = " + nu2_t2);

            // Take action in time step 2 and calculate the reward
            // In Eps2Test class
            double reward = env.takeAction(nu1_t2, nu2_t2, 2);
            System.out.println("Reward: " + reward);
            System.out.println("State after Time Step 2: " + arrayToString(env.getState()));

            System.out.println("------------------------------------------------------");
        }
    }

    // Helper method to convert an array to a string for easy printing
    private static String arrayToString(double[] array) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < array.length; i++) {
            sb.append(array[i]);
            if (i < array.length - 1) {
                sb.append(", ");
            }
        }
        sb.append("]");
        return sb.toString();
    }
}
