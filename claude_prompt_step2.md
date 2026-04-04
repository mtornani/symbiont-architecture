# Step 2: Scaling SAM to a Micro-Network (Symbiont Cluster)

Hello Claude,

Based on the successful validation of the single `SamTernaryNeuron` (Rule-Relocation Problem solved), we are now pivoting to **Step 2**. Benchmarking a single ternary neuron against massive LLMs like OpenVLA is technically incoherent. Instead, our "technical revolution" must be demonstrated through **behavioral adaptivity**, specifically by moving from a single cell to a network.

Your goal for this step is to implement **Phase 1: Scaling to a Micro-Network**. We need to build a "Symbiont Cluster" to demonstrate **Distributed Inhibition**—where a risk encountered by one node protects the entire system.

## Technical Specifications

Please implement the following architecture in a new script (e.g., `sam_cluster.py`):

1. **The Network (Symbiont Cluster)**:
   - Instantiate a micro-network of **3 to 5 Ternary Neurons**.
   - These neurons should have slightly varied initial weights to simulate diverse processing nodes (e.g., some more risk-averse, some more exploratory).

2. **The Shared Endocrine System**:
   - Create a central `EndocrineSystem` class/object outside the neurons.
   - This shared environment must broadcast global `cortisol` (stress) and `dopamine` (reward) levels to all neurons simultaneously.
   - Neurons should read from and potentially contribute to these global endocrine levels.

3. **Emergent Goal: Distributed Inhibition**:
   - We need to demonstrate that if *one* neuron encounters a high-risk / low-reward situation (spiking the global cortisol level), the resulting endocrine spike **inhibits the entire cluster**.
   - This collective homeostatic response should prevent a system-wide cascade failure without needing top-down, hardcoded if/then rules for each neuron.

4. **New Endocrine Variable**:
   - Add an **`oxytocin`** variable to the `EndocrineSystem` as a placeholder.
   - For now, document that this will act as a social/cooperative signal for future multi-agent iterations. You do not need to implement complex oxytocin logic right now, just incorporate it into the shared state architecture.

## Expected Output

- A unified Python script demonstrating the operation of the Symbiont Cluster.
- A concise test run or simulation function within the script mapping out an environmental shock (high-risk event hitting one neuron) and printing the response of the whole cluster (showing distributed inhibition).
- *Optional but recommended*: A plot showing the individual neuron states alongside the shared endocrine levels during the shock.

Please proceed with writing the implementation and the internal test. Let me know if you need any clarification on the architecture!
