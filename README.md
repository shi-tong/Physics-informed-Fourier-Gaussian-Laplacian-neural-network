# Data Description for Paper:

**"Physics-informed Fourier-Gaussian-Laplacian neural network for temperature field reconstruction and long-term prediction in laser wire additive manufacturing"**

This repository contains the code and data for the submitted paper titled *"Physics-informed Fourier-Gaussian-Laplacian neural network for temperature field reconstruction and long-term prediction in laser wire additive manufacturing"*.

## FGL-PINN Model Training and Testing Process

### Step-by-Step Guide:

1. **Download Source Data**

   * Obtain the required source data files (`E1-K3`).

2. **Configure Input Parameters**

   * Specify the process parameters and material properties.

   * Provide these inputs in the `if __name__ == '__main__':` section of `FGL-PINN.py`, specifically within the argument parsing block.

3. **Set Training Data Path**

   * In the `data` directory, set the file path to the corresponding `X_train.npy` file.

4. **Set Validation Data Path**

   * In the `valid` directory, set the file path to the corresponding `X_valid.npy` file.

5. **Hyperparameter Optimization**

   * Initiate Bayesian optimization by executing:

     ```python
     res = gp_minimize(objective, space, n_calls=..., random_state=42)
     ```

   * This step identifies the optimal hyperparameters.

6. **Adjust Training Epochs**

   * Modify the total number of training epochs via the `iterations = ...` parameter.

7. **Save Model Outputs**

   * Upon completion, save:

     * The trained model weights.

     * All recorded loss metrics for subsequent evaluation and deployment.

