import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Global parameters (consider passing these to AI init or making them configurable)
lmbd = 0.5
nju = 0.5
R = 10.0
N = 3 # Number of paths
batch_size = 1
seed = 42

class AI(nn.Module):
    def __init__(self, lmbd, nju, R, N):
        super().__init__()
        self.lmbd = lmbd
        self.nju = nju
        self.R = R
        self.N = N
        # The input to the linear layer is N*3 (emission, time, cost for each of N paths)
        # The output is N raw scores, one for each path.
        # We removed Softmax here because calc_probs will handle the final probability distribution.
        self.model = nn.Sequential(
            nn.Linear(self.N * 3, self.N),
            # No Softmax here. Let r be raw scores for calc_probs.
            # An activation like ReLU or Sigmoid could be added here if needed to make 'r' positive or bounded,
            # but for this specific utility function, raw scores are fine as softmax handles ranges.
        )

    def forward(self, x):
        # x is (batch_size, N*3)
        return self.model(x) # Output is (batch_size, N)

    def calc_probs(self, emissions_per_path, r_scores_from_nn):
        # emissions_per_path shape: (batch_size, N)
        # r_scores_from_nn shape: (batch_size, N)

        # Calculate utility for each path
        # Utilities are based on emission and the reward component 'r' from the NN.
        # -self.lmbd * emissions_per_path: higher emission means lower utility
        # +self.R * r_scores_from_nn * self.nju: higher 'r' from NN means higher utility, scaled by R and nju.
        utilities = -self.lmbd * emissions_per_path + self.R * r_scores_from_nn * self.nju

        # Apply softmax across the N paths for each item in the batch
        # This converts utilities into a probability distribution over the paths.
        probs = torch.softmax(utilities, dim=1) # Softmax over the N paths dimension
        return probs

    def train(self, data, epochs):
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Ensure data tensors are on the correct device if using GPU
        train_input = data["train"]#.to(device)
        emission_values = data["emission"]#.to(device)

        for epoch in range(epochs):
            optimizer.zero_grad()

            # r will be (batch_size, N) raw scores from the neural network
            r = self.forward(train_input)

            # Calculate probabilities based on emissions and the NN's output 'r'
            probs = self.calc_probs(emission_values, r)

            # The loss function: -sum(P_i * E_i) over N paths, summed over the batch
            # We want to minimize the weighted sum of emissions.
            # torch.sum(probs * emission_values, dim=1) calculates sum for each batch item
            loss_per_batch = torch.sum(probs * emission_values, dim=1)
            total_loss = loss_per_batch.sum() # Sum over all batches

            total_loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss.item():.4f}")

    def predict(self, test_data_batch):
        # test_data_batch is a list of N tuples: [(e1, t1, c1), (e2, t2, c2), (e3, t3, c3)]
        
        # Prepare emissions for calc_probs
        # Shape (1, N)
        emissions_for_calc_probs = torch.tensor(
            [[item[0] for item in test_data_batch]], dtype=torch.float32
        )

        # Prepare input for the neural network (flattened)
        # Shape (1, N*3)
        flattened_input = []
        for item in test_data_batch:
            flattened_input.extend(item)
        nn_input = torch.tensor([flattened_input], dtype=torch.float32)

        # Get raw scores from the neural network
        r_scores = self.forward(nn_input) # Shape (1, N)

        # Calculate final probabilities
        probs = self.calc_probs(emissions_for_calc_probs, r_scores)

        # Calculate the "loss" (weighted sum of emissions) for prediction
        # This will be the expected emission based on the predicted probabilities
        expected_emission_loss = torch.sum(probs * emissions_for_calc_probs, dim=1).item() # .item() to get scalar

        return probs.squeeze(0), expected_emission_loss # Squeeze to remove batch dim if batch_size=1

def transform(data, batch_size, seed):
    generator = np.random.RandomState(seed)
    # Permute the indices of the outer list (dataset)
    permutation = generator.permutation(len(data))

    all_emissions = [] # To store (batch_size, N)
    all_train_inputs = [] # To store (batch_size, N*3)

    for batch_i in range(len(data) // batch_size):
        batch_id = permutation[batch_i * batch_size:(batch_i + 1) * batch_size]
        
        current_batch_emissions = []
        current_batch_train_input = []

        for dataset_idx in batch_id:
            # Each `dataset_idx` corresponds to one `[(e,t,c), (e,t,c), (e,t,c)]` entry
            # We assume each entry represents a single "scenario" or "instance" for N paths
            
            # Flatten for NN input
            flat_path_data = []
            # Extract emissions for calc_probs
            path_emissions_only = []

            for emission, time, cost in data[dataset_idx]:
                path_emissions_only.append(emission)
                flat_path_data.extend([emission, time, cost])
            
            current_batch_emissions.append(path_emissions_only)
            current_batch_train_input.append(flat_path_data)
        
        all_emissions.extend(current_batch_emissions) # Each element is a list of N emissions for one batch item
        all_train_inputs.extend(current_batch_train_input) # Each element is a flattened list of N*3 values for one batch item

    return {
        "train": torch.tensor(all_train_inputs, dtype=torch.float32), # (num_batches, N*3)
        "emission": torch.tensor(all_emissions, dtype=torch.float32), # (num_batches, N)
    }


# Dataset definition
dataset = [
    [(2.5, 3.3, 1.6), (5.6, 4.1, 1.1), (0.5, 0.3, 0.1)], # Scenario 1
    [(6.1, 8.5, 10.6), (5.6, 4.1, 0.2), (0.9, 0.1, 0.1)], # Scenario 2
    [(4.4, 3.3, 2.2), (8.5, 4.3, 0.0), (0.1, 0.6, 100)]  # Scenario 3
]

# Transform the dataset
data = transform(dataset, batch_size, seed)
print("Transformed data shapes:")
print(f"data['train'].shape: {data['train'].shape}")    # Expected (num_batches, N*3)
print(f"data['emission'].shape: {data['emission'].shape}") # Expected (num_batches, N)

# Initialize and train the model
model = AI(lmbd, nju, R, N)
model.train(data, 100)

# Make a prediction on a new test case
test_case = [(4.0, 5.0, 1.0), (6.0, 5.0, 4.0), (3.0, 4.0, 5.0)]
predicted_probs, predicted_loss = model.predict(test_case)
print(f"\nPrediction for test case {test_case}:")
print(f"Predicted Probabilities: {predicted_probs.tolist()}")
print(f"Predicted Expected Emission (Loss): {predicted_loss:.4f}")

# Plotting the effect of nju
print("\nPlotting effect of nju...")
x = np.linspace(0.01, 1, 20) # Avoid nju=0 if it causes issues with gradients
models_for_plot = [AI(lmbd, i, R, N) for i in x]
y = []
for idx, model_for_plot in enumerate(models_for_plot):
    print(f"Training model {idx+1}/{len(models_for_plot)} for nju={x[idx]:.2f}")
    model_for_plot.train(data, 50) # Use fewer epochs for plotting speed
    # Predict for the same test case as above
    _, loss_val = model_for_plot.predict(test_case)
    y.append(loss_val)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label="Expected Emission (Loss) vs nju")
plt.title("Effect of nju on Expected Emission")
plt.xlabel("nju")
plt.ylabel("Expected Emission")
plt.grid(True)
plt.legend()
plt.savefig("plot.png")
plt.show()