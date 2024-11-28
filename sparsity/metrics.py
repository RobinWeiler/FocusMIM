
# To calculate sparsity metrics, add at the respective places in the pretraining script.

# 1. At the beginning of each training epoch
sparsity_unit_total = 0
sparsity_population_total = 0

# 2. After inference on a minibatch. Metrics accumulate across minibatches, and are thus later (before plotting) divided by the number of minibatches.
sparsity_unit_batch = 1 - (torch.mean(reps, dim=0) ** 2 + 1e-7) / (torch.mean(reps ** 2, dim=0) + 1e-7)
sparsity_unit_total += torch.mean(sparsity_unit_batch).item()

sparsity_population_batch = 1 - (torch.mean(reps_flat, dim=1) ** 2 + 1e-7) / (torch.mean(reps_flat ** 2, dim=1) + 1e-7)
sparsity_population_total += torch.mean(sparsity_population_batch).item()

# 3. After each training epoch
writer.add_scalar('Loss/train/sparsity_unit', sparsity_unit_total, epoch)
writer.add_scalar('Loss/train/sparsity_population', sparsity_population_total, epoch)