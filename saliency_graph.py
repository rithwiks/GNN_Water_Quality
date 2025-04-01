import torch
import matplotlib.pyplot as plt
import numpy as np
from baseline import Net, test_dataloader, diatoms
import sys
sys.modules['__main__'].Net = Net

# construct map_onehot_to_taxon without diplicate
df_unique = diatoms[['TaxonName','onehot']].drop_duplicates(subset='onehot')
map_onehot_to_taxon = dict(zip(df_unique['onehot'], df_unique['TaxonName']))

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 2292
model = Net(input_dim, 2, 4096, 1024, 256)
model = torch.load('diatom_model_complete.pth', map_location=device)
model.to(device)
model.eval()

saliency_list = []  # we will put abs gradient in it

for batch_x, batch_y, batch_keys in test_dataloader: # iter on dataloader's batches
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.to(device)
    
    # grad on entry
    batch_x.requires_grad = True

    # forward
    logits = model(batch_x) # score before sofmax
    predicted_class = logits.argmax(dim=1)

    # backward only for predicted class
    # one-hot
    one_hot = torch.zeros_like(logits)
    for i, c in enumerate(predicted_class):
        # for each sample i, we put 1 in the corresponding column 
        one_hot[i, c] = 1.0
    
    # reset grad, then backward
    model.zero_grad()
    if batch_x.grad is not None:
        batch_x.grad.zero_()

    # logits * one_hot so we keep only the logts of the classes we want to backward
    # .sum() for it to be a scalar on the batch
    scalar_value = (logits * one_hot).sum()
    scalar_value.backward()

    # get grad of predicted class from input x
    grad_input = batch_x.grad.detach().cpu().numpy()

    # get abs grad
    saliency_list.append(np.abs(grad_input))

# concatenate
all_saliency = np.concatenate(saliency_list, axis=0)

# so each element reprensents the abs mean grad for a feature
saliency_per_feature = np.mean(all_saliency, axis=0)

# top-K
K = 20
topk_indices = np.argsort(saliency_per_feature)[::-1][:K]
topk_values  = saliency_per_feature[topk_indices]

# get topk_taxonnames from topk_indices
topk_taxonnames = []
for idx in topk_indices:
    if idx in map_onehot_to_taxon:
        topk_taxonnames.append(map_onehot_to_taxon[idx])
    else:
        topk_taxonnames.append(f"Unknown_{idx}")

# plot top 20 saliency
plt.figure(figsize=(10,5))
plt.bar(range(K), topk_values)
plt.xticks(range(K), topk_taxonnames, rotation=45)
plt.title("Top-20 most important Diatoms (saliency abs mean)")
plt.xlabel("Diatoms")
plt.ylabel("grad abs mean")
plt.tight_layout()
plt.savefig("saliency_abs_mean_top_20")
plt.close()

# plot all saliency
sorted_saliency = np.sort(saliency_per_feature)[::-1]

plt.figure(figsize=(12,5))
plt.plot(sorted_saliency)
plt.title("All saliency values sorted (abs mean gradient per Diatom)")
plt.xlabel("Diatoms")
plt.ylabel("grad abs mean")
plt.tight_layout()
plt.savefig("saliency_abs_mean_all_sorted")
plt.close()

# bar plot all saliency
plt.figure(figsize=(12,5))
plt.bar(range(len(sorted_saliency)), sorted_saliency)
plt.plot(sorted_saliency)
plt.title("Bar plot all saliency values sorted (abs mean gradient per Diatom)")
plt.xlabel("Diatoms")
plt.ylabel("grad abs mean")
plt.tight_layout()
plt.savefig("bar_plot_saliency_abs_mean_all_sorted")
plt.close()