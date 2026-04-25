# Planned LoRA Figures

## Four Main Figures

### 1. Performance vs. Method

Use a bar chart comparing final validation or test performance across methods such as full fine-tuning, linear probing, adapters, prefix tuning, and LoRA with ranks 1, 4, and 8. This is the headline plot for answering whether LoRA can match full fine-tuning on task quality. It should be the most prominent poster figure because it gives the simplest performance comparison.

### 2. Performance vs. Trainable Parameters

Use a scatter plot with trainable parameters on the x-axis, ideally log scale, and final validation or test performance on the y-axis. This plot should show LoRA points close to full fine-tuning in performance while using far fewer trainable parameters. It directly communicates the main efficiency-quality tradeoff in the LoRA paper.

### 3. Performance vs. LoRA Rank

Use a line plot with LoRA rank `r` on the x-axis and validation or test performance on the y-axis. This rank ablation tests how much low-rank capacity is needed before performance saturates. It connects directly to the paper's claim that useful task updates often live in a low-rank subspace.

### 4. Memory or Storage Savings

Use a bar chart comparing practical resource costs across frozen inference, full fine-tuning, and LoRA ranks such as 4, 8, and 16. The y-axis can be peak GPU memory or stored model size per downstream task, depending on what we can measure most reliably. This figure makes the practical benefit of LoRA visible beyond accuracy alone.

## Lower-Priority Figures

<small>

### 5. Training and Validation Curves

Use line plots over training steps or epochs for full fine-tuning and several LoRA ranks. These curves can show whether LoRA learns as quickly, underfits at low rank, or behaves more stably than full fine-tuning.

### 6. Target-Module Ablation

Use a grouped bar chart comparing LoRA applied to modules such as `W_q`, `W_v`, `W_q + W_v`, all attention projections, and attention plus MLP. This helps test whether LoRA works best because of low rank alone or because it is inserted into the most useful matrices.

### 7. Storage Cost Per Task

Use a bar chart comparing how much data must be stored for each downstream task under full fine-tuning versus LoRA. This is useful because full fine-tuning stores a whole model copy, while LoRA stores only small task-specific adapter matrices.

### 8. Inference Latency Before and After Merging

Use a bar chart comparing base model latency, unmerged LoRA latency, and merged LoRA latency. This figure supports the claim that LoRA can avoid extra inference latency once the low-rank update is merged into the base weights.

### 9. Singular Value or Effective Rank Analysis

Use a line plot of normalized singular values from the learned LoRA update `Delta W = BA`, or from the difference between full fine-tuned and original weights if available. This is a more advanced figure that supports the low intrinsic-rank motivation behind LoRA.

</small>
