# DSA 8401 Applied Machine Learning — Practice Paper 2
## Unsupervised Learning, Deep Learning, Modern Generative AI
**50 questions. Five options each (A–E). One correct answer per question. Answer key at the end.**

---

**1.** Unsupervised learning is defined as returning a "structural summary" of an unknown distribution p from an i.i.d. sample. Which statement about validating such a summary is correct per the course notes?

A. There is always a held-out loss that uniquely determines the correct number of clusters, exactly as in supervised learning
B. Validation must come from internal geometric indices, likelihood-based complexity penalties, and downstream utility, since there is no universally correct answer
C. Silhouette score alone is sufficient and no other diagnostic is ever needed
D. Unsupervised models cannot be validated at all and must be deployed on faith
E. Only the elbow heuristic is considered rigorous; all other diagnostics are discouraged

**2.** K-means minimises the inertia J(M,c) = Σᵢ ‖xᵢ − μ_c(i)‖². Lloyd's algorithm alternates assignment and update steps. Which statement about its convergence is correct?

A. Lloyd's algorithm is guaranteed to find the global optimum of J for any k ≥ 2
B. Lloyd's algorithm never increases J and terminates at a locally optimal partition, but the local optimum reached depends entirely on initialisation
C. Lloyd's algorithm is only guaranteed to converge when k = 2
D. Lloyd's algorithm increases J monotonically until a fixed budget of iterations is exhausted
E. Convergence requires the data to be normally distributed within each cluster

**3.** k-means++ seeding chooses the first centroid uniformly at random and each subsequent centroid with probability proportional to its squared distance from the nearest already-chosen centroid. What guarantee does this seeding alone (before any Lloyd iteration) provide?

A. It guarantees the exact global optimum in expectation
B. It guarantees E[J++] ≤ 8(ln k + 2)J*, a logarithmic-factor approximation to the optimal inertia J*
C. It guarantees convergence in exactly k iterations
D. It guarantees zero variance across different random seeds
E. It guarantees that every cluster will contain the same number of points

**4.** Because K-means assigns points via arg min_j ‖x − μ_j‖², the boundary between any two clusters is a hyperplane (the perpendicular bisector of the segment joining their centroids). What geometric limitation follows directly?

A. K-means can only discover clusters that are exactly spherical in three dimensions
B. K-means can only produce convex, roughly isotropic cells, so elongated, nested, or crescent-shaped structure is carved across rather than discovered
C. K-means cannot be applied to any dataset with more than two features
D. K-means requires the number of clusters to equal the number of features
E. K-means boundaries are always curved, never linear

**5.** Since inertia J(k) is non-increasing in k and reaches zero at k = m, it cannot be used alone to select k. A wallet-segmentation sweep shows the silhouette score peaking at k = 5 with a value around 0.3, well below the 0.7+ values seen on textbook synthetic blobs. The correct interpretation is:

A. A silhouette of 0.3 indicates the clustering has failed completely and should be discarded
B. Silhouette values around 0.3 are normal for real behavioural data; textbook illustrations use well-separated synthetic data and are not representative benchmarks
C. The silhouette score is miscalibrated and only the elbow heuristic should be trusted
D. A silhouette below 0.5 means k should be increased until the score exceeds 0.7
E. Silhouette scores are only meaningful when computed on raw, unscaled features

**6.** DBSCAN classifies a point as a core point when its ε-neighbourhood contains at least minPts points. Which statement about DBSCAN's reproducibility is correct per Proposition 5.3?

A. DBSCAN's clustering is fully random and changes completely with the order points are visited
B. The set of core points and their partition into clusters is order-independent; only the label of a border point reachable from two or more clusters may depend on visiting order
C. DBSCAN requires a fixed random seed to produce reproducible clusters, exactly like K-means
D. DBSCAN's core-point classification depends on the order points are visited, but cluster membership of core points does not
E. DBSCAN produces different numbers of clusters every time it is run, even with identical ε and minPts

**7.** DBSCAN's key operational weakness, addressed by HDBSCAN, is that:

A. It cannot handle more than two dimensions
B. A single global ε cannot serve clusters of very different densities: tuning for the sparse cluster merges the dense ones, and tuning for the dense ones dissolves the sparse cluster into noise
C. It requires labelled data to determine minPts
D. It always produces exactly as many clusters as K-means would on the same data
E. It cannot be combined with PCA-reduced feature spaces

**8.** A Gaussian mixture model with k components can simultaneously serve as a clustering method, a density estimator, a soft dimensionality-reduction encoding, and an anomaly detector. Which statement explains why?

A. Because the EM algorithm always converges to the same partition as K-means
B. Because a fitted mixture supplies, from the same parameters, a hard assignment (arg max responsibility), a density p(x), a k-dimensional responsibility vector, and a log-density anomaly score, all at once
C. Because GMMs require no covariance matrix estimation, unlike K-means
D. Because GMMs are mathematically identical to DBSCAN under a Gaussian kernel
E. Because a GMM component count k must always equal the number of PCA dimensions retained

**9.** K-means can be shown to be a limiting case of a Gaussian mixture model. Under which restriction on the GMM does this equivalence hold (Proposition 5.5)?

A. Fixing all mixture weights π_j equal and all covariance matrices Σ_j = σ²I as σ² → 0
B. Fixing the number of components k = 1
C. Setting all covariance matrices to the identity matrix scaled by an arbitrarily large constant
D. Removing the mixture weights entirely and using only the means
E. Restricting the data to exactly two dimensions

**10.** The distance-concentration result (Proposition 5.6) shows that as dimensionality n grows with i.i.d. coordinates, (D_max,n − D_min,n)/D_min,n converges to zero in probability. What is the practical consequence for nearest-neighbour-based methods like k-NN, K-means, and DBSCAN?

A. Nearest-neighbour search becomes faster as dimensionality grows
B. In high enough dimension with independent features, every point becomes roughly equidistant from every other, so "nearest neighbour" becomes close to meaningless
C. The curse of dimensionality only affects supervised learning, never unsupervised methods
D. Distance concentration guarantees perfect cluster separation in high dimensions
E. The result implies K-means should always be run with k equal to the number of dimensions

**11.** PCA admits two equivalent formulations. Variance maximisation selects the top-d eigenvectors of the covariance matrix S to maximise retained variance. What is the second, equivalent formulation (Proposition 5.7)?

A. Maximising the number of clusters recoverable from the projected data
B. Minimising the mean squared reconstruction error of an orthogonal rank-d projection, which is solved by exactly the same top-d eigenvectors
C. Minimising the silhouette coefficient of the projected data
D. Maximising the entropy of the projected distribution
E. Minimising the number of components needed to reach 50% explained variance

**12.** Two standardised features have sample covariance matrix S = [[4, 2], [2, 4]]. The eigenvalues are λ₁ = 6 and λ₂ = 2. What fraction of total variance does the first principal component explain, and what is the mean squared reconstruction error from projecting onto it alone?

A. 50% explained; reconstruction error 3
B. 75% explained; reconstruction error 2
C. 25% explained; reconstruction error 6
D. 100% explained; reconstruction error 0
E. 75% explained; reconstruction error 4

**13.** Why is randomised SVD preferred over forming the covariance matrix S = (1/m)XᵀX explicitly when computing PCA on large datasets?

A. Forming S explicitly squares the condition number and costs O(mn²), while randomised SVD obtains the top d components in roughly O(mnd)
B. Randomised SVD always produces a different, more accurate answer than the exact eigendecomposition
C. Forming S explicitly is numerically unstable only when n < m
D. Randomised SVD removes the need to centre the data before computing components
E. Randomised SVD is required whenever fewer than two components are retained

**14.** Fitting StandardScaler and PCA on the full dataset before the train/test split is flagged as a specific leakage pattern. What is the structural fix prescribed throughout the notes?

A. Always use MiniBatchKMeans instead of PCA to avoid the issue entirely
B. Fit every transformer inside a Pipeline object that is itself what gets cross-validated, so each fold re-fits its own scaler and PCA on training data only
C. Compute PCA components manually using NumPy to bypass scikit-learn's leakage-prone defaults
D. Apply PCA only to the test set, never to the training set
E. Increase the number of retained components until the leakage effect becomes negligible

**15.** A wallet's PCA reconstruction error is unusually high, while its Gaussian-mixture log-density score is unremarkable. What does this combination most plausibly indicate, per the notes' distinction between these two anomaly channels?

A. The wallet is a typical point that happens to sit off the learned low-dimensional subspace entirely, a different failure mode than a point merely sitting in a low-density region of that subspace
B. The GMM and PCA scores are mathematically identical, so this combination cannot occur
C. The wallet must have a data entry error, since both scores should always agree
D. The result implies k for K-means should be increased
E. Reconstruction error is only meaningful for categorical features, so the result should be ignored

**16.** Before shipping a segmentation model, the notes recommend re-fitting on adjacent time windows or bootstrap resamples and computing the adjusted Rand index (ARI) between the resulting partitions. An ARI of approximately 0.3 across quarters indicates:

A. An excellent, highly stable segmentation that should be shipped immediately
B. An unstable partition, essentially fitted noise, that should not be shipped as this quarter's segments will not resemble next quarter's
C. That k should be reduced to 0.3 times its current value
D. That the silhouette score must also equal 0.3 for consistency
E. That ARI is undefined for values below 0.5 and the check should be skipped

**17.** In credit scoring, why is clustering on unscaled mobile-money features (e.g., transaction value in KES alongside a small-integer count feature) described as producing a partition that "will look perfectly convincing" but is actually flawed?

A. Because K-means minimises a Euclidean norm, so a feature with a standard deviation in the thousands can dominate the objective roughly a million-fold over a count feature with a standard deviation around 10, yielding a partition on that one feature alone
B. Because unscaled features always cause K-means to fail to converge
C. Because clustering unscaled data always produces exactly one cluster
D. Because unscaled features make DBSCAN's ε undefined
E. Because scaling is only relevant for supervised learning, never for clustering

**18.** A dense layer with n_in inputs and n_out units holds how many parameters, including bias terms?

A. n_in × n_out
B. (n_in + 1) × n_out
C. n_in × (n_out + 1)
D. n_in + n_out
E. (n_in × n_out) + 1

**19.** Proposition 6.1 shows that composing purely affine (no non-linearity) layers yields another affine map. What is the direct consequence for a multi-layer perceptron with ϕ = identity in every layer?

A. Such a network can approximate any continuous function given enough depth
B. Such a network has exactly the representational capacity of a single affine map, regardless of depth L
C. Such a network is strictly more powerful than one with ReLU activations
D. Such a network cannot be trained with gradient descent under any circumstances
E. Such a network becomes equivalent to a convolutional layer once depth exceeds three

**20.** The sigmoid derivative σ'(z) = σ(z)(1 − σ(z)) is bounded above by exactly 1/4, attained only at z = 0. What is the direct architectural consequence of this bound stated in the notes?

A. Sigmoid networks converge faster than ReLU networks because of this bound
B. Every sigmoid layer multiplies the backward error signal by a factor of at most 1/4 before the weight matrices even contribute, seeding the vanishing-gradient problem in deep sigmoid stacks
C. The bound guarantees sigmoid units never saturate
D. The bound applies only to the output layer, never to hidden layers
E. The bound means sigmoid activations cannot be used with cross-entropy loss

**21.** A ReLU unit's pre-activation is negative for every training example it has ever seen. What is true of this "dead" unit, and what is the recommended first remedy?

A. It has zero gradient with respect to all incoming weights and cannot revive itself; the first-preference remedy is to reduce the learning rate and use He initialisation, escalating to leaky ReLU or ELU only if the problem persists
B. It will revive automatically once enough additional epochs pass
C. It should immediately be replaced with a sigmoid unit, which is the only fix
D. Dead units are beneficial regularisers and should not be fixed
E. The fix is to increase the learning rate substantially so the unit's weights move further

**22.** Softmax is shift-invariant: softmax(z + c·1) = softmax(z) for any constant c. Why does every serious deep learning library exploit this by subtracting max_k(z_k) before exponentiating?

A. To convert the softmax into a sigmoid function
B. To make every exponent non-positive, bounding each exponential at 1 and removing the risk of numerical overflow, without changing the result
C. To force the output probabilities to always sum to exactly 0
D. Because subtracting the max increases the theoretical capacity of the network
E. Because it removes the need for a cross-entropy loss function entirely

**23.** For yᵢ | xᵢ ~ Bernoulli(p̂ᵢ), Proposition 6.4 shows that maximising the likelihood of the observed labels is equivalent to minimising which quantity?

A. The mean squared error between yᵢ and p̂ᵢ
B. The binary cross-entropy loss, −(1/n)Σᵢ[yᵢ log p̂ᵢ + (1−yᵢ)log(1−p̂ᵢ)]
C. The hinge loss
D. The Kullback–Leibler divergence between two Gaussian distributions
E. The Gini impurity of the predicted probabilities

**24.** A mini-batch gradient estimator is described as unbiased with variance that shrinks as batch size grows. What is the practical trade-off this creates in choosing batch size?

A. Larger batches always strictly dominate smaller ones on every metric, so batch size should be maximised without limit
B. Smaller batches give noisier but cheaper-per-step gradient estimates, while larger batches reduce per-step variance at higher compute cost per step; batch size is a genuine trade-off, not a free lunch
C. Batch size has no effect on gradient variance, only on wall-clock time
D. Mini-batch gradients are always biased, regardless of batch size
E. Batch size must always equal the full training set size for convergence guarantees to hold

**25.** He initialisation sets the weight variance to ς² = 2/n_in for layers followed by ReLU. Per Proposition 6.11, why is the factor of 2 needed (versus Glorot's factor without it)?

A. Because ReLU has an unbounded derivative, requiring a larger variance ceiling
B. Because ReLU zeroes out half of a symmetric pre-activation distribution, so E[x²] for a ReLU-activated input equals half of the pre-activation variance; the factor of 2 exactly compensates so the second moment is preserved from layer to layer
C. Because He initialisation is designed only for the output layer, not hidden layers
D. Because the number 2 is empirically tuned per dataset with no theoretical justification
E. Because it matches the dimensionality of typical RGB image inputs

**26.** Batch normalisation replaces a pre-activation zᵢ with z̃ᵢ = γẑᵢ + β where ẑᵢ = (zᵢ − μ_B)/√(σ_B² + ϵ). The notes describe a specific "deployment bug factory." What is it?

A. BatchNorm always requires retraining from scratch when moved from GPU to CPU
B. At inference, BatchNorm must use accumulated moving averages rather than batch statistics (since a single served row has no meaningful batch statistics); hand-rolled serving code that skips this produces a model that scored well offline but fails in production
C. BatchNorm cannot be used with the Adam optimiser
D. BatchNorm silently disables dropout layers during training
E. BatchNorm requires labelled data at inference time to compute its normalisation

**27.** For small batch sizes or for sequence models, the notes recommend layer normalisation over batch normalisation. Why?

A. Layer normalisation normalises across the features of a single example rather than across the batch, so its behaviour is identical in training and inference and independent of batch size
B. Layer normalisation requires no learned parameters, unlike batch normalisation
C. Layer normalisation can only be applied to convolutional layers, not dense layers
D. Layer normalisation eliminates the need for any activation function
E. Layer normalisation is mathematically equivalent to dropout

**28.** Dropout randomly zeroes each unit's activation with probability p during training and rescales by 1/(1−p). Proposition 6.12 shows that, for a linear model, dropout is equivalent in expectation to:

A. A hard cluster assignment identical to K-means
B. Fitting the deterministically shrunk model qw together with an ℓ2 penalty whose per-coordinate weight is proportional to x_j², where q = 1−p
C. An unregularised ordinary least squares fit
D. A convolutional layer with stride equal to p
E. Replacing the loss function with the hinge loss

**29.** An embedding layer looks up a row of a learned matrix E for each vocabulary level. Proposition 6.13 shows an embedding lookup is mathematically equivalent to what operation, and why does the lookup implementation still matter practically?

A. Equivalent to a convolution; the lookup matters because convolutions cannot process categorical data directly
B. Equivalent to a dense layer applied to a one-hot encoding (with no bias); a literal one-hot matrix product is infeasible in memory for large vocabularies, so the table-lookup implementation matters for efficiency
C. Equivalent to max pooling; the lookup matters because pooling has no learnable parameters
D. Equivalent to batch normalisation; the lookup matters for numerical stability only
E. Equivalent to a softmax layer; the lookup matters only for multi-class problems

**30.** An agent-identifier embedding table has 120,000 levels and a production request arrives carrying an agent ID that did not exist at training time. What is the recommended handling per the "silent out-of-vocabulary failure" pitfall?

A. Reserve and train index 0 for unseen levels (e.g., by mapping a rare-level bucket to it), so an unseen agent receives a sensible trained average rather than an untrained random initialisation vector
B. Silently drop any row with an unseen agent ID from the prediction pipeline
C. Retrain the entire model in real time whenever a new agent ID appears
D. Map unseen agent IDs to the embedding of the most recently seen agent
E. Raise the embedding dimension until all possible future agent IDs are covered in advance

**31.** For a 2D convolutional layer (Definition 7.1) with kernel size f, cin input channels and cout output channels, the parameter count is f²·cin·cout + cout. Why does this count NOT depend on the spatial dimensions H and W of the input?

A. Because convolution always pads the input to a fixed size regardless of H and W
B. Because a kernel's weights are shared across every spatial position; only the number of positions the kernel visits (and hence compute cost, not parameter count) grows with H and W
C. Because H and W are always fixed at 224 by convention
D. Because parameters are only associated with pooling layers, not convolutional layers
E. Because cin and cout are computed as functions of H and W

**32.** A stack of three consecutive 3×3 convolutions at unit stride has the same receptive field as a single 7×7 convolution. What is the stated parameter-efficiency argument for preferring the stack (per channel count c)?

A. The 7×7 convolution uses 27c² weights versus 49c² for the stack, favouring the single large kernel
B. The stack of three 3×3 convolutions uses 27c² weights versus 49c² for the single 7×7 convolution, a saving of roughly 45%, while also inserting two extra non-linearities
C. Both approaches use exactly the same number of weights, so the choice is purely about training speed
D. The stack requires four times as many weights as the single 7×7 convolution
E. Receptive field size is unrelated to kernel stacking in convolutional networks

**33.** Global average pooling at the top of a CNN backbone replaces each H'×W' feature map with its scalar mean. Compared with flattening the same 7×7×2048 tensor before a dense classification layer, what is the stated benefit?

A. Flattening always produces higher accuracy, so global average pooling is only used to save memory at the cost of performance
B. Global average pooling avoids the roughly 49-fold parameter blow-up that flattening would cause in the following dense layer, and makes the network tolerant of changes in input resolution
C. Global average pooling requires labelled bounding boxes, unlike flattening
D. Flattening and global average pooling produce identical parameter counts in the classification head
E. Global average pooling can only be applied to grayscale images

**34.** The residual block computes y = φ(F(x;θ) + x). Per the "gradient highway" theorem, why does this architecture solve the degradation problem where deeper plain networks perform worse even on the training set?

A. Because residual blocks always have fewer parameters than plain blocks
B. Because the gradient reaching an early layer contains an additive identity term equal to ∂L/∂x_L exactly, regardless of depth, so it cannot vanish even if every residual branch's Jacobian shrinks toward zero
C. Because residual blocks eliminate the need for any activation function
D. Because residual connections increase the effective learning rate automatically
E. Because residual blocks replace convolution with a fully connected layer at every stage

**35.** In a ResNet-50-style bottleneck block, a plain 3×3 convolution mapping 256 channels to 256 channels would cost about 9×256² ≈ 590,000 weights. The bottleneck design (1×1 squeeze to 64 channels, 3×3 at 64 channels, 1×1 restore to 256) costs approximately 2×256×64 + 9×64² ≈ 70,000 weights. What is the architectural principle this bottleneck design illustrates?

A. Bottlenecks eliminate the need for skip connections entirely
B. Depth is only affordable if the per-block cost is kept low; squeezing channels before the expensive spatial convolution and restoring them afterward buys roughly an eight-fold saving per block
C. Bottleneck blocks always underperform plain 3×3 blocks and are used only for memory reasons
D. The bottleneck design removes the need for batch normalisation
E. 1×1 convolutions cannot be used to change channel count, only spatial resolution

**36.** During transfer learning, a pre-trained backbone's BatchNorm layers carry running averages computed over the original pre-training data. The "BatchNorm trap" pitfall describes what failure mode when fine-tuning begins?

A. Validation accuracy improves steadily forever once fine-tuning starts, with no risk of collapse
B. If the backbone is called without training=False (or its BatchNorm layers are not frozen), normalisation switches to the small fine-tuning batch's statistics and progressively overwrites the pre-trained running averages, often causing a sudden accuracy collapse right at the unfreeze point that survives even a later switch back to inference mode
C. The trap only affects models trained without any convolutional layers
D. The trap is fixed automatically by increasing the learning rate after unfreezing
E. The trap causes gradients to vanish but never causes accuracy to drop

**37.** For a crop-disease photo classifier, a horizontal flip is a label-preserving augmentation. For a lender's identity-document tamper classifier, the same horizontal flip is described as actively harmful. Why?

A. Because horizontal flips are computationally more expensive for documents than for crop photos
B. Because a mirrored identity document changes the label's meaning: mirrored text is itself a tamper signal in the lender's data, so flipping manufactures label noise and penalises the model for detecting real evidence of tampering
C. Because document images always have a different aspect ratio than crop-disease images
D. Because flips are mathematically undefined for grayscale images
E. Because the augmentation library only supports flips for RGB crop images

**38.** A SimpleRNN carries a hidden state forward one time step at a time. Its core weakness, per the RNN lab material, is:

A. It requires convolutional layers to process any sequence longer than ten steps
B. The vanishing gradient: over a long sequence, the gradient signal shrinks multiplicatively as it is backpropagated through time, preventing the network from learning dependencies more than a few steps back
C. It cannot be trained using backpropagation at all
D. It requires the input sequence to be perfectly periodic
E. It has no hidden state, unlike an LSTM

**39.** An LSTM adds a cell state and three gates (forget, input, output) compared with a SimpleRNN. What is the structural reason this allows learning of longer-range dependencies?

A. The cell state gives the gradient an additive path through time instead of a purely multiplicative one, avoiding the repeated multiplicative shrinkage that causes vanishing gradients in a SimpleRNN
B. The three gates triple the effective learning rate automatically
C. The cell state removes the need for backpropagation through time entirely
D. LSTMs process the entire sequence in parallel rather than step by step, unlike SimpleRNNs
E. The forget gate eliminates the need for any activation function in the network

**40.** In the RNN forecasting lab, the train/test split is made strictly by date rather than randomly. What is the stated reason?

A. Random splits are computationally more expensive than temporal splits for time series
B. A random split would let the model train on future values and predict the past, which constitutes leakage
C. Date-based splits are required by TensorFlow's API and cannot be avoided
D. Random splits always produce class imbalance in time series data
E. Temporal splits are only needed when the series has fewer than 100 observations

**41.** Retrieval-Augmented Generation (RAG) is motivated as a cheaper alternative to fine-tuning for grounding a language model in domain-specific documents it has never seen. What is the central mechanism RAG uses instead of updating model weights?

A. RAG periodically retrains the entire language model overnight on new documents
B. RAG fetches relevant text at question time via retrieval and pastes it into the prompt, leaving the model's weights untouched
C. RAG requires replacing the language model with a rule-based lookup table
D. RAG works only when the language model has an unlimited context window
E. RAG converts every document into a supervised classification label before querying

**42.** In a RAG pipeline, embedding vectors are typically compared using cosine similarity rather than Euclidean distance. Why, and what computational simplification follows when vectors are L2-normalised to unit length?

A. Because direction, not magnitude, carries the meaning of a text embedding; when vectors are unit-normalised, cosine similarity reduces to a simple dot product, making retrieval over the whole index a single matrix multiplication
B. Because cosine similarity is always computationally cheaper than Euclidean distance regardless of normalisation
C. Because Euclidean distance is undefined for vectors with more than 100 dimensions
D. Because normalisation converts the embeddings into one-hot vectors
E. Because cosine similarity requires labelled training data, unlike Euclidean distance

**43.** RAG has two distinct failure modes that must be diagnosed separately before attempting a fix. What are they, and how is each measured?

A. Overfitting, measured by training loss, and underfitting, measured by validation loss
B. Retrieval failure (the right chunk was never fetched), measured by hit rate/recall at k, and generation failure (the right chunk was fetched but the model still answered badly), measured by faithfulness to the retrieved context
C. Vanishing gradients and exploding gradients, both measured by the same loss curve
D. Tokenisation failure and embedding failure, both fixed by increasing model size
E. Class imbalance and data leakage, both fixed by resampling the corpus

**44.** In RAG chunking, why is chunk size described as "the central trade-off"?

A. Chunks that are too small are retrieved without enough surrounding context to answer from, while chunks that are too large dilute the embedding (forcing one vector to represent several unrelated ideas) and waste prompt tokens
B. Chunk size has no effect on retrieval quality, only on storage cost
C. Larger chunks always improve both retrieval accuracy and generation faithfulness simultaneously with no downside
D. Chunk size must always equal exactly one sentence for retrieval to function
E. Smaller chunks always eliminate the risk of hallucination entirely

**45.** A "temperature" of 0 is recommended for the generation step of a RAG pipeline. Why?

A. Because temperature 0 disables the retrieval step entirely
B. Because a temperature of 0 makes output near-deterministic, which is desirable in RAG since the answer should be pinned to the retrieved context rather than creative
C. Because temperature only affects the embedding model, not the chat model
D. Because higher temperatures always produce faster inference
E. Because temperature 0 guarantees zero hallucination under any retrieval quality

**46.** "Hallucination" in the context of language models is best described, per the glossary, as:

A. A model producing output at a temperature setting above 1.0
B. A fluent, confident, and factually wrong statement produced by a language model, typically about something absent from its training data
C. A synonym for overfitting in the supervised learning sense
D. A hardware fault causing corrupted token output
E. Any output that disagrees with a retrieved document, regardless of factual accuracy

**47.** In the self-attention mechanism underlying transformer architectures, attention weights for a query are typically computed as softmax(QKᵀ/√d_k)V. What is the role of the 1/√d_k scaling factor?

A. It converts the attention mechanism into a convolution
B. It prevents the dot products QKᵀ from growing large in magnitude as dimensionality d_k increases, which would otherwise push softmax into regions of extremely small gradients
C. It guarantees the attention weights sum to exactly the sequence length
D. It removes the need for positional information in the input
E. It is only used during inference, never during training

**48.** Why do transformer architectures require an explicit positional encoding mechanism, unlike RNNs?

A. Because self-attention treats the input as an unordered set of tokens and has no inherent notion of sequence order, so position must be injected separately
B. Because transformers cannot process more than one token at a time
C. Because positional encoding replaces the need for an embedding layer
D. Because RNNs also require positional encoding, so the two architectures are identical in this respect
E. Because positional encoding is only needed for image inputs, not text

**49.** A "token", per the glossary, is the unit a language model reads and bills in. Roughly how does a token compare to an English word?

A. One token is roughly ten English words
B. One token is roughly three-quarters of an English word
C. One token is always exactly one English word
D. One token is roughly one English sentence
E. Tokens and words are unrelated units with no consistent relationship

**50.** A course notebook uses no RAG framework (no LangChain, no LlamaIndex, no vector database), relying instead on NumPy arrays and direct HTTP calls to embedding and chat models. What pedagogical point does this design choice make, per the lab material?

A. That RAG frameworks are illegal to use in production systems
B. That every moving part of the pipeline (chunking, embedding, similarity search, prompt construction) stays visible and inspectable rather than hidden behind a framework abstraction
C. That vector databases are strictly required for any RAG system to function at all
D. That NumPy is faster than any dedicated vector database at any scale
E. That embedding models cannot be accessed via HTTP APIs

---

## Answer Key — Paper 2

1. B — internal indices, likelihood-penalised criteria, and downstream utility, since no ground-truth loss exists.
2. B — J is non-increasing, convergence is to a local optimum dependent on initialisation.
3. B — this is the stated k-means++ approximation guarantee.
4. B — arg-min-distance assignment always yields convex, hyperplane-bounded cells.
5. B — 0.3 is normal for real behavioural data; synthetic benchmarks are not comparable.
6. B — core-point structure is order-independent; only ambiguous border points vary.
7. B — a single global ε cannot serve clusters of very different densities.
8. B — one fitted mixture yields hard assignment, density, soft encoding, and anomaly score together.
9. A — equal weights and Σⱼ = σ²I as σ²→0 recovers K-means from a GMM.
10. B — distances concentrate, so nearest-neighbour structure becomes uninformative.
11. B — reconstruction-error minimisation is provably equivalent to variance maximisation.
12. B — λ1/(λ1+λ2) = 6/8 = 75%; reconstruction error equals the discarded eigenvalue λ2 = 2.
13. A — forming S costs O(mn²) and squares the condition number; randomised SVD is O(mnd).
14. B — every fitted transform belongs inside the Pipeline that is cross-validated as a whole.
15. A — reconstruction error flags points off the subspace, a distinct failure mode from low mixture density.
16. B — ARI ≈0.3 signals an unstable, non-reproducible partition.
17. A — Euclidean-norm clustering is dominated by the largest-scale unscaled feature.
18. B — (n_in + 1) × n_out, including one bias per output unit.
19. B — composed affine maps stay affine regardless of depth.
20. B — each sigmoid layer caps backward signal at a factor of 1/4, seeding vanishing gradients.
21. A — zero gradient forever; fix by lowering the learning rate and using He init first.
22. B — subtracting the max keeps every exponent ≤0, preventing overflow without changing the result.
23. B — maximising Bernoulli likelihood is exactly minimising binary cross-entropy.
24. B — smaller batches trade noise for cost; larger batches trade cost for lower variance.
25. B — ReLU zeroes half the symmetric input distribution, halving E[x²], so variance must double to compensate.
26. B — production requests may be single rows, so training-mode batch statistics must not be used at inference.
27. A — layer norm normalises within an example, making it batch-size- and mode-independent.
28. B — this is exactly Proposition 6.12's stated equivalence.
29. B — an embedding lookup equals a bias-free dense layer on a one-hot vector; the table lookup avoids materialising that one-hot matrix.
30. A — a trained index 0 gives unseen agents a sensible average vector, not an untrained random one.
31. B — weight sharing means parameters don't scale with spatial size, though compute does.
32. B — 27c² for the stack versus 49c² for the single 7×7 kernel, roughly 45% fewer weights.
33. B — avoids the ~49x parameter blow-up of flattening and decouples the head from input resolution.
34. B — the additive identity term in the gradient recursion cannot vanish, unlike a pure product of Jacobians.
35. B — squeezing then restoring channels affordably buys depth, an eight-fold saving per block.
36. B — unfrozen BatchNorm overwrites pretrained running statistics with small-batch statistics, corrupting the checkpoint itself.
37. B — a mirrored document changes the true label since mirrored text is itself evidence of tampering.
38. B — this is the definition of the vanishing gradient problem in a SimpleRNN.
39. A — the additive cell-state path avoids the repeated multiplicative shrinkage of a SimpleRNN.
40. B — a random split would leak future information into the training of a past prediction.
41. B — RAG fetches text at query time and pastes it into the prompt, leaving weights untouched.
42. A — direction carries meaning; unit-normalised cosine similarity is a single dot product / matmul.
43. B — retrieval failure (hit rate at k) versus generation failure (faithfulness) are diagnosed separately.
44. A — small chunks lose context, large chunks dilute the embedding and waste tokens.
45. B — temperature 0 pins output to the retrieved context rather than allowing creative drift.
46. B — this is the glossary's definition of hallucination.
47. B — scaling by 1/√d_k prevents large-magnitude dot products from saturating the softmax into low-gradient regions.
48. A — self-attention is permutation-invariant over tokens, so position must be injected explicitly.
49. B — a token is roughly three-quarters of an English word, per the glossary.
50. B — the framework-free design keeps every pipeline step visible rather than hidden by abstraction.
