# DSA 8401 Applied Machine Learning — Practice Paper 1
## Fundamentals of ML, Data Preparation & Feature Engineering, Supervised Learning
**50 questions. Five options each (A–E). One correct answer per question. Answer key at the end.**

---

**1.** A digital lender defines: Task T = flag a transaction as fraudulent at initiation; Experience E = eighteen months of transaction logs with labels arriving weeks late from chargebacks; Performance P = cost-weighted recall at a fixed daily alert budget. Under Mitchell's operational definition of learning, which statement about this instantiation is correct?

A. The definition is invalid because E must consist of fully labelled data available at training time, not delayed labels
B. P is mis-specified; only accuracy satisfies Mitchell's definition of a performance measure
C. The definition is valid, but E and P already encode significant engineering reality: delayed, noisy labels and an operations-constrained metric
D. T is invalid because a task must be defined independently of the deployment moment, not "at the moment it is initiated"
E. Mitchell's definition requires E, T and P to be drawn from the same distribution, which is violated here

**2.** The expected risk R(f) = E_(x,y)~D[L(y, f(x))] cannot be computed directly in practice. Why, and what do we compute instead?

A. Because L is non-convex; we instead compute a convexified surrogate risk over the same distribution D
B. Because D is unknown; we instead compute the empirical risk over a finite training sample drawn i.i.d. from D
C. Because x and y are not independent; we instead compute a conditional risk given x
D. Because f is not differentiable; we instead compute a subgradient approximation of R(f)
E. Because the loss function L is unbounded; we instead compute a clipped empirical loss

**3.** A lender's default model was trained pre-pandemic. After a macroeconomic shock, applicants with an identical transaction profile now default at a substantially higher rate, while the demographic mix of applicants is unchanged. This is best classified as:

A. Label-prior shift only, since P(y) has changed while P(x) and P(y|x) are both fixed
B. Data drift (covariate shift), since the shock altered the input distribution P(x)
C. Concept drift, since P(y|x) has changed while the applicant profile distribution P(x) is essentially unchanged
D. Sampling bias, since the training sample no longer represents the population of interest
E. Training/serving skew, since the feature computation pipeline itself has changed

**4.** According to the course's "build-or-not" framework, which scenario most strongly argues for a hand-written rule rather than an ML model?

A. Fraud patterns that shift on a weekly basis, defeating any static rulebook
B. A high-dimensional, strongly non-linear relationship between hundreds of interacting behavioural features
C. A short, auditable rule already meets the required standard and errors must be fully explainable
D. The organisation wants to extract latent structure such as customer segments from unlabelled data
E. The environment is expected to drift continuously as adversaries adapt

**5.** For a continuous target, the constant prediction that minimises expected squared error and the constant prediction that minimises expected absolute error are, respectively:

A. The conditional mean and the conditional mean
B. The conditional mode and the conditional median
C. The conditional mean and the conditional median
D. The conditional median and the conditional mean
E. The conditional variance and the conditional mean

**6.** A portfolio has a 3% default rate. A model that predicts "repays" for every applicant achieves roughly 97% accuracy while catching zero defaulters. This scenario is used in the course notes to illustrate that:

A. Accuracy is undefined under class imbalance and cannot legally be reported
B. A model must exceed 97% accuracy before accuracy even begins to carry decision-relevant information, since it weighs all errors equally under asymmetric costs
C. The correct fix is always to oversample the minority class before any model is trained
D. Precision and recall are mathematically undefined when a class has zero true positives
E. ROC-AUC and accuracy are always equal under class imbalance, so either may be reported

**7.** In the end-to-end ML lifecycle presented as a loop rather than a waterfall, which claim about where effort is spent is asserted by the course notes?

A. Roughly 80% of real project effort lives in framing and data work (steps 1–3), not in model selection
B. Model selection and hyperparameter tuning dominate the effort budget of real-world projects, mirroring the emphasis of most textbooks
C. Deployment and monitoring together consume the majority of project time in a typical first iteration
D. Framing the problem is a one-time, ten-minute exercise that should not be revisited once modelling begins
E. The loop terminates once a model beats its baseline on the validation set; monitoring is optional afterwards

**8.** Data leakage is formally defined in terms of a mismatch between two feature maps. Which pair, and what does the mismatch mean?

A. φ_train and φ_serve; leakage exists when the model is evaluated on φ_train(x) but will be applied to a φ_serve(x) that differs in distribution
B. φ_public and φ_private; leakage exists whenever a private feature is exposed in a public API response
C. φ_raw and φ_scaled; leakage exists whenever any scaling is applied before model training
D. φ_train and φ_test; leakage exists only when the same row literally appears in both splits
E. φ_linear and φ_nonlinear; leakage exists whenever a nonlinear transform is fitted on the full dataset

**9.** A lender considers simply removing gender and region from the feature set to "achieve fairness." Which principle from the course's fairness discussion is most directly violated by assuming this is sufficient?

A. Group-fairness criteria are provably always mutually satisfiable once protected attributes are removed
B. Removing the protected attribute does not, by itself, produce fairness, because other features can act as proxies for it
C. Explainability requirements are automatically satisfied once protected attributes are dropped
D. Fairness auditing is unnecessary once the model has been dropped from production monitoring
E. Dropping a feature always increases model accuracy on the protected subgroup

**10.** A lender sets cost of a missed default at KES 10,000 and cost of a wrongly declined good borrower at KES 3,000, with correct decisions costing nothing. Using the cost-optimal threshold rule, at what calibrated probability t* should the lender begin flagging an applicant as high risk?

A. t* ≈ 0.500
B. t* ≈ 0.769
C. t* ≈ 0.231
D. t* ≈ 0.300
E. t* ≈ 0.100

**11.** Under Rubin's taxonomy, a lender notices GPS coordinates are missing exclusively for feature-phone users, and this is fully explained by the observed device-type column. This missingness pattern is:

A. MNAR, because the missingness depends on the unobserved GPS value itself
B. MCAR, because the missing rows form a uniform random sample
C. MAR, because conditional on an observed column (device type), missingness carries no further information about the missing GPS values
D. Ignorable only if the sample size exceeds 10,000 rows
E. Structurally undefined, since Rubin's framework only applies to numeric, not geospatial, features

**12.** An analyst finds that among applicants with GPS present, the 90-day default rate is 4.1%, while among those with GPS missing, it is 9.7%. Per the course's diagnostic protocol, the correct response is to:

A. Discard the GPS column entirely, since missing data can never be predictive
B. Impute the missing GPS values using the global mean latitude and longitude
C. Conclude the missingness is MCAR and proceed with simple mean imputation
D. Retain a binary missingness indicator as a feature in its own right, since the gap suggests MNAR-like structure where missingness itself is signal
E. Apply z-score outlier removal to the GPS column before imputing

**13.** Why does the course recommend the median, not the mean, for simple imputation of transaction-amount columns?

A. The median is always unbiased under MNAR, while the mean is never unbiased under any missingness regime
B. Transaction amounts are heavy-tailed, so a single very large value can drag the mean far from a typical value, while the median is robust to that tail
C. scikit-learn's SimpleImputer only supports the median strategy for numeric columns
D. The mean requires the data to be normally distributed, which is a strict prerequisite for any imputation
E. The median is computed on the full dataset (train and test together), avoiding leakage, whereas the mean cannot be

**14.** For a high-cardinality categorical column, why does the course recommend an explicit "UNKNOWN" sentinel over most-frequent-category imputation for missing categorical values?

A. Most-frequent imputation is computationally infeasible for more than 50 categories
B. A sentinel preserves the information that the value was absent rather than silently inflating the modal category's apparent frequency
C. scikit-learn's OneHotEncoder rejects most-frequent imputation for categorical columns
D. A sentinel guarantees the encoder will never encounter an unseen category at serving time
E. Most-frequent imputation always introduces target leakage, while a sentinel never does

**15.** Scikit-learn's IterativeImputer, used to approximate MICE, models each incomplete column as a function of the other columns and cycles until the fills stabilise. Which statement about it is accurate?

A. It uses a random forest regressor by default and cannot express fill uncertainty
B. It uses a Bayesian ridge regressor by default; with sample_posterior=True, fills become draws rather than point estimates, propagating imputation uncertainty
C. It is guaranteed to converge to the unique MCAR-optimal solution within a fixed number of steps set by max_iter
D. It requires no random_state, since the underlying regressor is deterministic
E. It should always be preferred over simple imputation because it never degrades serving-time latency

**16.** A feature has quartiles Q1 = 20 and Q3 = 80 (IQR = 60). Under the standard IQR outlier rule used in the notes, a point with value 185 is:

A. Flagged as an outlier, since the upper fence is Q3 + 1.5·IQR = 80 + 90 = 170, and 185 exceeds it
B. Not an outlier, since the upper fence is 260
C. An outlier only if a z-score confirms |z| > 3 as well
D. Not an outlier, since the IQR rule only applies to the lower tail for right-skewed monetary data
E. Undeterminable without knowing the mean and standard deviation

**17.** The Isolation Forest anomaly score is defined as s(x) = 2^(−E[h(x)]/c(n)), where h(x) is the average isolation depth over trees and c(n) normalises for the expected depth of a binary search tree on n points. A score close to 1 indicates:

A. A typical, well-connected inlier that requires many splits to isolate
B. An anomaly, since it was isolated in unusually few splits relative to the expected depth
C. A tie between the Gini and entropy impurity criteria at that point
D. That the point lies exactly on the K-means cluster boundary
E. Numerical instability in the isolation forest's random seed

**18.** A cleaning script winsorizes or deletes a small cluster of applicants whose accounts show three enormous round-figure deposits shortly before applying, a known balance-inflation gaming pattern. Per the course's "deleting the signal" pitfall, this is:

A. Correct practice, since any value beyond three IQRs should always be removed for model stability
B. A mistake if applied mechanically, because these are exactly the behavioural extremes the model is meant to detect, not data-quality errors
C. Irrelevant, since gaming patterns cannot be captured by any tabular feature set
D. Correct only when the deposits exceed the 99.9th percentile of the full unsplit dataset
E. Acceptable, provided the deletion happens after model training rather than before

**19.** Why must entity resolution (merging records for the same real-world customer across multiple SIM cards) happen before the train/test split, and splitting must then use the resolved customer_id?

A. Because scikit-learn's train_test_split function raises an error on duplicated names
B. Because failing to do so scatters copies of the same customer across train and test, letting the model "recognise" the test customer from training rather than generalise
C. Because entity resolution is only mathematically valid on data structured by SIM identifier
D. Because fuzzy name matching requires a fixed train/test split to compute similarity scores
E. Because regulatory rules in Kenya require deduplication only for the test set

**20.** Which scaler is most fragile to a single extreme outlier, because that one point redefines the scaling bounds and crushes the rest of the distribution toward zero?

A. The robust scaler, since it uses the interquartile range
B. Standardization (z-score scaling), since it uses the sample mean
C. Min–max scaling, since a single outlier redefines x_max or x_min
D. Quantile transformation, since it maps values through the empirical CDF
E. Box–Cox transformation, since λ is estimated by maximum likelihood

**21.** Net cash-flow features in a mobile-money dataset can be negative. Which transform is used specifically because it extends the Box–Cox idea to handle the full real line, including negative values?

A. The log(1+x) transform
B. The Yeo–Johnson transform
C. Min–max scaling
D. The hashing trick
E. Standardization

**22.** With an intercept term, one-hot encoding all K levels of a nominal feature (rather than K−1) creates a problem for an unregularised linear model. This problem is:

A. The "dummy trap": the K indicator columns sum to one, producing perfect collinearity with the intercept
B. Catastrophic forgetting, since the model overwrites earlier learned weights
C. The curse of dimensionality, which only affects distance-based learners, not linear ones
D. Vanishing gradients in the coefficient estimation
E. A violation of Rubin's MAR assumption

**23.** The hashing trick maps a categorical level c to a bucket via φ(c) = h(c) mod d. Which property makes it attractive for an agent-identifier column in a system where new agents are onboarded continuously?

A. It guarantees zero collisions regardless of d, unlike frequency encoding
B. It requires no fitted vocabulary at all, so levels unseen at training time are still mapped to a valid bucket at serving time
C. It always outperforms target encoding in predictive lift
D. It automatically satisfies the MAR assumption for the encoded column
E. It eliminates the need for any train/test split discipline

**24.** Target encoding is defined as TE(c) = (n_c·ȳ_c + m·ȳ) / (n_c + m). For a category with n_c = 4, ȳ_c = 0.75, global mean ȳ = 0.05, and smoothing constant m = 20, the encoded value is closest to:

A. 0.750
B. 0.050
C. 0.400
D. 0.167
E. 0.283

**25.** Why does naive (non-out-of-fold) target encoding constitute self-leakage, and what is the standard fix?

A. Because each row's encoded value is computed partly from its own label; the fix is to encode each fold using statistics fitted only on the other folds
B. Because target encoding always violates GDPR; the fix is to use one-hot encoding instead
C. Because target encoding cannot represent unseen categories; the fix is to use the hashing trick instead
D. Because target encoding requires a neural network to compute; the fix is gradient clipping
E. Because target encoding is only valid for continuous targets; the fix is bucketing the target first

**26.** For a periodic feature such as hour-of-day, why must both sin(2πt/T) and cos(2πt/T) be included rather than just one?

A. Because scikit-learn's ColumnTransformer requires an even number of derived features
B. Because either coordinate alone maps two genuinely different times of day to the same encoded value, destroying information
C. Because sin and cos together are required to satisfy the MCAR assumption
D. Because a single trigonometric feature always produces negative variance
E. Because tree-based models cannot split on real-valued features at all

**27.** In scorecard practice, the weight of evidence for a bin is WoE(bin) = ln[P(bin | y=0) / P(bin | y=1)]. A logistic regression fitted on WoE-coded bins is described in the notes as being equivalent to:

A. A Random Forest with monotonic constraints
B. The traditional credit scorecard format that many East African lenders' risk committees expect
C. A kernel SVM with an RBF kernel
D. An unregularised neural network with a single hidden layer
E. A Gaussian mixture model with two components

**28.** Huyen's field guide enumerates six common causes of leakage. Which of the following is explicitly listed as one of the six?

A. Using more than fifteen engineered features in a single model
B. Choosing a tree-based model over a linear model for tabular data
C. Scaling or imputing using statistics computed on the full dataset before splitting
D. Reporting PR-AUC instead of ROC-AUC on an imbalanced dataset
E. Training a model for more than 100 boosting rounds

**29.** In the temporal evaluation design for the credit scorer, every observation carries two clocks: the feature window ends at scoring time t, and the label matures over [t, t+90 days]. A loan disbursed 40 days before a split boundary should be:

A. Included as "non-default", since 40 days is close enough to the 90-day window to be treated as resolved
B. Excluded from the split, since its label window has not closed and including it as non-default injects optimism
C. Included as "default", to bias the model conservatively toward caution
D. Duplicated across both the training and validation sets to increase statistical power
E. Used only for scaling and imputation statistics, never for the model target

**30.** Logistic regression models the log-odds as a linear function: log[p̂/(1−p̂)] = θᵀx + b. If the fitted, standardised coefficient on tenure_months is θⱼ = −0.35, one standard deviation of additional tenure multiplies the odds of default by approximately:

A. 1.42
B. 0.35
C. 0.70
D. −0.35
E. 0.50

**31.** In scikit-learn's LogisticRegression, the regularisation strength parameter C is related to the penalty weight λ by:

A. C = λ, so a larger C means stronger regularisation
B. C = 1/λ, so a larger C means weaker regularisation
C. C = λ², so C and regularisation strength move in the same direction
D. C has no relationship to λ; it only controls the solver's convergence tolerance
E. C = −λ, so negative C values indicate strong regularisation

**32.** The soft-margin SVM objective is min (1/2)‖w‖² + C·Σᵢ max(0, 1 − yᵢ(wᵀxᵢ + b)). Which statement about this objective is correct?

A. Every training point contributes equally to the final decision boundary, regardless of its margin
B. Only points on or inside the margin (the support vectors) carry non-zero weight in the solution; the rest could be deleted without moving the boundary
C. The hinge loss term is zero everywhere except at points exactly on the boundary
D. Increasing C always widens the margin regardless of the data
E. The objective is non-convex, so multiple restarts are required to find a global optimum

**33.** Why are k-nearest-neighbours classifiers described as having a latency profile that is "the exact opposite of what production wants"?

A. Because kNN requires GPU acceleration unavailable in most serving environments
B. Because there is no training phase, but each prediction costs O(m) distance computations against the training set, unlike models with cheap prediction and expensive training
C. Because kNN can only be implemented in a non-differentiable programming language
D. Because kNN requires the entire training set to be re-shuffled before every prediction
E. Because kNN's memory footprint grows quadratically with the number of features regardless of sample size

**34.** For a positive class with true recall r = 0.7 estimated from a validation set containing n+ = 150 positives, the standard error of the recall estimate, se(recall) = sqrt(r(1−r)/n+), is closest to:

A. 0.0037
B. 0.700
C. 0.037
D. 0.21
E. 0.0007

**35.** Stratified k-fold cross-validation, rather than plain k-fold, is described as mandatory under class imbalance because:

A. It guarantees the model achieves higher accuracy on the minority class
B. An unstratified fold may contain almost no positives at low prevalence, which makes the standard-error of any recall-type estimate explode
C. It removes the need for a held-out test set entirely
D. It automatically corrects for data drift between folds
E. It converts the classification problem into a regression problem

**36.** Nested cross-validation separates two roles that a single cross-validation loop conflates. What is the key distinction?

A. An inner loop tunes hyperparameters within each outer training set; the outer loop evaluates the whole selection procedure on data it never touched, avoiding the optimism of tuning and reporting on the same folds
B. An inner loop trains on the test set; the outer loop trains on the training set
C. Nested CV eliminates the need for a validation set by using bootstrap resampling exclusively
D. The inner loop always uses stratified folds, while the outer loop always uses random folds
E. Nested CV multiplies compute by a factor of two, regardless of the number of inner folds

**37.** Under Proposition 3.2, AUC has a probabilistic interpretation. If s+ is the score of a random positive and s− the score of an independent random negative, AUC equals:

A. P(s+ < s−)
B. P(s+ > s−) + (1/2)·P(s+ = s−)
C. P(s+ = s−) only
D. The correlation coefficient between s+ and s−
E. 1 − P(s+ > s−)

**38.** Under heavy class imbalance (e.g. 3% prevalence), which statement about ROC-AUC versus PR-AUC is correct, per the course's rule of thumb?

A. ROC-AUC and PR-AUC are always numerically identical under any prevalence
B. ROC-AUC can look deceptively good because its false-positive-rate denominator is dominated by the huge negative class, while PR-AUC (whose chance baseline equals the prevalence) exposes model differences more clearly among rare positives
C. PR-AUC should never be reported when positives are rare, since its chance baseline is undefined
D. ROC-AUC's chance baseline shifts with prevalence, while PR-AUC's chance baseline is fixed at 0.5
E. Under imbalance, accuracy becomes a better summary statistic than either curve

**39.** A score function p̂ is "calibrated" if, per Definition 3.6:

A. It ranks positives above negatives with probability 1
B. For every q in [0,1], among all cases scored q, about a fraction q are actually positive
C. Its Brier score is exactly zero
D. It is monotonically increasing in every input feature
E. Its ROC-AUC exceeds 0.9

**40.** Per the rule of thumb from Niculescu-Mizil and Caruana cited in the notes, which pairing is most accurate regarding out-of-the-box calibration?

A. Logistic regression is usually well calibrated out of the box; SVMs and boosted trees typically are not and benefit from post-hoc calibration
B. Boosted trees are always perfectly calibrated because their leaves output empirical frequencies
C. SVMs are inherently probabilistic and need no calibration
D. Logistic regression requires isotonic regression before its scores can be trusted at all
E. Calibration quality is unrelated to model family and depends only on dataset size

**41.** SMOTE creates a synthetic minority point as x_new = xᵢ + λ(x_zᵢ − xᵢ) with λ ~ U(0,1). The course identifies a "cardinal sin" regarding when this must run. What is it?

A. SMOTE must run after fitting the final model, never before
B. SMOTE must run inside each cross-validation fold, on the training portion only; running it before the split lets synthetic points derived from validation-fold originals leak into training
C. SMOTE must always use k=1 nearest neighbour to avoid interpolation artefacts
D. SMOTE must never be combined with class weighting under any circumstances
E. SMOTE must be applied to the test set to match the training class balance

**42.** McNemar's test for comparing two classifiers on one test set uses the discordant counts n01 and n10 in the statistic χ² = (|n01 − n10| − 1)² / (n01 + n10). This test is preferred in the notes because:

A. It requires no assumption about the distribution of scores, unlike the paired-fold approach, and was found among the best-behaved classical comparison tests
B. It is the only test that can be applied when sample sizes exceed 10,000
C. It replaces the need for a held-out test set entirely
D. It directly estimates the cost-optimal decision threshold
E. It is equivalent to bootstrapping the ROC-AUC statistic

**43.** In a decision tree split search, a node holds 1,000 loans with 40 defaults (p = 0.04, Gini = 2p(1−p) = 0.0768). Splitting on "days_past_due > 7" sends 900 rows left (18 defaults, Gini = 0.0392) and 100 rows right (22 defaults, Gini = 0.3432). The improvement in impurity ΔI from this split is closest to:

A. 0.3432
B. 0.0072
C. 0.0768
D. 0.0696
E. 0.4200

**44.** A single decision tree is described as having low bias but high variance, because greedy hierarchical growth means a small perturbation to the training data can rewrite every decision beneath a flipped root split. Which pair of remedies does the chapter present as the two structural responses to this instability?

A. Regularisation and cross-validation
B. Bagging/Random Forests (parallel averaging over perturbed data) and boosting (sequential fitting of residual errors)
C. Standardisation and one-hot encoding
D. Increasing the learning rate and reducing tree depth simultaneously
E. Kernel methods and dimensionality reduction

**45.** Under the Condorcet jury theorem (ML form), M = 1,000 independent classifiers, each correct with probability p = 0.51, vote by majority. The probability the majority vote is correct is approximately:

A. 0.51
B. 0.99
C. 0.73
D. 0.50
E. 1.00

**46.** Per Proposition 4.2, the variance of an average of M estimators with common variance σ² and common pairwise correlation ρ is Var[f̄(x)] = ρσ² + (1−ρ)σ²/M. As M → ∞ while ρ stays fixed and positive, this variance:

A. Converges to zero regardless of ρ
B. Converges to ρσ², a floor that no amount of averaging can remove
C. Diverges to infinity
D. Converges to σ²/M, independent of ρ
E. Becomes negative, which is why ρ must be re-estimated

**47.** Bagging trains M base learners on bootstrap samples drawn with replacement, each of size n. As n → ∞, the probability that a given training row is left "out-of-bag" for a given tree converges to:

A. 1/2
B. 1/n
C. e⁻¹, approximately 36.8%
D. 1 − e⁻¹, approximately 63.2%
E. 0, since bootstrap sampling always includes every row at least once

**48.** A Random Forest restricts the split search at every node to a fresh random subset of k out of d features, in contrast to bagging alone. What problem does this specifically address?

A. It reduces the training set size to speed up fitting
B. It decorrelates the trees: without it, a dominant predictor is chosen near the root of nearly every tree, keeping the pairwise correlation ρ (and hence the variance floor) high
C. It converts the ensemble from a variance-reduction method into a bias-reduction method
D. It guarantees every tree in the forest is identical, improving reproducibility
E. It eliminates the need for out-of-bag evaluation

**49.** In AdaBoost, a weak learner achieves weighted error ε_m on its stage. Under the coefficient formula α_m = (1/2)ln[(1−ε_m)/ε_m], a learner with ε_m = 0.5 (a coin flip) receives:

A. α_m = 1, the maximum possible weight
B. α_m = 0, so it is effectively silenced in the final vote
C. A negative α_m, so the committee listens to the opposite of what it says
D. α_m = ∞, since the formula is undefined at ε_m = 0.5
E. α_m equal to the average of all prior learners' coefficients

**50.** LightGBM grows trees leaf-wise (best-first), splitting the leaf with the largest gain anywhere in the tree, while XGBoost's default grows level-wise. What is the direct consequence of LightGBM's strategy noted in the text?

A. Leaf-wise growth is always slower than level-wise growth on wide tables
B. For a fixed leaf budget, leaf-wise growth reaches lower training loss but correspondingly overfits faster, which is why num_leaves is capped directly rather than left uncontrolled
C. Leaf-wise growth cannot handle categorical features natively, unlike level-wise growth
D. Leaf-wise growth guarantees a shallower tree than level-wise growth for the same number of leaves
E. Leaf-wise growth eliminates the need for any learning-rate hyperparameter

---

## Answer Key — Paper 1

1. C — E and P already encode delayed labels and a cost-weighted, operations-bound metric; that is the point of the example.
2. B — D is unknown, so we substitute the empirical risk over a finite i.i.d. sample.
3. C — same input profile, different outcome, is concept drift (P(y|x) changed).
4. C — a short auditable rule meeting the bar is the textbook "prefer rules" case.
5. C — squared error is minimised by the mean; absolute error by the median.
6. B — accuracy weighs all errors equally and is misleading under 12x asymmetric costs.
7. A — the course states 80% of effort lives in framing and data (steps 1–3).
8. A — leakage is exactly a φ_train vs φ_serve distributional mismatch.
9. B — dropping the attribute does not remove proxy signal correlated with it.
10. C — t* = cFP/(cFP+cFN) = 3000/13000 ≈ 0.231.
11. C — missingness fully explained by an observed column is the definition of MAR.
12. D — the gap (4.1% vs 9.7%) is the MNAR screen; keep the indicator.
13. B — median is robust to the heavy right tail; a single whale distorts the mean.
14. B — a sentinel keeps the "absence" information distinct from the modal class.
15. B — Bayesian ridge is the default regressor; sample_posterior=True draws fills.
16. A — upper fence = 80 + 1.5(60) = 170; 185 exceeds it, so it is flagged.
17. B — scores near 1 mean short average isolation depth, i.e. an anomaly.
18. B — these are gaming behaviour, the signal the model is meant to catch, not noise.
19. B — unresolved entities scattered across splits let the model "recognise" test rows.
20. C — min-max scaling is redefined entirely by a single new max or min.
21. B — Yeo–Johnson extends Box–Cox to the full real line, including negatives.
22. A — K one-hot columns sum to 1, producing collinearity with an intercept.
23. B — hashing needs no fitted vocabulary, so unseen levels still map somewhere.
24. D — TE = (4×0.75 + 20×0.05)/(4+20) = (3+1)/24 ≈ 0.167.
25. A — the row's own label contaminates its own encoded value unless folds are held out.
26. B — a single trig coordinate maps two different times onto the same value.
27. B — WoE-coded logistic regression is exactly the traditional scorecard.
28. C — full-dataset scaling/imputation before splitting is cause #2/#3 on Huyen's list.
29. B — an immature label must be excluded, not assumed non-default.
30. C — e^(−0.35) ≈ 0.70, a 30% odds reduction.
31. B — sklearn parameterises C = 1/λ; larger C means weaker regularisation.
32. B — only support vectors (zero or positive hinge loss) determine the boundary.
33. B — kNN has no training cost but O(m) cost per prediction, the wrong profile for low-latency serving.
34. C — sqrt(0.7×0.3/150) ≈ sqrt(0.0014) ≈ 0.037.
35. B — unstratified folds can starve of positives at low prevalence, inflating variance.
36. A — inner loop tunes, outer loop gives an honest, untouched-fold estimate.
37. B — this is the exact statement of Proposition 3.2.
38. B — ROC's FPR denominator is swamped by negatives; PR-AUC is more informative under rarity.
39. B — this is the literal definition of calibration.
40. A — logistic regression is well calibrated by construction; SVMs/boosted trees are not.
41. B — SMOTE run before splitting leaks synthetic points derived from validation originals.
42. A — McNemar's test is distribution-light on discordant pairs and well-behaved per Dietterich.
43. B — 0.9(0.0392) + 0.1(0.3432) = 0.0696; ΔI = 0.0768 − 0.0696 = 0.0072.
44. B — bagging attacks variance in parallel; boosting attacks bias sequentially.
45. C — with mean 510, sd ≈15.8, P(majority correct) ≈ 0.73 as stated in the worked example.
46. B — the ρσ² term is a floor independent of M; only the second term shrinks with M.
47. C — (1 − 1/n)^n → e⁻¹ ≈ 0.368 as n → ∞.
48. B — random feature subsets at each split decorrelate trees that would otherwise share a dominant root split.
49. B — at ε_m = 0.5, ln(1)=0, so α_m = 0 and the learner is silenced.
50. B — leaf-wise growth reaches lower training loss per leaf budget but overfits faster, hence capping num_leaves.
