# Demonstration and viva guide

## Five-minute demonstration

1. Start the app with `start.bat`. State the project name and the three grade bands.
2. Explain that the demo uses real UCI mathematics data. Point out the 79 held-out students and baseline metric.
3. Enter `explain naive bayes` in the agent task box. Expand Agent activity to show tool selection and outcomes.
4. Open Predict a student, enter a profile and submit. Explain the band, all three probabilities and the evidence chart. The checklist is rule-based.
5. Open Batch predictions, enable the five example profiles, predict and download the CSV. Say that these profiles demonstrate the interface, not test accuracy.
6. Open Model insights. Explain one confusion-matrix cell, macro F1 and the training-only alpha selection.
7. Switch to Early warning. Discuss why its accuracy falls to 41.8%. Download the report.

## Questions to prepare

**What is the problem?**
Predict a final performance band from available student features and provide an interpretable review aid.

**Which algorithm did you implement?**
Categorical Naive Bayes using scikit-learn, with fixed categorical bins and additive smoothing.

**Why this Naive Bayes variant?**
Our inputs are discrete categories: yes/no fields, ordinal codes and binned grades/absences. Gaussian Naive Bayes instead assumes continuous Gaussian feature likelihoods.

**What does “naive” mean?**
Features are assumed conditionally independent given the target class. Real features can violate that assumption.

**What is learned?**
Class priors and category likelihoods. Cross-validation selects the smoothing strength. The controller and advice rules are programmed explicitly.

**What makes it an agent?**
It accepts goals, checks state, calls the tools needed to complete the goal, handles invalid inputs, reuses its model and logs the results. It is a bounded workflow agent, not an LLM chatbot.

**Why no Gemini or Groq?**
The project uses structured data and the required classifier is Naive Bayes. An LLM is not necessary to coordinate these tools and would add a network/API dependency.

**What are the inputs and output?**
Study time, failures, absences, travel time, support/access categories and optionally G1/G2. Output: Needs support, On track or Strong, derived from G3 grade ranges.

**How did you prevent target leakage?**
G3 is used only to create labels. Features are explicitly allowlisted. The test partition is not used to select alpha. Fixed bins do not estimate statistics from test rows.

**Are previous grades allowed?**
Only if known at prediction time. Prior-grade mode is for later predictions; early-warning mode removes G1/G2. Other cumulative features also need a timing check.

**What is the accuracy?**
On the bundled dataset and seed-42 holdout: 86.1% with previous grades; 41.8% without them. These figures are not claims about other student populations.

**Why macro F1?**
It balances precision and recall separately for each class, then gives each class equal weight. Accuracy alone can conceal weak minority-class performance.

**What does smoothing do?**
It adds a positive pseudo-count so an unseen feature category does not force a class likelihood to zero.

**How are missing values handled?**
Blank features get a dedicated unknown category. Missing targets and completely empty student records are rejected. The app flags missing-input predictions for review.

**Are the probabilities confidence guarantees?**
No. They are uncalibrated model probabilities, and correlated features can cause overconfidence.

**What does the explanation chart measure?**
The difference in each feature's log likelihood between the predicted class and the second-best class. Together with the prior term, these reconstruct the log posterior odds.

**Will following the plan improve the grade?**
The plan is a transparent checklist, not a causal prediction. Improvement would need separate intervention evidence.

**What would you improve next?**
Add external and temporal validation, probability calibration, fairness checks and more representative data before any real educational use.

## Code files to know

- `student_agent/data.py`: feature contract, grade targets, encoding.
- `student_agent/model.py`: classifier, split, CV, metrics, predictions, explanations.
- `student_agent/agent.py`: controller state and tool orchestration.
- `app.py` and `app_pages/`: interface and session handling.
- `tests/`: verification of behavior and data/model boundaries.

Read these files and reproduce the demonstration yourself before presenting the project.
