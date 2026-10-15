import streamlit as st

st.title("Your AI lab, explained")
st.markdown("**Student Performance Prediction using Naive Bayes**")
st.markdown("The project predicts a student's final grade band and demonstrates an agent that coordinates validation, learning, evaluation, prediction, and explanation.")
with st.container(border=True):
    st.subheader("What makes this an agent?")
    st.markdown("The controller accepts a goal, observes its state, selects the required tools, executes them, and records the result. If prediction is requested before training, it trains first. If the data is invalid, it stops with a correction message. If an evaluated model exists for the same dataset and mode, it reuses it.")
    st.markdown("Its planning policy is a **deterministic workflow**. The predictive intelligence comes from a **learned Naive Bayes model**. It does not use an LLM, autonomous web browsing, or open-ended reasoning.")
st.subheader("Naive Bayes in one equation")
st.latex(r"P(c \mid x_1,\ldots,x_n) \propto P(c)\prod_{i=1}^{n}P(x_i\mid c)")
st.markdown("The prior tells us how common a performance band is. Each likelihood tells us how common a feature value is within that band. The classifier combines them and chooses the band with the highest posterior probability. The ‘naive’ assumption is conditional independence of features given the class.")
st.markdown("We use **Categorical Naive Bayes** because study-time categories, yes/no support information, and fixed grade/absence bins have discrete values. Additive smoothing keeps unseen categories from having zero likelihood. Missing feature values have a dedicated category.")
st.subheader("From CSV to support plan")
st.markdown("1. Read a UTF-8 CSV and validate the required fields.\n2. Remove exact full-row duplicates; reject missing targets and invalid values.\n3. Derive the target from G3 and encode only approved predictors.\n4. Make a stratified 80/20 train/test split with seed 42.\n5. Compare alpha 0.5, 1, and 2 with five-fold cross-validation on training rows.\n6. Fit the selected model on training rows and evaluate the held-out test rows.\n7. Predict, calculate feature evidence, and apply transparent study-support rules.")
st.subheader("A five-minute demonstration")
st.markdown("1. Open Overview and explain the three grade bands.\n2. Ask the agent to ‘explain naive bayes’ and inspect Agent activity.\n3. Predict one student and explain the class probabilities.\n4. Open Batch predictions, select the five example profiles, and download results.\n5. Show the confusion matrix and export the evaluation report.\n6. Switch to Early warning and discuss the change in performance.")
with st.expander("Viva questions", expanded=True):
    st.markdown("**Why not use an LLM?** This is structured-data classification. Naive Bayes is the required learning algorithm, and the controller can select tools with explicit rules.\n\n**Why exclude G3?** It defines the target. Including it as an input would reveal the answer.\n\n**Are G1 and G2 leakage?** They are usable only if their grades are already known at prediction time. The app offers a separate mode without them.\n\n**Why macro F1?** Accuracy can hide poor performance on a smaller class. Macro F1 gives each class equal weight.\n\n**Can features really be independent?** Often they are not. For example G1 and G2 are correlated; that can make the probability estimates overconfident.\n\n**Does a study plan guarantee improvement?** No. The plan is a rule-based checklist; the model has not learned causal intervention effects.")
st.subheader("Data and scope")
st.markdown("The bundled demo uses **395 mathematics records** from the [UCI Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student+performance), credited to Paulo Cortez (2008), [DOI: 10.24432/C5TG7T](https://doi.org/10.24432/C5TG7T), under **CC BY 4.0**. The Portuguese dataset is supplied in the source archive but is not merged, because some students appear in both subjects.")
st.markdown("The selected features omit gender, age, family occupations, and health information. That does not establish fairness: access and support variables may still reflect social differences. This is an educational demonstration for teacher review, not an admissions or grading system.")
st.link_button("Naive Bayes reference", "https://scikit-learn.org/stable/modules/naive_bayes.html", icon=":material/open_in_new:")
