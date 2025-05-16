# Nepali-Hate-Sentiment-Detection

[[Live Demo]](https://demo-for-nephased.vercel.app/) | [[Python Package]](https://github.com/angeltamang123/nephased)

- Prototyping across classical ML models, Deep Neural Network(DNN) and transformers.
- Models trained on NepSa dataset with 2835 General, 407 Profanity and 287 Violence class instances. A thorough deduplication process was utilized as the dataset contained duplicates(due to it's creation for Aspect Term Extraction) while following the NepSa annotation guidelines resulting in: 2040 General, 333 Profanity, and 267 Violence instances.
- Nepali-Stemmer was used to further clean our data after punctuation removal.
- Neural Network based embedders: Word2Vec and Fasttext, and TF-IDF were used to experiment on static embeddings. The result were underwhelming with Word2Vec trained SVM achieving 33% F1-score
- Transformer architectures were introduced: SONAR(Sentence-Level Multimodal and Language Agnostic Representations)'s encoder, SONAR being trained for Machine Translation, was used to generate static transformer embeddings to train a DNN. While distil-bert-nepali, available on huggingface, was finetuned on our dataset.
- To address the imbalanced nature of the dataset, cost sensitive learning was employed where frequency based class weights are supplied to the loss function penalizing miss-classification of minorities to help models focus on the minorities. This alone improved performance as the base SONAR classifier improved from 31% F1-score to 43%, whereas for finetuned-distilBERT the improvement was from 50% F1-score to 55%.
- Hyperparameter tuning was done using frameworks like raytune for DNN trained using torch, using Optuna algorithm for DNN and distilBERT. While for the classical model random search was used.To further optimize the weights sent to the loss function- a genetic based algorithm i.e. Differential Evolution was introduced to train SONAR and finetune distilBERT across different population members and genetic evolution.
- The final model is the finetuned distilbert model achieving 63% F1 score, while other metrics like Accuracy, Precision, Recall, Precision-Recall and Loss curve was also used
- The above mentioned prototype is the 1st version of Nephased( Live Demo linked ). The demo is possible by hosting the model on Huggingface gradio spaces, and Nextjs-flask app
- The 2nd prototypes will be trained on NepSa dataset extended by team, naming it NepSa++. The Dataset is extended by following the annotation guidelines, where kirpen-dorff's alpha to check agreement between annotators is calculated to be 92.44%. The early test on static embeddings using fasttext trained on the extended dataset show promising signs with 73% F1-score. Stay Tuned!!!
