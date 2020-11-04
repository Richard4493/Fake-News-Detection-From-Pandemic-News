Fake data detection from pandemic data  using twitter.

Description: Collect data from twitter about covid. Train the machine to detect fake news using a dataset using the textual features of news.
            Using the trained model to classify fake news and real news from twitter.
            
Methodology Overview: 1.Using twitter api we collect data from twitter of about 100 tweets about covid19 and stores it into a csv file.
                      2.Using a covid news dataset we compare different classification algorithms and use the best one to train the model
                      (uses sklearn library)
                      3. Analysing the textual features of fake news and using them while training to increase accuracy of the model.  

Team Members:
Richard T S - CSE-B Roll.No:44 ,,
Sabarinath R- CSE-B Roll.No:48 ,,
Sabir Ibrahim-CSE-B Roll.No:49


Dataset and code idea:
https://towardsdatascience.com/covid-fake-news-detection-with-a-very-simple-logistic-regression-34c63502e33b

Results:
1. Collected datails from about covid19news and stored it in a csv file.
2. Compared logistic regression,decision tree and pac : logistic regression gives highest accuracy.

Instructions:
To run the code download and place the dataset (csv file) in the code directory. Clone and run the code after installing the required packages(sklearn,nltk,pandas).

Status:  https://drive.google.com/drive/folders/1ZprcsdATdCBEgVGulOOqqyX1_d5tZhpb?usp=sharing