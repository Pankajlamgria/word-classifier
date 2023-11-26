from sklearn import svm
import spacy
import os
import json
trainx=[]
trainy=[]
testx=[]
testy=[]

for file in os.listdir('./pycon2020/data/training'):
    category=file.strip('train_').split('.')[0]
    with open(f"./pycon2020/data/training/{file}")as f:
        for line in f:
            jsondata=json.loads(line)
            trainx.append(jsondata['reviewText'])
            trainy.append(category)
# trainx=trainx[0:4]
# trainy[0]="bird"
# trainy=trainy[0:4]

nlp=spacy.load('en_trf_bertbaseuncased_lg')
text=[]
for i in trainx:
    text.append(nlp(i).vector)

clf=svm.SVC(kernel="linear")
clf.fit(text,trainy)



# to testing
for file in os.listdir("./pycon2020/data/test"):
    category=file.strip('test_').split('.')[0]
    with open(f"./pycon2020/data/test/{file}")as f:
        for line in f:
            jsondata=json.loads(line)
            testx.append(jsondata['reviewText'])
            testy.append(category)
            
            

# testx=["check this story out"]
# testx=testx[0:4]
# testy[0]="beard"
# testy=testy[0:4]
docs=[nlp(test) for test in testx]
testx_vector=[doc.vector for doc in docs]
print(clf.score(testx_vector,testy))