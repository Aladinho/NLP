from tabulate import tabulate
import nltk
from nltk.classify.naivebayes import NaiveBayesClassifier
from nltk.sentiment.util import mark_negation
from nltk.tokenize import wordpunct_tokenize
from nltk import pos_tag
from nltk.corpus import sentiwordnet as swn
from nltk.wsd import lesk
from nltk.metrics.scores import precision, recall, f_measure
import string
import collections
import nltk
import json


# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('sentiwordnet')
# nltk.download('opinion_lexicon')

def parse(filename):

    with open(filename) as myFile:
        for l in myFile:
            yield eval(l)


def parseJSON(file):
    dataWithoutNeg = []
    dataWithNeg = []
    cptL = 0

    for reviews in parse(file):
        cptL += 1
        classi = classificationOverall(reviews["overall"])
        tokenizeReviewText = nltk.word_tokenize(reviews["reviewText"])
        tokenLow = [w.lower() for w in tokenizeReviewText]
        dataWithoutNeg.append((tokenLow, classi))

        tokenNeg = mark_negation(tokenLow)
        tokenFinal = [w for w in tokenNeg if w not in list(string.punctuation)]
        dataWithNeg.append((word_feats(tokenFinal), classi))

    return dataWithoutNeg, dataWithNeg


def classificationOverall(overall):
    if (0 <= overall and overall <= 2.0):
        return "neg"
    elif (2.1 <= overall and overall <= 3.0):
        return "neutre"
    elif (3.1 <= overall and overall <= 5.0):
        return "pos"

def word_feats(words):
    return dict([(word, True) for word in words])

def modelUnigram(trainData, testData):

    print("--MODEL UNIGRAM--")
    tab = []
    classifier = NaiveBayesClassifier.train(trainData)
    realSet = collections.defaultdict(set)
    testSet = collections.defaultdict(set)
    tabOut = []
    tabOver = []

    for i, (wordFeat, overall) in enumerate(testData):
        realSet[overall].add(i)
        predicted = classifier.classify(wordFeat)
        tabOut.append(predicted)
        tabOver.append(overall)
        tab.append(predicted)
        testSet[predicted].add(i)

    print("Accuracy Naive Bayes for Unigram Model : ", nltk.classify.util.accuracy(classifier, testData))

    return realSet, testSet, tab, tabOut, tabOver

def tagCount(data, tagData, negationData, pos_score, neg_score, obj_score, booleanNeg):
    cpt = 0
    for w, tag in tagData:

        if 'NN' in tag:
            if negationData[cpt][-4:] == '_NEG':
                booleanNeg = True
            sentiSynset = str(lesk(data[0], w, 'n'))[8:-2]
            if len(sentiSynset) > 0:
                if booleanNeg:
                    pos_score += swn.senti_synset(sentiSynset).neg_score()
                    neg_score += swn.senti_synset(sentiSynset).pos_score()
                    obj_score += swn.senti_synset(sentiSynset).obj_score()
                else:
                    pos_score += swn.senti_synset(sentiSynset).pos_score()
                    neg_score += swn.senti_synset(sentiSynset).neg_score()
                    obj_score += swn.senti_synset(sentiSynset).obj_score()
        elif 'VB' in tag:
            if negationData[cpt][-4:] == '_NEG':
                booleanNeg = True
            sentiSynset = str(lesk(data[0], w, 'v'))[8:-2]
            if len(sentiSynset) > 0:
                if booleanNeg:
                    pos_score += swn.senti_synset(sentiSynset).neg_score()
                    neg_score += swn.senti_synset(sentiSynset).pos_score()
                    obj_score += swn.senti_synset(sentiSynset).obj_score()
                else:
                    pos_score += swn.senti_synset(sentiSynset).pos_score()
                    neg_score += swn.senti_synset(sentiSynset).neg_score()
                    obj_score += swn.senti_synset(sentiSynset).obj_score()
        elif 'JJ' in tag:
            if negationData[cpt][-4:] == '_NEG':
                booleanNeg = True
            sentiSynset = str(lesk(data[0], w, 'a'))[8:-2]
            if len(sentiSynset) > 0:
                if booleanNeg:
                    pos_score += swn.senti_synset(sentiSynset).neg_score()
                    neg_score += swn.senti_synset(sentiSynset).pos_score()
                    obj_score += swn.senti_synset(sentiSynset).obj_score()
                else:
                    pos_score += swn.senti_synset(sentiSynset).pos_score()
                    neg_score += swn.senti_synset(sentiSynset).neg_score()
                    obj_score += swn.senti_synset(sentiSynset).obj_score()
            else:
                sentiSynset = str(lesk(data[0], w, 's'))[8:-2]
                if len(sentiSynset) > 0:
                    if booleanNeg:
                        pos_score += swn.senti_synset(sentiSynset).neg_score()
                        neg_score += swn.senti_synset(sentiSynset).pos_score()
                        obj_score += swn.senti_synset(sentiSynset).obj_score()
                    else:
                        pos_score += swn.senti_synset(sentiSynset).pos_score()
                        neg_score += swn.senti_synset(sentiSynset).neg_score()
                        obj_score += swn.senti_synset(sentiSynset).obj_score()
        elif 'RB' in tag:
            if negationData[cpt][-4:] == '_NEG':
                booleanNeg = True
            sentiSynset = str(lesk(data[0], w, 'r'))[8:-2]
            if len(sentiSynset) > 0:
                if booleanNeg:
                    pos_score += swn.senti_synset(sentiSynset).neg_score()
                    neg_score += swn.senti_synset(sentiSynset).pos_score()
                    obj_score += swn.senti_synset(sentiSynset).obj_score()
                else:
                    pos_score += swn.senti_synset(sentiSynset).pos_score()
                    neg_score += swn.senti_synset(sentiSynset).neg_score()
                    obj_score += swn.senti_synset(sentiSynset).obj_score()

        cpt += 1
    return pos_score,neg_score,obj_score

def lexicon_feature(tabScore):
    for score in tabScore:
        return dict({'positive': score[0],'negative': score[1],'neutral': score[2]})

def modelTrainingLexicon(traginingData, testData):
    print("--Lexicon Model--")
    tab = []
    dataLexiconFeature = []
    dataLexiconFeatureT = []
    for data in traginingData:
        booleanNeg = False
        pos_score = neg_score = obj_score = 0
        tagData = pos_tag(data[0])
        negationData = mark_negation(data[0])
        pos_score, neg_score, obj_score =tagCount(data,tagData,negationData,pos_score,neg_score,obj_score,booleanNeg)
        total = int(pos_score) - int(neg_score)
        if (total < 0):
            overall = 'neg'
        elif (total > 0):
            overall = 'pos'
        elif (total == 0):
            overall = 'neutre'
        tab.append(pos_score)
        tab.append(neg_score)
        tab.append(obj_score)
        feats = ({'positive': pos_score, 'negative': neg_score}, data[1])
        dataLexiconFeature.append(feats)

    for dataT in testData:
        booleanNegT = False
        pos_scoreT = neg_scoreT = obj_scoreT = 0
        tagData = pos_tag(dataT[0])
        negationDataT = mark_negation(dataT[0])
        pos_scoreT, neg_scoreT, obj_score = tagCount(dataT, tagData, negationDataT, pos_scoreT, neg_scoreT, obj_scoreT,
                                                   booleanNegT)
        total = int(pos_scoreT) - int(neg_scoreT)

        tab.append(pos_scoreT)
        tab.append(neg_scoreT)
        tab.append(obj_scoreT)
        featsT = ({'positive': pos_scoreT, 'negative': neg_scoreT}, dataT[1])
        dataLexiconFeatureT.append(featsT)


    classifier = NaiveBayesClassifier.train(dataLexiconFeature)
    realSet = collections.defaultdict(set)
    testSet = collections.defaultdict(set)

    tabPr = []
    tabOut = []

    for i, (feat, ovAll) in enumerate(dataLexiconFeatureT):
        realSet[ovAll].add(i)
        predicted = classifier.classify(feat)
        tabOut.append(predicted)
        tabPr.append(predicted)
        testSet[predicted].add(i)


    print("Accuracy Naive Bayes for Lexicon Model : ", nltk.classify.util.accuracy(classifier, dataLexiconFeatureT))

    return realSet, testSet, tabPr, tabOut

def printEval(realSet, testSet):

    precisionPos = precision(realSet['pos'], testSet['pos'])
    precisionNeg = precision(realSet['neg'], testSet['neg'])
    precisionNeutre = precision(realSet['neutre'], testSet['neutre'])


    recallPos = recall(realSet['pos'], testSet['pos'])
    recallNeg = recall(realSet['neg'], testSet['neg'])


    fmesurePos = f_measure(realSet['pos'], testSet['pos'])
    fmesureNeg = f_measure(realSet['neg'], testSet['neg'])


    # print("Precision    Pos: " + precisionPos + " - Neg: " + float(precisionNeg)
    # # print("Recall   Pos: %f - Neg: %f - Neutral: %f" %(recallPos, recallNeg, recallNeutre))
    # # print("F-Mesure Pos: %f - Neg: %f - Neutral: %f" %(fmesurePos, fmesureNeg, fmesureNeutre))

    print("Precision    Pos: %f - Neg: %f " %(float(precisionPos), float(precisionNeg)))
    print("Recall   Pos: %f - Neg: %f " %(float(recallPos), float(recallNeg)))
    print("F-Mesure Pos: %f - Neg: %f " %(float(fmesurePos), float(fmesureNeg)))

    # print("Pos : ",realSet['pos'], testSet['pos'])
    # print("Neg : ",realSet['neg'], testSet['neg'])
    # print("Neutre: ",realSet['neutre'], testSet['neutre'])


if __name__ == "__main__":
    trainingSet, trainingSetWithNeg = parseJSON('Video_Games_train.json')
    testSet, testSetWithNeg = parseJSON('test_games.json')
    # Unigram Wordfeat
    realSetNB, testSetNB, tabModelNB, out, tabOver = modelUnigram(trainingSetWithNeg, testSetWithNeg)
    printEval(realSetNB, testSetNB)
    # Lexicon
    realSetLexicon, testSetLexicon, tabPreLexicon, out2 = modelTrainingLexicon(trainingSet,testSet)
    printEval(realSetLexicon, testSetLexicon)

    tabOut1 = []
    tabOut2 = []
    outputFile= open("testout.txt", 'w')
    outputFile.close()
    outputFile = open("testout.txt", 'w')
    outputFile.write("\n ---------------------------------- \n")
    outputFile.write("< Line , obs , pre1 , pre2 >\n")
    cpt=1
    for i in range(len(out)):
        str="< %d" %cpt+ "    , "+tabOver[i]+" , "+out[i]+"  , "+out2[i]+"  >\n"
        cpt += 1
        outputFile.write(str)
    outputFile.write("\n---------------------------------- \n")
    outputFile.close()









