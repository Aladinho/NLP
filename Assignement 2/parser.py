from tabulate import tabulate

class Parser:

    # Init my stack, my buffer, buffer temp and table of features
    # Call the method loadFeat to fill my table of features
    # Call the method inputParser to take sentences from the input
    # Call the method dependency_parse -> algo from the course
    def __init__(self,fileN=None):
        featTable = {}
        cpt = 0
        featTable = self.loadFeat(featTable)
        dataLoad = self.inputParser(fileN)
        conftableTxt = open("conftable.txt", 'w+')
        for j in range(4):
            myStack = []
            myBuffer = []
            bufferTemp = []
            textToWrite=""
            tabText = []
            for i in dataLoad[j] :
                myBuffer.append(i)
                i.append(0)
                bufferTemp.append(i)
                tabText.append(i[1])
            for k in tabText:
                textToWrite = textToWrite+" "+str(k)
            cpt = cpt + 1
            conftableTxt.write("<sentence file='input.txt' id='" +str(cpt) +"' text='"+textToWrite+"'>"+"\n")
            bufferTemp = self.dependency_parse(myStack,myBuffer,featTable,bufferTemp,conftableTxt)
            conftableTxt.write("\n")
            self.generateOutputConnLLU(bufferTemp)
        conftableTxt.close()

    # Method to load the features from the feattemp.txt
    # Param : table of features that we need to fill
    # Return : table of features filled with the file
    def loadFeat(self, featTable):
        with open("feattemp.txt") as f:
            for line in f:
                (key,val) = line.split()
                featTable[key] = val
        return featTable

    # Method to parse the input into sentences
    # Param : file in input
    # Return : Table of sentences
    def inputParser(self, filename):
        with open(filename) as f:
            input = [line.split() for line in f]
        allSentences = []
        s = []
        for i in input:
            s.append(i)
            if len(i) == 0 :
                s.pop()
                allSentences.append(s)
                s = []
        return allSentences

    # Method that represents the algo seen in class (see the report Discussion part 2)
    # Param : the stack, the buffer, table of featrues, temporary buffer, output file conftable
    # Return : the temporary buffer
    def dependency_parse(self, myStack, myBuffer, features, tmpBuffer, conftableParam):
        step = 0
        s1 = ["ROOT","ROOT","ROOT","ROOT"]
        s2 = ["None","None","None","None"]
        myStack.append(s1)
        while len(myBuffer) != 0 or len(myStack) != 1:
            if len(myBuffer) > 0:
                b1 = myBuffer[0][3]
            else:
                b1 = "None"
            tmp = s1[3] + "_" + s2[3] + "_" + b1
            if tmp in features :
                action = features[tmp]
                string = str(step) + "->" + str([elementStack[1] for elementStack in myStack]) + "->" + str([elementBuffer[1] for elementBuffer in myBuffer]) + "->" + action
                relation = self.apply(action, myStack, myBuffer, tmpBuffer)
                conftableParam.write(string + relation + "\n")
                s1 = myStack[-1]
                if len(myStack) > 1:
                    s2 = myStack[-2]
                step = step + 1
        endString = str(step)+ "->" +"['ROOT']->[]->Done"+"\n"+"</sentence>"+"\n"
        conftableParam.write(endString)
        return tmpBuffer

    # Method to apply the action from the table of features
    # Param : action, the stack, the buffer and the temporary buffer
    # Return : The relation between s1 and s2
    def apply(self, action, myStack, bufferParam, bufferTempParam):
        if action == "SHIFT":
            myStack.append(bufferParam[0])
            bufferParam.pop(0)
            relation = " "
        elif action == "LEFTARC":
            tmp = int(myStack[-2][0])
            bufferTempParam[tmp - 1][-1] = myStack[-1][0]
            relation = "->("+str(myStack[-2][1]) + "<-" + str(myStack[-1][1]) + ")"
            myStack.pop(-2)
        else:
            if len(bufferParam) == 0 and len(myStack) == 2:
                tmp = int(myStack[-1][0])
                bufferTempParam[tmp - 1][-1] = 0
            else:
                myStack[-1][-1] = myStack[-2][0]
                tmp = int(myStack[-2][0])
                bufferTempParam[tmp - 1][-1] = myStack[-2][0]
            relation = "->("+str(myStack[-2][1]) + "->" + str(myStack[-1][1]) + ")"
            myStack.pop()
        return relation

    # Method to generate the output file
    # Param : The temporary buffer
    def generateOutputConnLLU(self, bufferTempParam):
        for i in bufferTempParam:
            i.append("DEP")
        outputFile = open("output.txt",'a')
        outputFile.write(tabulate(bufferTempParam, headers=['ID ', 'FORM ', 'LEMMA ', 'UPOSTAG ', 'HEAD ', 'DEPREL '], tablefmt='orgtbl'))
        outputFile.write("\n --------------------------------------------------------------- \n")
        outputFile.close()

# Main method
if __name__ == "__main__":
   filen = "input.txt"
   a = Parser(filen)
