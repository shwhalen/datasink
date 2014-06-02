/*
    datasink: A Pipeline for Large-Scale Heterogeneous Ensemble Learning
    Copyright (C) 2013 Sean Whalen

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see [http://www.gnu.org/licenses/].
*/

import java.io.*
import java.text.*
import java.util.*
import java.util.zip.*

import weka.classifiers.*
import weka.classifiers.meta.*
import weka.core.*
import weka.core.converters.ConverterUtils.DataSource
import weka.filters.*
import weka.filters.supervised.instance.*
import weka.filters.unsupervised.attribute.*
import weka.filters.unsupervised.instance.*

void dump(instances, filename) {
    w = new BufferedWriter(new FileWriter(filename))
    w.write(instances.toString())
    w.write("\n")
    w.flush()
    w.close()
}

Instances balance(instances) {
    balanceFilter = new SpreadSubsample()
    balanceFilter.setDistributionSpread(1.0)
    balanceFilter.setInputFormat(instances)
    return Filter.useFilter(instances, balanceFilter)
}

// parse options
rootDir                     = args[0]
currentFold                 = args[1]
currentBag                  = Integer.valueOf(args[2])
String[] classifierString   = args[3..-1]
String classifierName       = classifierString[0]
String shortClassifierName  = classifierName.split("\\.")[-1]
String[] classifierOptions  = new String[0]
if (classifierString.length > 1) {
    classifierOptions = classifierString[1..-1]
}

// load data parameters from properties file
p = new Properties()
p.load(new FileInputStream(rootDir + "/weka.properties"))
inputFilename       = p.getProperty("inputFilename").trim()
workingDir          = rootDir + "/" + p.getProperty("workingDir", ".").trim()
idAttribute         = p.getProperty("idAttribute", "").trim()
classAttribute      = p.getProperty("classAttribute").trim()
balanceTraining     = Boolean.valueOf(p.getProperty("balanceTraining", "true"))
balanceTest         = Boolean.valueOf(p.getProperty("balanceTest", "false"))
assert p.containsKey("foldCount") || p.containsKey("foldAttribute")
if (p.containsKey("foldCount")) {
    foldCount       = Integer.valueOf(p.getProperty("foldCount"))
}
foldAttribute       = p.getProperty("foldAttribute", "").trim()
nestedFoldCount     = Integer.valueOf(p.getProperty("nestedFoldCount"))
bagCount            = Integer.valueOf(p.getProperty("bagCount"))
writeModel          = Boolean.valueOf(p.getProperty("writeModel", "false"))

// load data, determine if regression or classification
source              = new DataSource(rootDir + "/" + inputFilename)
data                = source.getDataSet()
regression          = data.attribute(classAttribute).isNumeric()
if (!regression) {
    predictClassValue = p.getProperty("predictClassValue").trim()
}

// shuffle data, set class variable
data.randomize(new Random(1))
data.setClass(data.attribute(classAttribute))
if (!regression) {
    predictClassIndex = data.attribute(classAttribute).indexOfValue(predictClassValue)
    assert predictClassIndex != -1
    printf "[%s] %s, generating probabilities for class %s (index %d)\n", shortClassifierName, data.attribute(classAttribute), predictClassValue, predictClassIndex
} else {
    printf "[%s] %s, generating predictions\n", shortClassifierName, data.attribute(classAttribute)
}

// add ids if not specified
if (idAttribute == "") {
    idAttribute = "ID"
    idFilter = new AddID()
    idFilter.setIDIndex("last")
    idFilter.setInputFormat(data)
    data = Filter.useFilter(data, idFilter)
}

// generate folds
if (foldAttribute != "") {
    foldCount = data.attribute(foldAttribute).numValues()
    foldAttributeIndex = String.valueOf(data.attribute(foldAttribute).index() + 1) // 1-indexed
    foldAttributeValueIndex = String.valueOf(data.attribute(foldAttribute).indexOfValue(currentFold) + 1) // 1-indexed
    printf "[%s] generating %s folds for leave-one-value-out CV\n", shortClassifierName, foldCount

    testFoldFilter = new RemoveWithValues()
    testFoldFilter.setModifyHeader(false)
    testFoldFilter.setAttributeIndex(foldAttributeIndex)
    testFoldFilter.setNominalIndices(foldAttributeValueIndex)
    testFoldFilter.setInvertSelection(true)
    testFoldFilter.setInputFormat(data)
    test = Filter.useFilter(data, testFoldFilter)

    trainingFoldFilter = new RemoveWithValues()
    trainingFoldFilter.setModifyHeader(false)
    trainingFoldFilter.setAttributeIndex(foldAttributeIndex)
    trainingFoldFilter.setNominalIndices(foldAttributeValueIndex)
    trainingFoldFilter.setInvertSelection(false)
    trainingFoldFilter.setInputFormat(data)
    train = Filter.useFilter(data, trainingFoldFilter)
} else {
    printf "[%s] generating folds for %s-fold CV\n", shortClassifierName, foldCount
    test = data.testCV(foldCount, Integer.valueOf(currentFold))
    train = data.trainCV(foldCount, Integer.valueOf(currentFold), new Random(1))
}

// resample and balance training fold if necessary
if (bagCount > 0) {
    printf "[%s] generating bag %d\n", shortClassifierName, currentBag
    train = train.resample(new Random(currentBag))
}
if (!regression && balanceTraining) {
    printf "[%s] balancing training samples\n", shortClassifierName
    train = balance(train)
}
if (!regression && balanceTest) {
    printf "[%s] balancing test samples\n", shortClassifierName
    test = balance(test)
}

// init filtered classifier
classifier = AbstractClassifier.forName(classifierName, classifierOptions)
removeFilter = new Remove()
if (foldAttribute != "") {
    removeIndices = new int[2]
    removeIndices[0] = data.attribute(foldAttribute).index()
    removeIndices[1] = data.attribute(idAttribute).index()
} else {
    removeIndices = new int[1]
    removeIndices[0] = data.attribute(idAttribute).index()
}
removeFilter.setAttributeIndicesArray(removeIndices)
filteredClassifier = new FilteredClassifier()
filteredClassifier.setClassifier(classifier)
filteredClassifier.setFilter(removeFilter)

// train, store duration
printf "[%s] fold: %s bag: %s training size: %d test size: %d\n", shortClassifierName, currentFold, (bagCount == 0) ? "none" : currentBag, train.numInstances(), test.numInstances()
start = System.currentTimeMillis()
filteredClassifier.buildClassifier(train)
duration = System.currentTimeMillis() - start
durationMinutes = duration / (1e3 * 60)
printf "[%s] trained in %.2f minutes, evaluating\n", shortClassifierName, durationMinutes

// write predictions to csv
classifierDir = new File(workingDir, classifierName)
if (!classifierDir.exists()) {
    classifierDir.mkdir()
}

outputPrefix = sprintf "predictions-%s-%02d", currentFold, currentBag
writer = new PrintWriter(new GZIPOutputStream(new FileOutputStream(new File(classifierDir, outputPrefix + ".csv.gz"))))
if (writeModel) {
    SerializationHelper.write(new GZIPOutputStream(new FileOutputStream(new File(classifierDir, outputPrefix + ".model.gz"))), filteredClassifier)
}
header = sprintf "# %s@%s %.2f minutes %s\n", System.getProperty("user.name"), java.net.InetAddress.getLocalHost().getHostName(), durationMinutes, classifierString.join(" ")
writer.write(header)
writer.write("id,label,prediction,fold,bag,classifier\n")
for (instance in test) {
    int id = instance.value(test.attribute(idAttribute))
    double prediction
    if (!regression) {
        label = (instance.stringValue(instance.classAttribute()).equals(predictClassValue)) ? 1 : 0
        prediction = filteredClassifier.distributionForInstance(instance)[predictClassIndex]
    } else {
        label = instance.classValue()
        prediction = filteredClassifier.distributionForInstance(instance)[0]
    }
    row = sprintf "%s,%s,%f,%s,%s,%s\n", id, label, prediction, currentFold, currentBag, shortClassifierName
    writer.write(row)
}
writer.flush()
writer.close()

if (nestedFoldCount == 0) {
    System.exit(0)
}

train = data.trainCV(foldCount, Integer.valueOf(currentFold), new Random(1))
printf "[%s] re-generated training data, starting %d-fold nested cv\n", shortClassifierName, nestedFoldCount
for (currentNestedFold in 0..nestedFoldCount - 1) {
    nestedTest = train.testCV(nestedFoldCount, currentNestedFold)
    nestedTrain = train.trainCV(nestedFoldCount, currentNestedFold, new Random(1))

    // resample and balance training fold if necessary
    if (bagCount > 0) {
        printf "[%s inner %s] generating bag %d\n", shortClassifierName, currentNestedFold, currentBag
        nestedTrain = nestedTrain.resample(new Random(currentBag))
    }
    if (!regression && balanceTraining) {
        printf "[%s inner %s] balancing training samples\n", shortClassifierName, currentNestedFold
        nestedTrain = balance(nestedTrain)
    }
    if (!regression && balanceTest) {
        printf "[%s inner %s] balancing test samples\n", shortClassifierName, currentNestedFold
        nestedTest = balance(nestedTest)
    }
    printf "[%s inner %s] fold: %s bag: %s training size: %d test size: %d\n", shortClassifierName, currentNestedFold, currentFold, (bagCount == 0) ? "none" : currentBag, nestedTrain.numInstances(), nestedTest.numInstances()

    start = System.currentTimeMillis()
    filteredClassifier.buildClassifier(nestedTrain)
    duration = System.currentTimeMillis() - start
    durationMinutes = duration / (1e3 * 60)
    printf "[%s inner %s] trained in %.2f minutes, evaluating\n", shortClassifierName, currentNestedFold, durationMinutes

    outputPrefix = sprintf "validation-%s-%02d-%02d", currentFold, currentNestedFold, currentBag
    writer = new PrintWriter(new GZIPOutputStream(new FileOutputStream(new File(classifierDir, outputPrefix + ".csv.gz"))))
    header = sprintf "# %s@%s %.2f minutes %s\n", System.getProperty("user.name"), java.net.InetAddress.getLocalHost().getHostName(), durationMinutes, classifierString.join(" ")
    writer.write(header)
    writer.write("id,label,prediction,fold,nested_fold,bag,classifier\n")
    for (instance in nestedTest) {
        int id = instance.value(nestedTest.attribute(idAttribute))
        double prediction
        if (!regression) {
            label = (instance.stringValue(instance.classAttribute()).equals(predictClassValue)) ? 1 : 0
            prediction = filteredClassifier.distributionForInstance(instance)[predictClassIndex]
        } else {
            label = instance.classValue()
            prediction = filteredClassifier.distributionForInstance(instance)[0]
        }
        row = sprintf "%s,%s,%f,%s,%s,%s,%s\n", id, label, prediction, currentFold, currentNestedFold, currentBag, shortClassifierName
        writer.write(row)
    }
    writer.flush()
    writer.close()
}
