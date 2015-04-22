/**
 * Copyright 2013 Brigham Young University
 * 
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */
package edu.byu.nlp.crowdsourcing.app;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Function;

import edu.byu.nlp.classify.NaiveBayesClassifier;
import edu.byu.nlp.classify.NaiveBayesLearner;
import edu.byu.nlp.crowdsourcing.TrainableMultiAnnModel;
import edu.byu.nlp.data.docs.CountCutoffFeatureSelectorFactory;
import edu.byu.nlp.data.docs.DocPipes.Doc2FeaturesMethod;
import edu.byu.nlp.data.docs.DocumentDatasetBuilder;
import edu.byu.nlp.data.docs.TokenizerPipes;
import edu.byu.nlp.data.pipes.EmailHeaderStripper;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.DatasetInstance;
import edu.byu.nlp.dataset.Datasets;
import edu.byu.nlp.util.jargparser.ArgumentParser;
import edu.byu.nlp.util.jargparser.annotations.Option;

/**
 * @author rah67
 * 
 */
public class Baseline {

  private static final Logger logger = LoggerFactory.getLogger(TrainableMultiAnnModel.class);

  @Option(help = "base directory of the documents")
  private static String basedir = "20_newsgroups";

  @Option
  private static String dataset = "tiny_set";

  @Option
  private static String split = "all";

  private enum DatasetType{NEWSGROUPS, REUTERS, ENRON, NB2, NB20}
  
  @Option(help = "base directory of the documents")
  private static DatasetType datasetType = DatasetType.NEWSGROUPS;

  @Option(help = "any features that don't appear more than this are discarded")
  private static int featureCountCutoff = 1;
  
  @Option
  private static long seed = System.nanoTime();

  @Option
  private static double splitPercent = 85;

  @Option
  private static String malletTrain = null;

  @Option
  private static String malletTest = null;
  
  public static void main(String[] args) throws IOException {
    new ArgumentParser(Baseline.class).parseArgs(args);
    RandomGenerator rnd = new MersenneTwister(seed);
    
    // Data
    Dataset trainingData;
    Dataset heldoutData;
    // use an existing split
    if (exists(malletTrain) && exists(malletTest)){
      trainingData = Datasets.readMallet2Labeled(malletTrain);
      heldoutData = Datasets.readMallet2Labeled(malletTest, 
          trainingData.getInfo().getLabelIndexer(), trainingData.getInfo().getFeatureIndexer(),
          trainingData.getInfo().getInstanceIdIndexer(), trainingData.getInfo().getAnnotatorIdIndexer());
    }
    // create a new split
    else{
      // Read and split the data
      Dataset fullData = readData(rnd);
      List<Dataset> partitions = Datasets.split(fullData, new double[]{splitPercent, 100-splitPercent});
      trainingData = partitions.get(0);
      heldoutData = partitions.get(1);
      
      // record the experimentData
      if (malletTrain!=null && !malletTrain.isEmpty()){
        Datasets.writeLabeled2Mallet(trainingData, malletTrain);
      }
      if (malletTest!=null && !malletTest.isEmpty()){
        Datasets.writeLabeled2Mallet(heldoutData, malletTest);
      }
    }
    
    
    // Train the model
    NaiveBayesClassifier model = new NaiveBayesLearner().learnFrom(trainingData);
    /*
    // Print out the data to eyeball the feature set.
    for (DatasetInstance instance : fullData.labeledData()) {
      StringBuilder str = new StringBuilder();
      for (Entry e : instance.getData().sparseEntries()) {
        str.append(fullData.getWordIndex().get(e.getIndex()));
        str.append(" ");
      }
      System.out.println(str.toString());
    }
    */

    // Compute accuracy
    System.out.println("Accuracy: " + computeAccuracy(model, heldoutData));
  }

//  private static Dataset convert(Dataset dataset) {
//    return new Dataset(dataset.labeledData(),
//      Collections.<DatasetInstance>emptyList(), dataset.getNumLabels(),
//      dataset.getNumFeatures(), dataset.labeledData().size());
//  }
  
  private static double computeAccuracy(NaiveBayesClassifier model, Dataset heldoutData) {
    Dataset labeledHeldoutData = Datasets.divideInstancesWithObservedLabels(heldoutData).getFirst();
    int correct = 0;
    for (DatasetInstance instance : labeledHeldoutData) {
      if (model.given(instance.asFeatureVector()).argMax() == instance.getLabel()) {
        ++correct;
      }
    }
    return (double) correct / labeledHeldoutData.getInfo().getNumDocuments();
  }

//  private static Dataset readData(RandomGenerator rnd) {
//    DocumentDatasetBuilder newsgroups =
//        new DocumentDatasetBuilder(basedir, dataset, split, new HeaderStripper(),
//            TokenizerPipes.McCallumAndNigam(), new CountCutoffFeatureSelectorFactory<String>(1));
//    Dataset data = newsgroups.dataset();
//
//    logger.info("Number of instances = " + data.labeledData().size());
//    logger.info("Number of tokens = " + data.getNumTokens());
//    logger.info("Number of features = " + data.getNumFeatures());
//    logger.info("Number of classes = " + data.getNumLabels());
//    
//    data.shuffle(rnd);
//    data = data.copy();
//    return data;
//  }

  private static Dataset readData(RandomGenerator rnd) throws IOException {
    Function<String, String> docTransform = null;
    switch(datasetType){
    case NB2:
    case NB20:
    case ENRON:
      break;
    case NEWSGROUPS:
    case REUTERS:
      docTransform = new EmailHeaderStripper();
      break;
    default:
      throw new IllegalStateException("unknown dataset type: " + datasetType);
    }

    Function<List<String>, List<String>> tokenTransform = null; // FIXME
    Dataset data =
        new DocumentDatasetBuilder(basedir, dataset, split, docTransform,
            TokenizerPipes.McCallumAndNigam(), tokenTransform, Doc2FeaturesMethod.WORD_COUNTS, new CountCutoffFeatureSelectorFactory<String>(featureCountCutoff))
            .dataset();
    data.shuffle(rnd);

    logger.info("Number of instances = " + data.getInfo().getNumDocuments());
    logger.info("Number of tokens = " + data.getInfo().getNumTokens());
    logger.info("Number of features = " + data.getInfo().getNumFeatures());
    logger.info("Number of classes = " + data.getInfo().getNumClasses());

    return data;
  }
  
  private static boolean exists(String path){
    return path!=null && !path.isEmpty() && Files.exists(Paths.get(path));
  }
}
