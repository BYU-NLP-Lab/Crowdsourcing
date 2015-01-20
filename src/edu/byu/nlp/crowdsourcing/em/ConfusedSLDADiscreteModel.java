/**
 * Copyright 2014 Brigham Young University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package edu.byu.nlp.crowdsourcing.em;

import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

import org.apache.commons.math3.random.RandomGenerator;

import cc.mallet.classify.MaxEnt;
import cc.mallet.classify.MaxEntTrainer;
import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.Label;
import cc.mallet.types.LabelAlphabet;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;

import edu.byu.nlp.classify.eval.BasicPrediction;
import edu.byu.nlp.classify.eval.Prediction;
import edu.byu.nlp.classify.eval.Predictions;
import edu.byu.nlp.classify.util.ModelTraining;
import edu.byu.nlp.classify.util.ModelTraining.SupportsTrainingOperations;
import edu.byu.nlp.crowdsourcing.CrowdsourcingUtils;
import edu.byu.nlp.crowdsourcing.ModelInitialization.AssignmentInitializer;
import edu.byu.nlp.crowdsourcing.PriorSpecification;
import edu.byu.nlp.crowdsourcing.gibbs.BlockCollapsedMultiAnnModelMath;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.DatasetInstance;
import edu.byu.nlp.dataset.Datasets;
import edu.byu.nlp.math.GammaFunctions;
import edu.byu.nlp.stats.RandomGenerators;
import edu.byu.nlp.util.DoubleArrays;
import edu.byu.nlp.util.IntArrays;
import edu.byu.nlp.util.Integers;
import edu.byu.nlp.util.Matrices;

/**
 * @author plf1
 *
 * For more info on this model, see notes in 
 * https://drive.google.com/drive/u/0/#folders/0B5phubFg2ZvVSDRvS0U1S3pScjQ/0B5phubFg2ZvVNWtfMEN1b0NMWlk
 */
public class ConfusedSLDADiscreteModel {
  private static final Logger logger = Logger.getLogger(ConfusedSLDADiscreteModel.class.getName());
  private static final boolean DEBUG = true;
  
  //////////////////////////////////////////////
  // Helper Code
  //////////////////////////////////////////////
  /**
   * this tracks the state of the sampler. It should 
   * be sufficient to save/restore the model at any point.
   */
  public static class State{
    
    // debugging flags
    private boolean includeMetadataSupervision = false; // if false, reduces to unsupervised LDA
    
    private Dataset data;
    private PriorSpecification priors;
    private double[][][] deltas; // a function of the priors object
    private int numTopics; // T: num topics. 
    private int numClasses; // K: num classes. (Derived from data)
    private int numDocuments; // D: num documents. (Derived from data)
    private int numAnnotators; // J: num annotators. (Derived from data)
    private int numFeatures; // V: num word types. (Derived from data)

    // Variable Assignments
    private int[][] z; // inferred topic assignments (one per doc and word position)
    private int[] y; // inferred 'true' label assignments (one per doc)
    private MaxEnt maxent; // logistic regression weights b.
    
    // static data-derived values
    private int[][][] a; // annotations indexed by [document][annotator][annotation_value]
    private int[] docSizes; // N_d: num words in dth doc. Derived from data. 
    private int[][] documents;  // documents represented as sequences of type indices. 
                                // indexed by [doc][word_position]. 

    // Sufficient statistics (these could be ints, but using double means less 
    // casting in the math down below; and allows us to pre-add prior values to save operations later)
    private double[][] perDocumentCountOfTopic; // n_{d,t} (replaces collapsed theta)
    private double[][] perTopicCountOfVocab; // n_{t,v} (replaces collapsed beta). (conceptually this would make more 
                                              // sense as n_{v,t} or perVocabCountOfTopic, but this ordering is more 
                                              // convenient when computing the log joint) 
    private double[][][] perAnnotatorCountOfYAndA; // n_{j,k,k'} (replaces collapsed gamma)
    
    // dynamic derived values  
    private double[] countOfTopic; // n_{t} (a function of perDocumentCountOfTopic that's used in the complete conditional)
    private double[][] perAnnotatorCountOfY; // \sum_k' n_{j,k,k'} number of times annotator annotated instance whose true class is Y
    private double[][] perDocumentCountOfAnnotator; // n_{d,j} number of annotations provided for each doc by each annotator

    // buffers (instantiated just once for a small efficiency gain)
    private final double[] zCoeffs; // stores topic probabilities while sampling a z
    private final double[] logisticClassScores; // stores class probabilities while sampling a z 
    private final double[] yCoeffs; // stores class probabilities while sampling a y
    private Map<String, Integer> instanceIndices;
    
    public State(Dataset data, PriorSpecification priors, int numTopics){
      this.data=data;
      this.priors=priors;
      this.numClasses=data.getInfo().getNumClasses();
      this.numDocuments=data.getInfo().getNumDocuments();
      this.numAnnotators=data.getInfo().getNumAnnotators();
      this.numFeatures=data.getInfo().getNumFeatures();
      this.numTopics=numTopics;
      this.y = new int[numDocuments];
      this.z = Datasets.featureVectors2FeatureSequences(data); // this gives us the right dimensions
      IntArrays.multiplyAndRoundToSelf(this.z, 0); // this gives right values
      // pre-compute static data-derived values
      this.instanceIndices = Datasets.instanceIndices(data);
      this.a = Datasets.compileDenseAnnotations(data);
      this.docSizes = Datasets.integerValuedInstanceSizes(data);
      this.documents = Datasets.featureVectors2FeatureSequences(data);
      this.deltas = CrowdsourcingUtils.annotatorConfusionMatricesFromPrior(priors, numClasses);
      // sufficient statistics (derived from y,z)
      this.perDocumentCountOfTopic = new double[numDocuments][numTopics];
      this.perTopicCountOfVocab = new double[numTopics][numFeatures];
      this.perAnnotatorCountOfYAndA = new double[numAnnotators][numClasses][numClasses];
      // derived quantities (functions of sufficient statistics) 
      this.countOfTopic = new double[numTopics];
      this.perAnnotatorCountOfY = new double[numAnnotators][numClasses];
      this.perDocumentCountOfAnnotator = new double[numDocuments][numAnnotators];
      // allocate buffers
      this.zCoeffs = new double[numTopics];
      this.logisticClassScores = new double[numClasses];
      this.yCoeffs = new double[numClasses];
    }

    /**
     * update the sufficient statistics to match current y and z values.
     * 
     * perDocumentCountOfTopic
     * perTopicCountOfVocab
     * perAnnotatorCountOfYAndA
     * 
     */
    public void updateSufficientStatistics() {
      // clear all values (just in case)
      Matrices.multiplyToSelf(this.perDocumentCountOfTopic, 0);
      Matrices.multiplyToSelf(this.perTopicCountOfVocab, 0);
      Matrices.multiplyToSelf(this.perAnnotatorCountOfYAndA, 0);
      
      for (int d=0; d<this.numDocuments; d++){
        
        // topic-related quantities
        for (int w=0; w<this.docSizes[d]; w++){
          int topic = this.z[d][w];
          int wordType = this.documents[d][w];
          
          this.perDocumentCountOfTopic[d][topic] += 1;
          this.perTopicCountOfVocab[topic][wordType] += 1;
        }
        
        // annotation-related quantities
        int classLabel = this.y[d];
        for (int j=0; j<numAnnotators; j++){
          for (int k=0; k<numClasses; k++){
            int numAnnotations = this.a[d][j][k];
            
            this.perAnnotatorCountOfYAndA[j][classLabel][k] += numAnnotations;
          }
        }
        
      } // end for doc
    }

    /**
     * updated derived counts to match sufficient statistics
     * 
     * countOfTopic
     * perAnnotatorCountOfA
     * perDocumentCountOfAnnotator
     */
    public void updateDerivedCounts() {
      // it's important that we derive countOfTopic from perTopicCountOfVocab instead of 
      // perDocumentCountOfTopic so that the right prior (eta) gets aggregated
      this.countOfTopic = Matrices.sumOverSecond(this.perTopicCountOfVocab); 
      for (int j=0; j<numAnnotators; j++){
        this.perAnnotatorCountOfY[j] = Matrices.sumOverSecond(this.perAnnotatorCountOfYAndA[j]);

        for (int d=0; d<numDocuments; d++){
          this.perDocumentCountOfAnnotator[d][j] = IntArrays.sum(this.a[d][j]);
        }
      }
    }
    

    public State clone(){
      State other = new State(data, priors, numTopics);
      // variable values
      other.maxent = this.maxent; // pass reference (not a deep copy)
      other.y = this.y.clone();
      other.z = this.z.clone();
      
      // data statistics are derived
      
      // sufficient statistics
      other.perDocumentCountOfTopic = Matrices.clone(this.perDocumentCountOfTopic);
      other.perTopicCountOfVocab = Matrices.clone(this.perTopicCountOfVocab);
      other.perAnnotatorCountOfYAndA = Matrices.clone(this.perAnnotatorCountOfYAndA);
      
      // derived quantities
      other.countOfTopic = this.countOfTopic.clone();
      other.perAnnotatorCountOfY = Matrices.clone(this.perAnnotatorCountOfY);
      other.perDocumentCountOfAnnotator = Matrices.clone(this.perDocumentCountOfAnnotator);
      
      // ignore buffers
      return other;
    }
  }
  
  // Builder pattern
  public static class ModelBuilder {
    
    private Dataset data;
    private int numTopics;
    private PriorSpecification priors;
    private AssignmentInitializer yInitializer;
    private AssignmentInitializer zInitializer;
    private String trainingOps;
    private RandomGenerator rnd;

    public ModelBuilder(Dataset dataset){
      this.data=dataset;
    }

    public ModelBuilder setZInitializer(AssignmentInitializer zInitializer){
      this.zInitializer=zInitializer;
      return this;
    }
    
    public ModelBuilder setYInitializer(AssignmentInitializer yInitializer){
      this.yInitializer=yInitializer;
      return this;
    }
    
    public ModelBuilder setNumTopics(int numTopics){
      this.numTopics=numTopics;
      return this;
    }
    
    public ModelBuilder setPriors(PriorSpecification priors){
      this.priors=priors;
      return this;
    }

    public ModelBuilder setTrainingOps(String trainingOps){
      this.trainingOps = trainingOps;
      return this;
    }

    public ModelBuilder setRandomGenerator(RandomGenerator rnd){
      this.rnd = rnd;
      return this;
    }
    
    protected ConfusedSLDADiscreteModel build() {
      ////////////////////
      // create model 
      ////////////////////
      State state = new State(data, priors, numTopics);
      
      // initialize state
      yInitializer.setData(data, state.instanceIndices);
      yInitializer.initialize(state.y);
      zInitializer.setData(data, state.instanceIndices);
      for (int d=0; d<state.numDocuments; d++){
        zInitializer.initialize(state.z[d]);
      }
      // initialize sufficient statistics
      state.updateSufficientStatistics();
      
      // Add prior values to all counts. This takes a liberty with the 
      // "count" semantics, but simplifies the math below since fractional 
      // prior values are ALWAYS added to their corresponding counts.
      // TODO: precompute sufficient statistics
      Matrices.addToSelf(state.perAnnotatorCountOfYAndA, state.deltas);
      Matrices.addToSelf(state.perDocumentCountOfTopic, priors.getBTheta());
      Matrices.addToSelf(state.perTopicCountOfVocab, priors.getBPhi());
      state.updateDerivedCounts(); // pre-compute some derived counts 

      // create model 
      ConfusedSLDADiscreteModel model = new ConfusedSLDADiscreteModel(state);
      maximizeB(state); // ensure that log-linear weights exist

      ////////////////////
      // train model 
      ////////////////////
      ModelTrainer trainer = new ModelTrainer(state);
      ModelTraining.doOperations(trainingOps, trainer);

      logger.info("Training finished with log joint="+unnormalizedLogJoint(state));
      System.out.println("Topic Results");
      printTopNWordsPerTopic(state, 10);
      
      return model;
      
    }

    private class ModelTrainer implements SupportsTrainingOperations{
      private State state;
      public ModelTrainer(State state){
        this.state=state;
      }
      /** {@inheritDoc} */
      @Override
      public void sample(String variableName, String[] args) {
        Preconditions.checkNotNull(variableName);
        Preconditions.checkArgument(args.length>=1);
        int numIterations = Integer.parseInt(args[0]);
        
        // Joint
        if (variableName.equals("all")){
          // sample topics and class labels jointly (SLOW)
          state.includeMetadataSupervision = true;
          for (int i=0; i<numIterations; i++){
            logger.info("maximizing log-linear weights b");
            maximizeB(state); 
            logger.info("sampling class label vector y");
            sampleY(state, rnd);
            logger.info("sampling topic matrix z");
            sampleZ(state, rnd);
            if (DEBUG){
              logger.info("sample Y+Z+B iteration "+i+" with (unnormalized) log joint "+unnormalizedLogJoint(state));
            }
          }
          logger.info("finished sampling "+numIterations+" times");
        }
        // Y
        else if (variableName.toLowerCase().equals("y")){
          state.includeMetadataSupervision = false;
          for (int i=0; i<numIterations; i++){
            sampleY(state, rnd);
            if (DEBUG){
              logger.info("sample Y iteration "+i+" with (unnormalized) log joint "+unnormalizedLogJoint(state));
            }
          }
          logger.info("finished sampling Y "+numIterations+" times");
        }
        // Z
        else if (variableName.toLowerCase().equals("z")){
          state.includeMetadataSupervision = false;
          for (int i=0; i<numIterations; i++){
            sampleZ(state, rnd);
            if (DEBUG){
              logger.info("sample Z iteration "+i+" with (unnormalized) log joint "+unnormalizedLogJoint(state));
            }
          }
          logger.info("finished sampling Z "+numIterations+" times");
        }
        // B
        else if (variableName.toLowerCase().equals("b")){
          throw new UnsupportedOperationException("cannot sample b");
        }
      }

      /** {@inheritDoc} */
      @Override
      public void maximize(String variableName, String[] args) {
        Preconditions.checkNotNull(variableName,"training operations must reference a specific variable (e.g., maximize-all, sample-y, etc)");
        int maxNumIterations = (args.length>=1)? Integer.parseInt(args[0]): 100;
        
        // Joint
        if (variableName.equals("all")){
          // maximize topics and class labels jointly (SLOW)
          state.includeMetadataSupervision = true;
          maximizeUntilConvergence(state, 0, 0, maxNumIterations); 
          printTopNWordsPerTopic(state, 10);
        }
        // Y
        else if (variableName.toLowerCase().equals("y")){ 
          // maximize class labels independently (FAST)
          state.includeMetadataSupervision = false;
          maximizeYUntilConvergence(state, 0, maxNumIterations);
        }
        // Z
        else if (variableName.toLowerCase().equals("z")){
          // maximize topics independently (FAST)
          state.includeMetadataSupervision = false;
          maximizeZUntilConvergence(state, 0, maxNumIterations);
        }
        // B
        else if (variableName.toLowerCase().equals("b")){
          // maximize log linear model independently 
          maximizeB(state);
        }
      }
      
    }
    
  } // end builder


  /**
   * A class to handle translation back and forth from our representations 
   * to a form mallet can handle (for the log-linear calculations)
   */
  private static class MalletInterface{
    private static Alphabet dataAlphabet;
    private static LabelAlphabet labelAlphabet;
    
    private MalletInterface(){}
    
    private static void ensureDataAlphabet(int numTopics){
      // cache data alphabet
      if (dataAlphabet==null){
        Alphabet alphabet = new Alphabet(numTopics);
        alphabet.startGrowth();
        for (int t=0; t<numTopics; t++){
          alphabet.lookupIndex(t, true);
        }
        alphabet.stopGrowth();
        dataAlphabet = alphabet;

        // sanity check - make sure we really create an identity mapping
        Preconditions.checkState(dataAlphabet.size()==numTopics);
        for (int i=0; i<numTopics; i++){
          Preconditions.checkState(dataAlphabet.lookupIndex(i)==i);
        }
      }
      Preconditions.checkState(dataAlphabet.size()==numTopics);
      
    }
    
    private static void ensureLabelAlphabet(int numLabels){
      // cache data alphabet
      if (labelAlphabet==null){
        LabelAlphabet alphabet = new LabelAlphabet();
        alphabet.startGrowth();
        for (int l=0; l<numLabels; l++){
          alphabet.lookupIndex(l, true);
        }
        alphabet.stopGrowth();
        labelAlphabet = alphabet;
        
        // sanity check - make sure we really create an identity mapping
        Preconditions.checkState(labelAlphabet.size()==numLabels);
        for (int i=0; i<numLabels; i++){
          Preconditions.checkState(labelAlphabet.lookupIndex(i)==i);
        }
      }
      Preconditions.checkState(labelAlphabet.size()==numLabels);
    }
    
    public static MaxEnt logisticRegressionFromTopicToClass(State s){
      // cache alphabets
      ensureDataAlphabet(s.numTopics);
      ensureLabelAlphabet(s.numClasses);
      
      // create a training set by adding each instance K times, each weighted by softlabels
      InstanceList trainingSet = new InstanceList(dataAlphabet, labelAlphabet);
      for (int doc=0; doc<s.numDocuments; doc++){
        // note: do soft-EM by maintaining distributions over doc labels?
        // note: do even softer-EM by maintaining distributions over topic vectors?
        trainingSet.add(instanceForTopicCounts(s.perDocumentCountOfTopic[doc], null, s.docSizes[doc], s.priors.getBTheta(), s.y[doc]));
      }
      
      // train
      return new MaxEntTrainer(s.maxent).train(trainingSet);
    }
    
    /**
     * Use a topiccount vector to create a zbar vector by adding an extra count to topic "extratopic", 
     * and subtracting an offset from the topic count vector (in this case, an alpha prior value), 
     * and normalizing by docSize
     */
    public static Instance instanceForTopicCounts(double[] topicCounts, Integer topicWithExtraCount, int docSize, double topicOffset, Integer classLabel){
      // cache data alphabet
      ensureDataAlphabet(topicCounts.length);
      
      // convert empirical topic vector to mallet feature vector
      int[] featureIndices = new int[topicCounts.length];
      double[] featureValues = new double[topicCounts.length];
      for (int t=0; t<topicCounts.length; t++){
        featureIndices[t] = dataAlphabet.lookupIndex(t, false);
        // normalize the topic count vector into zbar
        double extraCount = (topicWithExtraCount!=null && topicWithExtraCount==t)? 1: 0;
        featureValues[t] = (topicCounts[t] - topicOffset + extraCount)/docSize; 
      }
      FeatureVector fv = new FeatureVector(dataAlphabet, featureIndices, featureValues);
      
      // convert to instance
      String name = null; // not important
      String source = null; // not important
      Label target = null; // unknown
      if (classLabel!=null){
        target = labelAlphabet.lookupLabel(classLabel, false);
      }
      Instance inst = new cc.mallet.types.Instance(fv, target, name, source);
      return inst;
    }
  }
  
  


  //////////////////////////////////////////////
  // Model Code
  //////////////////////////////////////////////
  private State state;
  
  public ConfusedSLDADiscreteModel(State state) {
    this.state=state;
  }

  
  public static void maximizeUntilConvergence(State s, int minNumYChanges, int minNumZChanges, int maxIterations){
    int numYChanges = Integer.MAX_VALUE;
    int numZChanges = Integer.MAX_VALUE;
    int iterations = 0;
    while ((numYChanges > minNumYChanges || numZChanges > minNumZChanges) && iterations < maxIterations){
      logger.info("maximizing log-linear model parameters b");
      maximizeB(s); // set maxent model weights
      logger.info("maximizing topic assignments Z");
      numZChanges = maximizeZ(s); // set topic assignments
      logger.info("maximizing inferred labels Y");
      numYChanges = maximizeY(s); // set inferred label values
      logger.info("maximization iteration "+iterations+" with "+numYChanges+" Y changes and "+numZChanges+" Z changes with (unnormalized) joint "+unnormalizedLogJoint(s));
      iterations++;
    }
    logger.info("finished maximization after "+iterations+" iterations");
  }

  
  
  public Predictions predict(Dataset trainingInstances, Dataset heldoutInstances, RandomGenerator rnd){
    // em-derived labels
    List<Prediction> labeledPredictions = Lists.newArrayList();
    List<Prediction> unlabeledPredictions = Lists.newArrayList();
    for (DatasetInstance inst: state.data){
      
      int index = state.instanceIndices.get(inst.getInfo().getSource());
      
      BasicPrediction prediction = new BasicPrediction(state.y[index], inst);
      
      if (inst.getInfo().getNumAnnotations()>0){
        // annotated
        labeledPredictions.add(prediction);
      }
      else{
        // unannotated
        unlabeledPredictions.add(prediction);
      }
    }
    
    // generalization labels
    List<Prediction> heldoutPredictions = Lists.newArrayList();
    // TODO: figure this out later--we don't really care too much about it right now.
//    for (DatasetInstance inst: heldoutInstances){
//      Preconditions.checkArgument(inst.getInfo().getNumAnnotations()==0,
//          "test data must not have annotations!");
//      
//      // create a state with all the counts of the learned state, but only a single instance to do inference on
//      // this effectively allows us to sample the distribution over this new document given everything that 
//      // has gone before. 
//      State testState = state.clone();
//      testState.data = null;
//      // testdoc data stats
//      testState.numDocuments = 1;
//      testState.a = new int[1][testState.numAnnotators][testState.numClasses]; // empty
//      testState.docSizes =   new int[]{Integers.fromDouble(inst.asFeatureVector().sum(), Datasets.INT_CAST_THRESHOLD)}; // 1 docsize
//      testState.documents = new int[][]{SparseFeatureVectors.asSequentialIndices(inst.asFeatureVector())}; // 1 doc
//      // variables
//      testState.z = new int[][]{RandomGenerators.nextUniformIndependentIntArray(testState.docSizes[0], testState.numTopics, rnd)}; // random z
//      testState.y = RandomGenerators.nextUniformIndependentIntArray(testState.docSizes[0], testState.numClasses, rnd); // random y 
//      // doc-specific counts (happen to be trivial since we assume no annotations on a test doc)
//      testState.perDocumentCountOfAnnotator = new double[1][testState.numAnnotators]; // empty annotation counts
//      testState.perDocumentCountOfTopic = new double[][]{
//          DoubleArrays.fromInts(IntArrays.denseCounterOf(testState.z[0],testState.numTopics))}; // count all topic assignment for the doc
//      // sufficient statistics and derived counts
//      // add this document and all it's words to the current counts (they'll be subtracted in order to compute the complete conditionals)
//      includeDocumentInCounts(testState, 0); // this actually shouldn't do anything since there are no annotations in test doc
//      for (int w=0; w<testState.docSizes[0]; w++){
//        includeWordInCounts(testState, 0, w);
//      }
//      
//      // sample topics (few times--probably even 1--should suffice, 
//      // since the existing counts should overwhelmingly place this doc somewhere)
////      sampleZ(testState, rnd);
////      sampleZ(testState, rnd);
////      sampleZ(testState, rnd);
//      maximizeZ(testState);
//      
//      int predictedLabel;
//      if (state.includeMetadataSupervision){
//        // get transformed prediction (just a function of the log-linear mapping now)
//        Instance testzbar = MalletInterface.instanceForTopicCounts(
//            testState.perDocumentCountOfTopic[0], null, testState.docSizes[0], testState.priors.getBTheta(), null);
//        predictedLabel = testState.maxent.classify(testzbar).getLabeling().getBestIndex(); // assumes the identity label mapping
//      }
//      else{
//        // without a learned mapping from zbar to classLabel, we have no way of generating a prediction
//        predictedLabel = 0;
//      }
//      
//      heldoutPredictions.add(new BasicPrediction(predictedLabel, inst));
//    }
    
    double logJoint = unnormalizedLogJoint(state);
    double[] annotatorAccuracies = estimateAnnotatorAccuracies(state);
    double[][][] annotatorConfusionMatrices = state.perAnnotatorCountOfYAndA;
    double machineAccuracy = -1;
    double[][] machineConfusionMatrix = null;
    return new Predictions(labeledPredictions, unlabeledPredictions, heldoutPredictions, annotatorAccuracies, annotatorConfusionMatrices, machineAccuracy, machineConfusionMatrix, logJoint);
  }

  private static double[] estimateAnnotatorAccuracies(State s){
    // ybias is the p(y|a,w) estimated from our current sample of y's
    // (would be better if we had a distribution over them)
    double[] ybias = DoubleArrays.fromInts(IntArrays.denseCounterOf(s.y, s.numClasses));
    DoubleArrays.normalizeToSelf(ybias);
    return CrowdsourcingUtils.getAccuracies(ybias, s.perAnnotatorCountOfYAndA);
  }
  
  

  /////////////////////////////////////////////////
  // Code to Sample Z
  // (mutates model state in-place for efficiency)
  /////////////////////////////////////////////////
  private static void excludeWordFromCounts(State s, int doc, int word) {
    int topic = s.z[doc][word]; // the topic value that is being excluded
    int wordType = s.documents[doc][word];
    s.perDocumentCountOfTopic[doc][topic] -= 1;
    assert s.perDocumentCountOfTopic[doc][topic] >= 0;
    s.perTopicCountOfVocab[topic][wordType] -= 1;
    assert s.perTopicCountOfVocab[topic][wordType] >= 0;
    s.countOfTopic[topic] -= 1;
    assert s.countOfTopic[topic] >= 0;
  }
  
  private static void includeWordInCounts(State s, int doc, int word) {
    int topic = s.z[doc][word]; // the topic value that is being included
    int wordType = s.documents[doc][word];
    s.perDocumentCountOfTopic[doc][topic] += 1;
    s.perTopicCountOfVocab[topic][wordType] += 1;
    s.countOfTopic[topic] += 1;
  }
  
  public static void maximizeZUntilConvergence(State s, int minNumChanges, int maxIterations){
    int numChanges = Integer.MAX_VALUE;
    int iterations = 0;
    while (numChanges>minNumChanges && iterations<maxIterations){
      logger.info("maximizing topic assignments Z");
      numChanges = maximizeZ(s); // set topic assignments
      logger.info("maximizing Z iteration "+iterations+" with "+numChanges+" changes and (unnormalized) joint "+unnormalizedLogJoint(s));
      iterations++;
    }
    logger.info("finished maximizing Z after "+iterations+" iterations");
  }

  public static int maximizeZ(State s){
    int numChanges = 0;
    for (int d=0; d<s.numDocuments; d++){
      numChanges += updateZDoc(s, d, null);
    }
    return numChanges;
  }
  
  public static int sampleZ(State s, RandomGenerator rnd){
    int numChanges = 0;
    for (int d=0; d<s.numDocuments; d++){
      numChanges += updateZDoc(s, d, rnd);
    }
    return numChanges;
  }
  
  /**
   * If rnd is null, then it takes the assignment that maximizes 
   * the joint. Otherwise, samples. 
   */
  public static int updateZDoc(State s, int doc, RandomGenerator rnd){
    int numChanges = 0;
    int docsize = s.docSizes[doc];
    for (int word=0; word<docsize; word++){
      numChanges += updateZDocWord(s, doc, word, rnd);
    }
    return numChanges;
  }

  /**
   * If rnd is null, then it takes the assignment that maximizes 
   * the joint. Otherwise, samples. 
   */
  public static int updateZDocWord(State s, int doc, int word, RandomGenerator rnd){
    // decrement counts using current z value 
    // (full conditionals are defined in terms of excluding current word)
    excludeWordFromCounts(s, doc, word); 
    
    // precomputed values for efficiency
    int wordType = s.documents[doc][word];
    int documentClass = s.y[doc];

    // populate unnormalized probability vector and then sample
    for (int t=0; t<s.numTopics; t++){
      
      // vanilla lda contribution with symmetric priors 
      // (see Griffiths and Steyver's "finding scientific topics" or 
      // better, Jonathan Chang's dissertation appendix D.1)
      // (n_d,t + alpha_t) * (n_{w_dn,t} + eta_{w_dn}) / (N_t + V * eta) where counts exclude the current word
      //
      // note: the counts here already include added prior values because of the way we initialized them. Therefore, 
      // n_d,t * n_{w_dn,t} / N_t 
      s.zCoeffs[t] = (s.perDocumentCountOfTopic[doc][t]) * 
          ((s.perTopicCountOfVocab[t][wordType]) / (s.countOfTopic[t]));

      assert s.zCoeffs[t]>=0;
      
      if (s.includeMetadataSupervision){
        // metadata contribution 
        // this is the log-linear portion of the model that 
        // uses weights w to map from a document's empirical topics (zbar) 
        // to its inferred class label (sampled in a different step).
        // We could drop some terms if we implemented this ourselves, but 
        // it doesn't seem worth it.
        Instance zbar = MalletInterface.instanceForTopicCounts(s.perDocumentCountOfTopic[doc], t, s.docSizes[doc], s.priors.getBTheta(), null);
        s.maxent.getClassificationScores(zbar, s.logisticClassScores);
        s.zCoeffs[t] *= s.logisticClassScores[documentClass];
      }
      
      assert DoubleArrays.min(s.logisticClassScores)>=0;
    }
    
    // sample (or maximize) a new topic
    int oldTopic = s.z[doc][word];
    int newTopic = (rnd!=null)? RandomGenerators.nextIntUnnormalizedProbs(rnd, s.zCoeffs): DoubleArrays.argMax(s.zCoeffs);
    s.z[doc][word] = newTopic;

    // increment counts using newly sampled value
    includeWordInCounts(s, doc, word); 
    
    return oldTopic==newTopic? 0: 1; // return number of changes
  }


  /////////////////////////////////////////////////
  // Code to Sample Y
  // (mutates model state in-place for efficiency)
  /////////////////////////////////////////////////
  private static void includeDocumentInCounts(State s, int doc) {
    int classLabel = s.y[doc];
    
    for (int j = 0; j < s.numAnnotators; j++) {
      for (int k = 0; k < s.numClasses; k++) {
        s.perAnnotatorCountOfYAndA[j][classLabel][k] += s.a[doc][j][k];
      }
      s.perAnnotatorCountOfY[j][classLabel] += s.perDocumentCountOfAnnotator[doc][j];
    }
  }

  private static void excludeDocumentFromCounts(State s, int doc) {
    int classLabel = s.y[doc];
    
    for (int j = 0; j < s.numAnnotators; j++) {
      for (int k = 0; k < s.numClasses; k++) {
        s.perAnnotatorCountOfYAndA[j][classLabel][k] -= s.a[doc][j][k];
      }
      s.perAnnotatorCountOfY[j][classLabel] -= s.perDocumentCountOfAnnotator[doc][j];
    }
  }

  public static void maximizeYUntilConvergence(State s, int minNumChanges, int maxIterations){
    int numChanges = Integer.MAX_VALUE;
    int iterations = 0;
    while (numChanges>minNumChanges && iterations<maxIterations){
      logger.info("maximizing inferred labels Y");
      numChanges = maximizeY(s); // set inferred label values
      logger.info("maximizing Y iteration "+iterations+" with "+numChanges+" changes and (unnormalized) joint "+unnormalizedLogJoint(s));
      iterations++;
    }
    logger.info("finished maximizing Y after "+iterations+" iterations");
  }
  
  public static int maximizeY(State s){
    int numChanges = 0;
    for (int d=0; d<s.numDocuments; d++){
      numChanges += updateYDoc(s, d, null);
    }
    return numChanges;
  }
  
  public static int sampleY(State s, RandomGenerator rnd){
    int numChanges = 0;
    for (int d=0; d<s.numDocuments; d++){
      numChanges += updateYDoc(s, d, rnd);
    }
    return numChanges;
  }

  /**
   * If rnd is null, then it takes the assignment that maximizes 
   * the joint. Otherwise, samples. 
   */
  public static int updateYDoc(State s, int doc, RandomGenerator rnd){
    // decrement counts using current y value 
    // (full conditionals are defined in terms of excluding current document)
    excludeDocumentFromCounts(s, doc);

    if (s.includeMetadataSupervision){
      // precompute for efficiency
      Instance zbar = MalletInterface.instanceForTopicCounts(s.perDocumentCountOfTopic[doc], null, s.docSizes[doc], s.priors.getBTheta(), null);
      s.maxent.getClassificationScores(zbar, s.logisticClassScores);
      DoubleArrays.logToSelf(s.logisticClassScores);
    }
    
    // compute unnormalized probs in log space this time
    // because of the rising factorial in the annotation contribution
    for (int k=0; k<s.numClasses; k++){

      s.yCoeffs[k] = 0;
      
      if (s.includeMetadataSupervision){
        // metadata contribution
        s.yCoeffs[k] = s.logisticClassScores[k];
      }
          
      // annotation contribution
      for (int j = 0; j < s.numAnnotators; j++) {
        s.yCoeffs[k] +=
          // note: this method assumes that prior values have already been added to counts.
          // we do this when we initialized the counts
          BlockCollapsedMultiAnnModelMath.computeYSum(
              s.perAnnotatorCountOfYAndA[j][k], 
              s.perAnnotatorCountOfY[j][k], 
              s.a[doc][j], 
              Integers.fromDouble(s.perDocumentCountOfAnnotator[doc][j], Datasets.INT_CAST_THRESHOLD));
      }
      
    }

    int oldClassLabel = s.y[doc];
    int newClassLabel = (rnd!=null)? RandomGenerators.nextIntUnnormalizedLogProbs(rnd, s.yCoeffs): DoubleArrays.argMax(s.yCoeffs);
    s.y[doc] = newClassLabel;

    // increment counts using newly sampled y value
    includeDocumentInCounts(s, doc);
    
    return oldClassLabel==newClassLabel? 0: 1; // return number of changes made
  }



  /////////////////////////////////////////////////
  // Code to Optimize w (logistic regression)
  // (mutates model state in-place for efficiency)
  /////////////////////////////////////////////////
  public static void maximizeB(State s){
    // retrain a maxent model between per-document topic vectors 
    // and class labels.
    s.maxent = MalletInterface.logisticRegressionFromTopicToClass(s);
  }
  
  

  /////////////////////////////////////////////////
  // Diagnostics  
  /////////////////////////////////////////////////
  
  // Code to Calculate log joint (for convergence)
  public static double unnormalizedLogJoint(State s){
    double logTotal = 0;
    // per-document topic contributions  / B(alpha)
    // propto B(n_{d} + alpha)
    for (int doc=0; doc<s.numDocuments; doc++){
      // note: the counts already include added prior values because of the way we initialized them.
      logTotal += GammaFunctions.logBeta(s.perDocumentCountOfTopic[doc]); // ln B(n_{d} + alpha)
    }
    
    // per-topic vocab contribution B(eta + n_{k}) / B(eta)
    // propto B(eta + n_{k})
    for (int t=0; t<s.numTopics; t++){
      // note: the counts already include added prior values because of the way we initialized them.
      logTotal += GammaFunctions.logBeta(s.perTopicCountOfVocab[t]); // ln B(eta + n_{k})
    }
    
    // log-linear class label contribution p(y|z,b) 
    for (int doc=0; doc<s.numDocuments; doc++){
      // get prob of each class assignment
      Instance inst = MalletInterface.instanceForTopicCounts(s.perDocumentCountOfTopic[doc], null, s.docSizes[doc], s.priors.getBTheta(), null);
      s.maxent.getClassificationScores(inst, s.logisticClassScores);
      // add log prob of current class assignment
      logTotal += Math.log(s.logisticClassScores[s.y[doc]]);
    }
    
    // annotation contribution B(delta_{jk} + n_{j,k}) / B(d_{j,k}) * M(a)
    // propto B(delta_{jk} + n_{j,k})
    for (int j=0; j<s.numAnnotators; j++){
      for (int k=0; k<s.numClasses; k++){
        // note: the counts already include added prior values because of the way we initialized them.
        logTotal += GammaFunctions.logBeta(s.perAnnotatorCountOfYAndA[j][k]);
      }
    }
    
    return logTotal;
  }

  // Code to print top n words for topics
  public static void printTopNWordsPerTopic(State s, int n){
    int topic = 0;
    for (double[] vocab: s.perTopicCountOfVocab){
      System.out.println("Top "+n+" words for topic "+topic+":");
      for (int topIndex: DoubleArrays.argMaxList(n, vocab)){
        System.out.println("\t"+s.data.getInfo().getFeatureIndexer().get(topIndex)+"="+vocab[topIndex]);
      }
      topic++;
    }
  }

  
}
