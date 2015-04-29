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

import org.apache.commons.math3.random.RandomGenerator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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

import edu.byu.nlp.classify.data.DatasetLabeler;
import edu.byu.nlp.classify.eval.BasicPrediction;
import edu.byu.nlp.classify.eval.Prediction;
import edu.byu.nlp.classify.eval.Predictions;
import edu.byu.nlp.classify.util.ModelTraining;
import edu.byu.nlp.classify.util.ModelTraining.IntermediatePredictionLogger;
import edu.byu.nlp.classify.util.ModelTraining.SupportsTrainingOperations;
import edu.byu.nlp.crowdsourcing.CrowdsourcingUtils;
import edu.byu.nlp.crowdsourcing.ModelInitialization.AssignmentInitializer;
import edu.byu.nlp.crowdsourcing.ModelInitialization.MatrixAssignmentInitializer;
import edu.byu.nlp.crowdsourcing.PriorSpecification;
import edu.byu.nlp.crowdsourcing.gibbs.BlockCollapsedMultiAnnModelMath;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.DatasetInstance;
import edu.byu.nlp.dataset.Datasets;
import edu.byu.nlp.math.GammaFunctions;
import edu.byu.nlp.math.optimize.ConvergenceCheckers;
import edu.byu.nlp.math.optimize.IterativeOptimizer;
import edu.byu.nlp.math.optimize.IterativeOptimizer.ReturnType;
import edu.byu.nlp.math.optimize.ValueAndObject;
import edu.byu.nlp.stats.RandomGenerators;
import edu.byu.nlp.stats.SymmetricDirichletMultinomialDiagonalMatrixMAPOptimizable;
import edu.byu.nlp.stats.SymmetricDirichletMultinomialMatrixMAPOptimizable;
import edu.byu.nlp.util.DoubleArrays;
import edu.byu.nlp.util.IntArrayCounter;
import edu.byu.nlp.util.IntArrays;
import edu.byu.nlp.util.Integers;
import edu.byu.nlp.util.Matrices;
import edu.byu.nlp.util.Pair;

/**
 * @author plf1
 *
 * For more info on this model, see notes in 
 * https://drive.google.com/drive/u/0/#folders/0B5phubFg2ZvVSDRvS0U1S3pScjQ/0B5phubFg2ZvVNWtfMEN1b0NMWlk
 */
public class ConfusedSLDADiscreteModel {
  private static final Logger logger = LoggerFactory.getLogger(ConfusedSLDADiscreteModel.class);
  public static final int DEFAULT_TRAINING_ITERATIONS = 25;
  public static final int HYPERPARAM_TUNING_PERIOD = 25;
  
  //TODO: the static eta matrix implementation is not totally debugged yet. 
  // bad smells: 1) topic correspondence doesn't always work even with many perfect annotations
  //             2) accuracy on unannotated data is abysmal. I'd think it would be at least ok for good data. 
  // both of these things together suggest that class identity is getting swapped somewhere??
  private static final boolean STATIC_ETA = false; 
  
  //////////////////////////////////////////////
  // Helper Code
  //////////////////////////////////////////////
  /**
   * this tracks the state of the sampler. It should 
   * be sufficient to save/restore the model at any point.
   */
  public static class State{
    
    // flags
    boolean includeMetadataSupervision = false; // if false, reduces to unsupervised LDA
    
    // data
    Dataset data;
    PriorSpecification priors;
    private double[][][] deltas; // prior over gamma (a function of the priors object)
    private int numTopics; // T: num topics. 
    private int numClasses; // K: num classes. (Derived from data)
    int numDocuments; // D: num documents. (Derived from data)
    private int numAnnotators; // J: num annotators. (Derived from data)
    private int numFeatures; // V: num word types. (Derived from data)

    // Variable Assignments
    int[][] z; // inferred topic assignments (one per doc and word position)
    int[] y; // inferred 'true' label assignments (one per doc)
    IntArrayCounter yMarginals; // track posterior marginal distribution over y
    private MaxEnt maxent; // logistic regression weights b.
    
    // static data-derived values
    private int[][][] a; // annotations indexed by [document][annotator][annotation_value]
    private int[] docSizes; // N_d: num words in dth doc. Derived from data. 
    private int[] docAnnotationCounts; // number of annotations per doc. Derived from data
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
    private double[] cachedMetadataScores; // stores topic scores for a document d while sampling the words in that doc 
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
      this.yMarginals = new IntArrayCounter(numDocuments, numClasses);
      this.z = Datasets.featureVectors2FeatureSequences(data); // this gives us the right dimensions
      Matrices.multiplyAndRoundToSelf(this.z, 0); // initialize values to 0
      // pre-compute static data-derived values
      this.instanceIndices = Datasets.instanceIndices(data);
      this.a = Datasets.compileDenseAnnotations(data);
      this.docSizes = Datasets.countIntegerDocSizes(data);
      this.docAnnotationCounts = Datasets.countDocAnnotations(data);
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
      this.cachedMetadataScores = new double[numTopics];
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
        
        // sanity check
        Preconditions.checkState(DoubleArrays.sum(this.perDocumentCountOfTopic[d])==this.docSizes[d], 
            "total number of topics in a document ("+DoubleArrays.sum(perDocumentCountOfTopic[d])+") should equal docsize ("+docSizes[d]+")");
        
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
      other.yMarginals = this.yMarginals.clone();
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

  private static void addPriorsToCounts(State s){
    Matrices.addToSelf(s.perAnnotatorCountOfYAndA, s.deltas);
    Matrices.addToSelf(s.perDocumentCountOfTopic, s.priors.getBTheta());
    Matrices.addToSelf(s.perTopicCountOfVocab, s.priors.getBPhi());
    s.updateDerivedCounts(); // pre-compute some derived counts 
  }
  
  private static void subtractPriorsFromCounts(State s){
    Matrices.subtractFromSelf(s.perAnnotatorCountOfYAndA, s.deltas);
    Matrices.subtractFromSelf(s.perDocumentCountOfTopic, s.priors.getBTheta());
    Matrices.subtractFromSelf(s.perTopicCountOfVocab, s.priors.getBPhi());
    s.updateDerivedCounts(); // pre-compute some derived counts 
  }
  
  // Builder pattern
  public static class ModelBuilder {
    
    private Dataset data;
    private int numTopics;
    private PriorSpecification priors;
    private AssignmentInitializer yInitializer;
    private MatrixAssignmentInitializer zInitializer;
    private String trainingOps;
    private RandomGenerator rnd;
    private IntermediatePredictionLogger intermediatePredictionLogger;
    boolean predictSingleLastSample = false;

    public ModelBuilder(Dataset dataset){
      this.data=dataset;
    }

    public ModelBuilder setZInitializer(MatrixAssignmentInitializer zInitializer){
      this.zInitializer=zInitializer;
      return this;
    }
    
    public ModelBuilder setPredictSingleLastSample(boolean predictSingleLastSample){
      // if false, reports the mode of the marginal posterior for each y) = false; // if false, reports the mode of the marginal posterior for each y
      this.predictSingleLastSample = predictSingleLastSample;
      return this;
    }; 
    
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
    
    public ModelBuilder setIntermediatePredictionLogger(IntermediatePredictionLogger intermediatePredictionLogger){
      this.intermediatePredictionLogger=intermediatePredictionLogger;
      return this;
    }

    protected ConfusedSLDADiscreteModel build() {
      return build(true);
    }
    
    protected ConfusedSLDADiscreteModel build(boolean doTraining) {
      ////////////////////
      // create model 
      ////////////////////
      State state = new State(data, priors, numTopics);
      
      // initialize state
      yInitializer.setData(data, state.instanceIndices);
      yInitializer.initialize(state.y);
      zInitializer.setData(data, state.instanceIndices);
      for (int d=0; d<state.numDocuments; d++){
        zInitializer.getInitializerFor(d).initialize(state.z[d]);
      }
      // initialize sufficient statistics
      state.updateSufficientStatistics();
      
      // Add prior values to all counts. This takes a liberty with the 
      // "count" semantics, but simplifies the math below since fractional 
      // prior values are ALWAYS added to their corresponding counts.
      addPriorsToCounts(state);
      
      // create model 
      ConfusedSLDADiscreteModel model = new ConfusedSLDADiscreteModel(state);
      maximizeB(state); // ensure that log-linear weights exist

      if (doTraining){
        ////////////////////
        // train model 
        ////////////////////
        ModelTrainer trainer = new ModelTrainer(state);
        ModelTraining.doOperations(trainingOps, trainer, intermediatePredictionLogger);
  
        logger.info("Training finished with log joint="+unnormalizedLogJoint(state));
        logger.info("Final topics");
        logTopNWordsPerTopic(state, 10);
      }
      
      return model;
      
    }

    private class ModelTrainer implements SupportsTrainingOperations{
      private State state;
      public ModelTrainer(State state){
        this.state=state;
      }
      /** {@inheritDoc} */
      @Override
      public Double sample(String variableName, int iteration, String[] args) {
        // assume that all sampling should be done wrt the the current state of the other variables
        // (never ignore metadata supervision)
        state.includeMetadataSupervision = true;
        state.yMarginals.reset(); // reset marginals ever time we do a new operation
        
        Preconditions.checkNotNull(variableName);
        
        // Joint
        if (variableName.equals("all")){
          // sample topics and class labels jointly (SLOW)
//        state.includeMetadataSupervision = true;
          logger.debug("maximizing log-linear weights B iteration "+iteration);
          maximizeB(state); 
          logger.debug("sampling class label vector Y iteration "+iteration);
          sampleY(state, rnd);
          logger.debug("sampling topic matrix Z iteration "+iteration);
          sampleZ(state, rnd);
          // periodically tune hypers and report joint
          if (iteration%HYPERPARAM_TUNING_PERIOD==0){
            if (state.priors.getInlineHyperparamTuning()){
              updateBTheta(state);
              updateBPhi(state);
              updateBGamma(state);
            }
          }
        }
        // Y
        else if (variableName.toLowerCase().equals("y")){
//        state.includeMetadataSupervision = false;
          sampleY(state, rnd);
          logger.debug("sample Y+B iteration "+iteration);
          // periodically tune hypers and report joint
          if (iteration%HYPERPARAM_TUNING_PERIOD==0){
            maximizeB(state); 
            if (state.priors.getInlineHyperparamTuning()){
              updateBGamma(state);
            }
          }
        }
        // Z
        else if (variableName.toLowerCase().equals("z")){
//        state.includeMetadataSupervision = false;
          sampleZ(state, rnd);
          logger.debug("sample Z+B iteration "+iteration);
          // periodically tune hypers and report joint
          if (iteration%HYPERPARAM_TUNING_PERIOD==0){
            maximizeB(state); 
            if (state.priors.getInlineHyperparamTuning()){
              updateBTheta(state);
              updateBPhi(state);
            }
          }
        }
        // B
        else if (variableName.toLowerCase().equals("b")){
          throw new UnsupportedOperationException("cannot sample b");
        }
        else{
          throw new IllegalArgumentException("unknown variable name "+variableName);
        }
        
        // for efficiency, only calculate objective value periodically
        if (iteration%HYPERPARAM_TUNING_PERIOD==0){
          return unnormalizedLogJoint(state);
        }
        return null;
      }

      private int cumulativeNumChanges = 0;
      /** {@inheritDoc} */
      @Override
      public Double maximize(String variableName, int iteration, String[] args) {
        Preconditions.checkNotNull(variableName);
        
        // reset the cumulative number of changes
        if (iteration==0){
          cumulativeNumChanges = 0;
        }
        state.yMarginals.reset(); // reset marginals ever time we do a new operation
        
        // assume that all maximization should be done wrt the the current state of the other variables
        // (never ignore metadata supervision)
        state.includeMetadataSupervision = true;
        
        // Joint
        if (variableName.equals("all")){
          // maximize topics and class labels jointly (SLOW)
//          state.includeMetadataSupervision = true;
          
          logger.debug("maximizing log-linear model parameters b iteration "+iteration);
          maximizeB(state); // set maxent model weights
          logger.debug("maximizing topic assignments Z iteration "+iteration);
          cumulativeNumChanges += maximizeZ(state); // set topic assignments
          logger.debug("maximizing inferred labels Y iteration "+iteration);
          cumulativeNumChanges += maximizeY(state); // set inferred label values
          // tune hyperparams
          updateBTheta(state);
          updateBPhi(state);
          updateBGamma(state);
          
//          logTopNWordsPerTopic(state, 10);
        }
        // Y
        else if (variableName.toLowerCase().equals("y")){ 
          // maximize class labels independently (FAST)
//          state.includeMetadataSupervision = false;
          logger.debug("maximizing inferred labels Y iteration "+iteration);
          cumulativeNumChanges += maximizeY(state); // set inferred label values
          // update hyperparam
          updateBGamma(state);
        }
        // Z
        else if (variableName.toLowerCase().equals("z")){
          // maximize topics independently (FAST)
//          state.includeMetadataSupervision = false;
          logger.debug("maximizing topic assignments Z iteration "+iteration);
          cumulativeNumChanges += maximizeZ(state); // set topic assignments
          // tune hyperparams
          updateBTheta(state);
          updateBPhi(state);
        }
        // B
        else if (variableName.toLowerCase().equals("b")){
          // maximize log linear model independently 
          logger.debug("maximizing regression vector B iteration "+iteration);
          maximizeB(state);
        }
        else{
          throw new IllegalArgumentException("unknown variable name "+variableName);
        }
        return (double)cumulativeNumChanges;
      }
      @Override
      public DatasetLabeler getIntermediateLabeler() {
        return new DatasetLabeler() {
          @Override
          public Predictions label(Dataset trainingInstances, Dataset heldoutInstances) {
            return predict(state, predictSingleLastSample, trainingInstances, heldoutInstances, rnd);
          }
        };
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
      
      // create a training set by adding each instance with its features 
      // equal to "zbar" (normalized topic vector) and it's label equal to the current y
      // 
      // note: (ideally we would add each instance k times, each with its weight according to the sampler distribution
      // --this would make our EM a little softer)
      InstanceList trainingSet = new InstanceList(dataAlphabet, labelAlphabet);
      for (int doc=0; doc<s.numDocuments; doc++){
        // Important optimization: skip instances that have 0 annotations, since those y's analytically 
        // integrate out of the model and we aren't sampling them.
        if (s.docAnnotationCounts[doc]>0){
          // note: do even softer-EM by maintaining distributions over topic vectors?
          trainingSet.add(instanceForTopicCounts(s.perDocumentCountOfTopic[doc], null, s.docSizes[doc], s.priors.getBTheta(), s.y[doc]));
        }
      }
      
      // train
      MaxEntTrainer trainer = new MaxEntTrainer(s.maxent);
      trainer.setGaussianPriorVariance(s.priors.getEtaVariance());
      return trainer.train(trainingSet);
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

    public static double[] zbar(double[] topicCounts, double docSize, double priorBias){
      double[] zbar = DoubleArrays.subtract(topicCounts, priorBias); // remove bias of priors added to counts
      DoubleArrays.divideToSelf(zbar, docSize); // account for word removed
      return DoubleArrays.extend(zbar, 1);
    }

    public static double getEtaElement(int documentClass, int topic, State s) {
      int rowPos = documentClass*(s.numTopics+1);
      int pos = rowPos + topic;
      return s.maxent.getParameters()[pos];
    }

    /**
     * Get the weights w from the underlying log-linear model for a given class as a double[topic].
     * The final entry of each row (i.e., w[class][numTopics]) is the class bias weight.
     * b
     * Note that topics are playing the role of features here
     */
    public static double[] getEtaRow(int documentClass, State s) {
      int length = s.numTopics+1;
      double[] parameters = new double[length];
      int srcPos = documentClass*(length);
      System.arraycopy(s.maxent.getParameters(), srcPos, parameters, 0, parameters.length);
      return parameters;
    }
    
    /**
     * Get the weights w from the underlying log-linear model as a double[class][feature].
     * The final entry of each row (i.e., w[class][numFeatures]) is the class bias weight.
     */
    public static double[][] getEta(State s){
      Preconditions.checkState(s.maxent.getNumParameters()==s.numClasses*s.numTopics+s.numClasses);
      Preconditions.checkState(s.maxent.getDefaultFeatureIndex()==s.numTopics);
      double[][] maxLambda = new double[s.numClasses][]; // +1 accounts for class bias term
      for (int k=0; k<s.numClasses; k++){
        maxLambda[k] = getEtaRow(k, s);
      }
      return maxLambda;
    }

  }
  
  


  //////////////////////////////////////////////
  // Model Code
  //////////////////////////////////////////////
  private State state;
  
  public ConfusedSLDADiscreteModel(State state) {
    this.state=state;
  }

  public Dataset getData() {
    return this.state.data;
  }
  
  public int[] getY(){
    return this.state.y.clone();
  }
  
  public int[] getMarginalY(){
    return this.state.yMarginals.argmax();
  }

  public int[][] getZ(){
    return Matrices.clone(this.state.z);
  }
  
  public State getState(){
    return this.state;
  }

  

  

  public Predictions predict(Dataset trainingInstances, boolean predictSingleLastSample, Dataset heldoutInstances, RandomGenerator rnd){
    return predict(state, predictSingleLastSample, trainingInstances, heldoutInstances, rnd);
  }

  public static Predictions predict(State state, boolean predictSingleLastSample, Dataset trainingInstances, Dataset heldoutInstances, RandomGenerator rnd){
    // em-derived labels
    List<Prediction> labeledPredictions = Lists.newArrayList();
    List<Prediction> unlabeledPredictions = Lists.newArrayList();
    // the clause (IntArrays.sum(state.yMarginals.values(0))==0) checks whether yMarginals has been initialized.  
    // This will be false when model parameters were set most recently via maximization, in which case we want to use raw y values. 
    int[] inferredYValues = predictSingleLastSample || IntArrays.sum(state.yMarginals.values(0))==0? state.y: state.yMarginals.argmax();
    for (DatasetInstance inst: state.data){
      
      int index = state.instanceIndices.get(inst.getInfo().getSource());
      
      if (inst.getInfo().getNumAnnotations()>0){
        // annotated
        labeledPredictions.add(new BasicPrediction(inferredYValues[index], inst));
      }
      else{
        // unannotated - these y's were ignored during inference 
        // since they are a deterministic function of the z's and 
        // eta. Calculate them now
        Instance zbar = MalletInterface.instanceForTopicCounts(state.perDocumentCountOfTopic[index], null, state.docSizes[index], state.priors.getBTheta(), null);
        state.maxent.getClassificationScores(zbar, state.logisticClassScores);
        unlabeledPredictions.add(new BasicPrediction(DoubleArrays.argMax(state.logisticClassScores), inst));
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
    double[][][] annotatorConfusions = Matrices.clone(s.perAnnotatorCountOfYAndA);
    Matrices.normalizeRowsToSelf(annotatorConfusions);
    return CrowdsourcingUtils.getAccuracies(ybias, annotatorConfusions);
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
    int documentClass = s.y[doc];
    
    // this is a significant convergence optimization: if this document has 0
    // annotations, then this document's y variable analytically integrates out
    // (in the same way as unobserved features in naive bayes). Therefore this becomes 
    // a vanilla lda sample
    boolean originalMetadataSupervisionSetting = s.includeMetadataSupervision;
    if (s.docAnnotationCounts[doc]==0){
      s.includeMetadataSupervision=false;
    }
    
    if (s.includeMetadataSupervision){
      // precompute and cache metadata scores for each topic 
      // (they are invariant wrt words)
      for (int t=0; t<s.numTopics; t++){
        double numerator = Math.exp(MalletInterface.getEtaElement(documentClass,t,s)/(s.docSizes[doc]));
        // extend zbar with a bias term on the end always set to 1 (see MalletInterface.getParameters())
        double[] zzbar = MalletInterface.zbar(s.perDocumentCountOfTopic[doc], s.docSizes[doc], s.priors.getBTheta()); 
        double denominator = 0;
        for (int c=0; c<s.numClasses; c++){
          double dotprod = (MalletInterface.getEtaElement(c,t,s)/s.docSizes[doc]) + DoubleArrays.dotProduct(MalletInterface.getEtaRow(c,s),zzbar);
          denominator += Math.exp(dotprod / s.docSizes[doc]); 
        }
        
        s.cachedMetadataScores[t] = numerator/denominator;
      }
    }
    
    // do update
    for (int word=0; word<docsize; word++){
      numChanges += updateZDocWord(s, doc, word, rnd);
    }

    s.includeMetadataSupervision = originalMetadataSupervisionSetting; // restore original metadata setting (in case we disabled it due to 0 annotations on this doc)
    
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
        // precomputed at the document level.
        s.zCoeffs[t] *= s.cachedMetadataScores[t];
        
      }
      
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
    s.yMarginals.increment(s.y); // track marginal posterior
    return numChanges;
  }

  /**
   * If rnd is null, then it takes the assignment that maximizes 
   * the joint. Otherwise, samples. 
   */
  public static int updateYDoc(State s, int doc, RandomGenerator rnd){
    // as an optimization, don't bother sampling labels for unannotated 
    // documents--they are deterministic functions of zbar and eta. We 
    // can calculate that later during the prediction phase. During inference 
    // we can simply pretend these y's don't exist (they integrate out analytically
    // in the same way that missing data does in naive bayes). 
    if (s.docAnnotationCounts[doc]==0){
      return 0;  
    }
    
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
  private static double[] staticEtaParameters;
  public static void maximizeB(State s){
    if (STATIC_ETA){
      
      if (staticEtaParameters==null){
        // initialize the maxent model (don't know how to do this without going through training once)
        s.maxent = MalletInterface.logisticRegressionFromTopicToClass(s);
        
        staticEtaParameters = new double[s.numClasses*(s.numTopics+1)];
        for (int k=0; k<s.numClasses; k++){
          for (int t=0; t<s.numTopics; t++){
            staticEtaParameters[k*(s.numTopics+1)+t] = (k==t)? 1: 0;
          }
          // staticEtaMapping[(k*s.numTopics+1)+s.numTopics] = 0; // bias term remains 0
        }
      // instead of regular maximization we will force a 1-1 mapping between 
      // classes and the first K topics. The rest of the topics are allowed to 
      // float in an entirely unsupervised manner to handle nuisance words/topics. 
      s.maxent.setParameters(staticEtaParameters);
      }
    }
    else{
      // retrain a maxent model between per-document topic vectors 
      // and class labels.
      s.maxent = MalletInterface.logisticRegressionFromTopicToClass(s);
    }
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
      if (s.docAnnotationCounts[doc]>0){ // optimization: ignore unannotated documents (they integrate out)
        // get prob of each class assignment
        Instance inst = MalletInterface.instanceForTopicCounts(s.perDocumentCountOfTopic[doc], null, s.docSizes[doc], s.priors.getBTheta(), null);
        s.maxent.getClassificationScores(inst, s.logisticClassScores);
        // add log prob of current class assignment
        logTotal += Math.log(s.logisticClassScores[s.y[doc]]);
      }
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
  public static void logTopNWordsPerTopic(State s, int n){
    int topic = 0;
    for (double[] vocab: s.perTopicCountOfVocab){
      if (STATIC_ETA && topic<s.numClasses){
        logger.info("topic "+topic+" corresponds to class "+s.data.getInfo().getLabelIndexer().get(topic));
      }
      logger.info("Top "+n+" words for topic "+topic+":");
      for (int topIndex: DoubleArrays.argMaxList(n, vocab)){
        logger.info("\t"+s.data.getInfo().getFeatureIndexer().get(topIndex)+"="+vocab[topIndex]);
      }
      topic++;
    }
  }

  


  /////////////////////////////////////////////////
  // Hyperparameter learning  
  /////////////////////////////////////////////////
  public static void updateBTheta(State s) {
    logger.debug("optimizing btheta in light of most recent topic assignments");
    subtractPriorsFromCounts(s);
    double oldValue = s.priors.getBTheta();
    IterativeOptimizer optimizer = new IterativeOptimizer(ConvergenceCheckers.relativePercentChange(PriorSpecification.HYPERPARAM_LEARNING_CONVERGENCE_THRESHOLD));
    SymmetricDirichletMultinomialMatrixMAPOptimizable o = SymmetricDirichletMultinomialMatrixMAPOptimizable.newOptimizable(s.perDocumentCountOfTopic,2,2);
    ValueAndObject<Double> optimum = optimizer.optimize(o, ReturnType.HIGHEST, true, oldValue);
    s.priors.setBTheta(optimum.getObject());
    addPriorsToCounts(s);
    logger.info("new btheta="+s.priors.getBTheta()+" old btheta="+oldValue);
  }

  public static void updateBPhi(State s) {
    logger.debug("optimizing bphi in light of most recent topic assignments");
    subtractPriorsFromCounts(s);
    double oldValue = s.priors.getBPhi();
    IterativeOptimizer optimizer = new IterativeOptimizer(ConvergenceCheckers.relativePercentChange(PriorSpecification.HYPERPARAM_LEARNING_CONVERGENCE_THRESHOLD));
    // TODO: we could fit each class symmetric dirichlet separately. Alternatively, we could fit each individual parameter (maybe w/ gamma prior)
    SymmetricDirichletMultinomialMatrixMAPOptimizable o = SymmetricDirichletMultinomialMatrixMAPOptimizable.newOptimizable(s.perTopicCountOfVocab,2,2);
    ValueAndObject<Double> optimum = optimizer.optimize(o, ReturnType.HIGHEST, true, oldValue);
    s.priors.setBPhi(optimum.getObject());
    addPriorsToCounts(s);
    logger.info("new bphi="+s.priors.getBPhi()+" old bphi="+oldValue);
  }

  public static void updateBGamma(State s) {
    logger.debug("optimizing bgamma in light of most recent document labels");
    subtractPriorsFromCounts(s);
    Pair<Double,Double> oldValue = Pair.of(s.deltas[0][0][0], s.deltas[0][0][1]);
    IterativeOptimizer optimizer = new IterativeOptimizer(ConvergenceCheckers.relativePercentChange(PriorSpecification.HYPERPARAM_LEARNING_CONVERGENCE_THRESHOLD));
    SymmetricDirichletMultinomialDiagonalMatrixMAPOptimizable o = SymmetricDirichletMultinomialDiagonalMatrixMAPOptimizable.newOptimizable(s.perAnnotatorCountOfYAndA, 2, 2);
    ValueAndObject<Pair<Double,Double>> optimum = optimizer.optimize(o, ReturnType.HIGHEST, true, oldValue);
    double newDiag = optimum.getObject().getFirst();
    double newOffDiag = optimum.getObject().getSecond();
    for (int j=0; j<s.numAnnotators; j++){
    	for (int k=0; k<s.numClasses; k++){
    		for (int kprime=0; kprime<s.numClasses; kprime++){
    			s.deltas[j][k][kprime] = (k==kprime)? newDiag: newOffDiag;
    		}
    	}
    }
    addPriorsToCounts(s);
    // setting priors allows driver class to report settled-on values
    double newCGamma = newDiag + (s.numClasses-1)*newOffDiag;
    s.priors.setBGamma(newDiag/newCGamma);
    s.priors.setCGamma(newCGamma);
    logger.info("new bgamma="+optimum.getObject()+" old bgamma="+oldValue);
  }
  
  
}
