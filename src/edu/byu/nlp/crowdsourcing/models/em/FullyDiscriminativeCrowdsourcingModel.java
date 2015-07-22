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
package edu.byu.nlp.crowdsourcing.models.em;

import java.util.List;

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
import cc.mallet.types.LabelVector;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;

import edu.byu.nlp.classify.eval.BasicPrediction;
import edu.byu.nlp.classify.eval.Prediction;
import edu.byu.nlp.classify.eval.Predictions;
import edu.byu.nlp.crowdsourcing.CrowdsourcingUtils;
import edu.byu.nlp.crowdsourcing.PriorSpecification;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.DatasetInstance;
import edu.byu.nlp.data.types.SparseFeatureVector;
import edu.byu.nlp.data.types.SparseFeatureVector.EntryVisitor;
import edu.byu.nlp.dataset.Datasets;
import edu.byu.nlp.math.AbstractRealMatrixPreservingVisitor;
import edu.byu.nlp.util.DoubleArrays;
import edu.byu.nlp.util.Enumeration;
import edu.byu.nlp.util.IntArrays;
import edu.byu.nlp.util.Iterables2;
import edu.byu.nlp.util.Matrices;

/**
 * @author pfelt
 *
 */
public class FullyDiscriminativeCrowdsourcingModel {

  private static final double CONVERGENCE_THRESHOLD = 0.01;
  private static final int MAX_ITERATIONS = 30;
  private static final Logger logger = LoggerFactory.getLogger(FullyDiscriminativeCrowdsourcingModel.class);
  
  private MaxEnt maxent;
  private double[][] softlabels;
  private Instance[] instances;
  private int[][][] a;
  private List<DatasetInstance> externalInstances;
  private double logJoint;
  
  public FullyDiscriminativeCrowdsourcingModel(MaxEnt maxent, double[][] softlabels, Instance[] instances, 
      List<DatasetInstance> externalInstances, int[][][] a, double logJoint){
    this.maxent=maxent;
    this.softlabels=softlabels;
    this.instances=instances;
    this.externalInstances=externalInstances;
    this.a=a;
    this.logJoint=logJoint;
  }
  
  public static class ModelBuilder {
    private Dataset data;
    private Instance[] instances;
    private int a[][][];
    private MaxEnt maxent;
    private List<DatasetInstance> externalInstances;
    private PriorSpecification priors;
    private boolean semisupervised;
    
    public ModelBuilder(Dataset data, PriorSpecification priors, boolean semisupervised){
      this.data=data;
      this.priors=priors;
      this.semisupervised=semisupervised;
    }
    
    
    /**
     *  softlabels = soft majority vote
     *  maxent = trained on softlabels
     */
    private double[][] majorityVoteSoftLabels(){
      double[][] softlabels = Matrices.of(0, a.length, data.getInfo().getNumClasses());
      
      // initialize labels with soft majority vote
      for (int i=0; i<a.length; i++){
        for (int j=0; j<priors.getNumAnnotators(); j++){
          for (int k=0; k<data.getInfo().getNumClasses(); k++){
            softlabels[i][k] += a[i][j][k]; 
          }
        }
        DoubleArrays.normalizeToSelf(softlabels[i]);
      }
      
      return softlabels;
    }
    
    /**
     * run EM to train raykar et al. model
     */
    public FullyDiscriminativeCrowdsourcingModel build(){
      a = Datasets.compileDenseAnnotations(data);
      
      //////////////////
      // Init (with soft majority vote labels + maxent training)
      //////////////////
      logger.info("ignoring --training argument. Only EM is supported.");
      logger.info("Initializing EM (training model on majority vote labels)");
      double[][] softlabels = majorityVoteSoftLabels();

      DataAndAnnotationFeatureMalletMaxentTrainer trainer = DataAndAnnotationFeatureMalletMaxentTrainer.build(data);
      maxent = trainer.maxDataModel(softlabels, maxent);
      
      // convert the dataset to mallet
      externalInstances = Lists.newArrayListWithCapacity(data.getInfo().getNumDocuments());
      instances = new Instance[data.getInfo().getNumDocuments()];
      // convert instances
      int index=0;
      for (DatasetInstance inst: data){
        // convert feature vector
        instances[index] = DataAndAnnotationFeatureMalletMaxentTrainer.convert(maxent.getAlphabet(), inst);
        // remember the original instance
        externalInstances.add(inst);
        index++;
      }
      
      // EM training
      double previousValue = -Double.MAX_VALUE, value = -Double.MAX_VALUE;
      int iterations = 0;
      do{
        //////////////////
        // E - step (set softlabels)
        //////////////////
        logger.info(iterations+" E-step (calculating expected 'soft' labels)");
        softlabels = expectedSoftLabels(maxent, a, instances, semisupervised);
        
        //////////////////
        // M - step (maximize parameters with the soft labels determined during the E step)
        //////////////////
        logger.info(iterations+" M-step (maximizing parameters based on expected 'soft' labels)");
        maxent = trainer.maxDataModel(softlabels, maxent);
        
        previousValue = value;
        value = logJoint(softlabels,maxent,instances,a);

        logger.info(iterations+" EM value "+value+" (improvement of "+(value-previousValue)+")");
        iterations++;
      
      } while(iterations < MAX_ITERATIONS && value - previousValue > CONVERGENCE_THRESHOLD);
      
      return new FullyDiscriminativeCrowdsourcingModel(maxent, softlabels, instances, externalInstances, a, value);
    }

    private static double logJoint(double[][] softlabels, MaxEnt maxent, Instance[] instances, int[][][] a) {
      double logJoint = 0;
      for (int i=0; i<instances.length; i++){
        LabelVector labels = maxent.classify(instances[i]).getLabelVector();
        for (int k=0; k<softlabels[i].length; k++){
          // p(y|data)
          logJoint = Math.log(labels.value(k));
        }
      }
      // data
      return logJoint;
    }

    private static double[][] expectedSoftLabels(MaxEnt maxent, int[][][] a, Instance[] instances, boolean includeUnannotated) {
      LabelAlphabet labelAlphabet = maxent.getLabelAlphabet();
      
      double softlabels[][] = new double[instances.length][labelAlphabet.size()];
      for (int i=0; i<instances.length; i++){
        double[] dataEvidence = maxent.classify(instances[i]).getLabelVector().getValues();
        for (int k=0; k<dataEvidence.length; k++){
          // p(y|x) 
          softlabels[i][k] = dataEvidence[k]; 
        }
        DoubleArrays.normalizeToSelf(softlabels[i]);
      }
      return softlabels;
    }

  }
  
  private boolean isAnnotated(int a[][]){
    for (int j=0; j<a.length; j++){
      if (IntArrays.sum(a[j])>0){
        return true;
      }
    }
    return false;
  }
  
  private static List<Integer> softLabels2ExternalPredictions(double[] softlabels, Alphabet targetAlphabet){
    List<Integer> extPredictions = Lists.newArrayList();
    for (int k: DoubleArrays.argMaxList(-1, softlabels)){
      extPredictions.add((Integer) targetAlphabet.lookupObject(k));
    }
    return extPredictions;
  }
  
  private static List<Integer> unannotatedPrediction(DatasetInstance inst, Alphabet dataAlphabet, LabelAlphabet targetAlphabet, MaxEnt maxent){
    Instance convertedTestInstances = DataAndAnnotationFeatureMalletMaxentTrainer.convert(dataAlphabet, inst);
    double[] testSoftLabels = ModelBuilder.expectedSoftLabels(maxent, null, new Instance[]{convertedTestInstances},true)[0];
    return softLabels2ExternalPredictions(testSoftLabels, targetAlphabet);
  }
  
  public Predictions predict(
      Dataset trainingInstances, 
      Dataset heldoutInstances){
    
    // em-derived labels
    List<Prediction> labeledPredictions = Lists.newArrayList();
    List<Prediction> unlabeledPredictions = Lists.newArrayList();
    for (int i=0; i<instances.length; i++){
      List<Integer> pi = softLabels2ExternalPredictions(softlabels[i], maxent.getLabelAlphabet());
      if (isAnnotated(a[i])){
        labeledPredictions.add(new BasicPrediction(pi, externalInstances.get(i)));
      }
      else{
        // make a prediction based soley on the data model
        List<Integer> predictionIndices = unannotatedPrediction(externalInstances.get(i), maxent.getAlphabet(), maxent.getLabelAlphabet(), maxent);
        unlabeledPredictions.add(new BasicPrediction(predictionIndices, externalInstances.get(i)));
      }
    }
    
    // generalization labels
    List<Prediction> heldoutPredictions = Lists.newArrayList();
    for (DatasetInstance inst: heldoutInstances){
      List<Integer> predictionIndices = unannotatedPrediction(inst, maxent.getAlphabet(), maxent.getLabelAlphabet(), maxent);
      heldoutPredictions.add(new BasicPrediction(predictionIndices, inst));
    }
    
    int numAnnotators = a[0].length;
    int numClasses = a[0][0].length;
    double[] annotatorAccuracies = new double[numAnnotators];
    double[][][] annotatorConfusionMatrices = new double[numAnnotators][numClasses][numClasses];
    double machineAccuracy = -1;
    double[][] machineConfusionMatrix = new double[numClasses][numClasses];
    return new Predictions(labeledPredictions, unlabeledPredictions, heldoutPredictions, annotatorAccuracies, annotatorConfusionMatrices, machineAccuracy, machineConfusionMatrix, logJoint);
  }
  
  /**
   * TODO: is there a better way to do this? (how should we estimate class bias (theta) in this model?)
   */
  public static double[] getAnnotatorAccuracies(double[][][] gammas) {
    int num_annotators = gammas.length;
    int num_classes = gammas[0].length;
    double[] accuracies = new double[num_annotators];
    for (int j = 0; j < gammas.length; j++) {
      // assume a uniform distribution over classes
      double[] theta = DoubleArrays.of(1.0/num_classes, num_classes);
      CrowdsourcingUtils.getAccuracy(theta, gammas[j]);
    }
    return accuracies;
  }

  public static double getAccuracy(double[] theta, double[][] confusionProbs) {
    double acc = 0.0;
    for (int k = 0; k < confusionProbs.length; k++) {
      acc += theta[k] * confusionProbs[k][k];
    }
    return acc;
  }
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  /**
   * This helper class is a copy of MalletMaxentTrainer altered so as to 
   * operate on feature vectors composed not just of document features, but 
   * also annotation features concatenated onto the end.
   */
  public static class DataAndAnnotationFeatureMalletMaxentTrainer{

    private cc.mallet.types.Instance[] instances;
    private Alphabet dataAlphabet;
    private LabelAlphabet targetAlphabet;
    private List<DatasetInstance> externalInstances;

    private DataAndAnnotationFeatureMalletMaxentTrainer(){}
    
    /**
     * The trainer takes care of converting the dataset into types that 
     * mallet can work with. Labels are not converted here, but 
     * are passed in separately during training.
     */
    public static DataAndAnnotationFeatureMalletMaxentTrainer build(Dataset data){

      final DataAndAnnotationFeatureMalletMaxentTrainer trainer = new DataAndAnnotationFeatureMalletMaxentTrainer();
      
      trainer.externalInstances = Lists.newArrayListWithCapacity(data.getInfo().getNumDocuments());
      trainer.instances = new cc.mallet.types.Instance[data.getInfo().getNumDocuments()];
      trainer.dataAlphabet = new Alphabet();
      trainer.dataAlphabet.startGrowth();
      trainer.targetAlphabet = new LabelAlphabet();
      trainer.targetAlphabet.startGrowth();
      

      // create identity mallet data feature alphabet (so that our label indices correspond exactly to theirs)
      for (int f=0; f<data.getInfo().getNumFeatures(); f++){
        trainer.dataAlphabet.lookupIndex(dataFeature(f),true);
      }
      // also add annotation features to data alphabet
      for (int j=0; j<data.getInfo().getNumAnnotators(); j++){
        for (int k=0; k<data.getInfo().getNumClasses(); k++){
          trainer.dataAlphabet.lookupIndex(annotationFeature(j, k),true);
        }
      }
      trainer.dataAlphabet.stopGrowth();
      
      // create identity mallet label alphabet (so that our label indices correspond exactly to theirs)
      for (int l=0; l<data.getInfo().getNumClasses(); l++){
        trainer.targetAlphabet.lookupLabel(l,true);
      }
      trainer.targetAlphabet.stopGrowth();
      
      // alphabet sanity check #1 (make sure mallet alphabets return identity mappings
      Preconditions.checkState(trainer.dataAlphabet.size()==data.getInfo().getNumFeatures()+(data.getInfo().getNumAnnotators()*data.getInfo().getNumClasses()));
      Preconditions.checkState(trainer.targetAlphabet.size()==data.getInfo().getNumClasses());
      for (int f=0; f<data.getInfo().getNumFeatures(); f++){
        Preconditions.checkState(trainer.dataAlphabet.lookupIndex(dataFeature(f))==f);
        Preconditions.checkState(trainer.dataAlphabet.lookupObject(f).equals(dataFeature(f)));
      }
      for (int j=0; j<data.getInfo().getNumAnnotators(); j++){
        for (int k=0; k<data.getInfo().getNumClasses(); k++){
          int index = data.getInfo().getNumFeatures() + j*data.getInfo().getNumClasses() + k;
          Preconditions.checkState(trainer.dataAlphabet.lookupIndex(annotationFeature(j,k))==index);
          Preconditions.checkState(trainer.dataAlphabet.lookupObject(index).equals(annotationFeature(j,k)));
        }
      }
      for (int f=0; f<trainer.targetAlphabet.size(); f++){
        Preconditions.checkState(trainer.targetAlphabet.lookupIndex(f)==f);
        Preconditions.checkState(trainer.targetAlphabet.lookupLabel(f).getIndex()==f);
        Preconditions.checkState(trainer.targetAlphabet.lookupObject(f).equals(new Integer(f)));
      }
      
      // alphabet sanity check #2 (make sure every instance in the data has valid mallet data and label alphabet entries) 
      // this would only fail if our indexers were not working properly (there was a gap somewhere among 
      // the existing indices)
      for (DatasetInstance inst: data){
        // visit the data (to make sure all features and labels were added correctly)
        inst.asFeatureVector().visitSparseEntries(new EntryVisitor() {
          @Override
          public void visitEntry(int index, double value) {
            Preconditions.checkState(trainer.dataAlphabet.lookupIndex(dataFeature(index),false)>=0);
          }
        });
        if (inst.hasLabel()){ // ignore null label but use hidden labels to help ensure good alphabet coverage (no cheating here)
          Preconditions.checkState(trainer.targetAlphabet.lookupIndex(inst.getLabel())>=0);
        }
      }
      
      // convert each dataset instance to a mallet instance 
      for (Enumeration<DatasetInstance> item: Iterables2.enumerate(data)){
        // convert feature vector
        trainer.instances[item.getIndex()] = convert(trainer.dataAlphabet, item.getElement());
        // remember the original instance
        trainer.externalInstances.add(item.getElement());
      }

      return trainer;
    }
    
    /**
     * Train a log-linear model using the given soft labels (must match the 
     * dataset this trainer was build on).
     */
    public MaxEnt maxDataModel(double[][] softlabels, MaxEnt previousModel){
      // create a training set by adding each instance K times, each weighted by softlabels
      InstanceList trainingSet = new InstanceList(dataAlphabet, targetAlphabet);
      for (int i=0; i<instances.length; i++){
        for (int k=0; k<targetAlphabet.size(); k++){
          if (!Double.isNaN(softlabels[i][k])){ // ignore nans (instances with no annotations)
            // give this instance label k with weight softlabels[k]
            cc.mallet.types.Instance inst = instances[i].shallowCopy();
            inst.setTarget(targetAlphabet.lookupLabel(k));
            trainingSet.add(inst, softlabels[i][k]);
          }
        }
      }
      // train
      return new MaxEntTrainer(previousModel).train(trainingSet);
    }
    
    /**
     * Convert a single DatasetInstance to a mallet instance with no label
     * 
     * This is similar to MalletMaxentTrainer.convert(). However, in addition to 
     * porting an instance's feature vector over, this method additionally concatenates 
     * a set of features corresponding to annotation counts for each annotator.
     */
    public static cc.mallet.types.Instance convert(final Alphabet dataAlphabet, DatasetInstance inst){
      SparseFeatureVector features = inst.asFeatureVector();
      String source = inst.getInfo().getRawSource();
      
      final List<Integer> featureIndices = Lists.newArrayList();
      final List<Double> featureValues = Lists.newArrayList();
      
      // data features
      features.visitSparseEntries(new EntryVisitor() {
        @Override
        public void visitEntry(int index, double value) {
          int featureIndex = dataAlphabet.lookupIndex(dataFeature(index));
          if (featureIndex>=0){ // ignore unknown features (for generalization)
            featureIndices.add(dataAlphabet.lookupIndex(dataFeature(index)));
            featureValues.add(value);
          }
        }
      });
      
      // annotation features
      inst.getAnnotations().getLabelAnnotations().walkInOptimizedOrder(new AbstractRealMatrixPreservingVisitor() {
        @Override
        public void visit(int annotator, int annotationValue, double count) {
          featureIndices.add(dataAlphabet.lookupIndex(annotationFeature(annotator, annotationValue)));
          featureValues.add(count);
        }
      });
      
      // add to trainingData
      FeatureVector malletFV = new FeatureVector(dataAlphabet, IntArrays.fromList(featureIndices), DoubleArrays.fromList(featureValues));
      String name = source;
      Label target = null; // no label for now
      
      // only convert instances with non-null labels
      return new cc.mallet.types.Instance(malletFV, target, name, source);
    }

    public static String annotationFeature(int annotator, int annotationValue){
      return "ann="+annotator+" val="+annotationValue;
    }

    public static String dataFeature(int feature){
      return "feat="+feature;
    }
    
    
  } // end of trainer
  
  
  
  
  
  
  
}
