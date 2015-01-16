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
import java.util.logging.Logger;

import cc.mallet.classify.MaxEnt;
import cc.mallet.classify.MaxEntTrainer;
import cc.mallet.types.Alphabet;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.LabelAlphabet;
import cc.mallet.types.LabelVector;

import com.google.common.collect.Lists;

import edu.byu.nlp.classify.MalletMaxentTrainer;
import edu.byu.nlp.classify.eval.BasicPrediction;
import edu.byu.nlp.classify.eval.Prediction;
import edu.byu.nlp.classify.eval.Predictions;
import edu.byu.nlp.crowdsourcing.CrowdsourcingUtils;
import edu.byu.nlp.crowdsourcing.PriorSpecification;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.DatasetInstance;
import edu.byu.nlp.data.types.SparseFeatureVector.EntryVisitor;
import edu.byu.nlp.dataset.Datasets;
import edu.byu.nlp.math.Math2;
import edu.byu.nlp.util.DoubleArrays;
import edu.byu.nlp.util.IntArrays;
import edu.byu.nlp.util.Matrices;

/**
 * @author pfelt
 *
 */
public class RaykarModel {

  private static final Logger logger = Logger
      .getLogger(RaykarModel.class.getName());
  
  private MaxEnt maxent;
  private double[][][] gammas;
  private double[][] softlabels;
  private Instance[] instances;
  private int[][][] a;
  private Alphabet dataAlphabet;
  private LabelAlphabet targetAlphabet;
  private List<DatasetInstance> externalInstances;
  private double logJoint;
  
  public RaykarModel(MaxEnt maxent, double[][][] gammas, double[][] softlabels, Instance[] instances, 
      List<DatasetInstance> externalInstances, int[][][] a, Alphabet dataAlphabet, LabelAlphabet targetAlphabet, double logJoint){
    this.maxent=maxent;
    this.gammas=gammas;
    this.softlabels=softlabels;
    this.dataAlphabet=dataAlphabet;
    this.targetAlphabet=targetAlphabet;
    this.instances=instances;
    this.externalInstances=externalInstances;
    this.a=a;
    this.logJoint=logJoint;
  }
  
  public static class ModelBuilder {
    private Dataset data;
    private Instance[] instances;
    private Alphabet dataAlphabet;
    private LabelAlphabet targetAlphabet;
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
    
    private void initTrainingSet(){
//      List<DatasetInstance> allInstances = Lists.newArrayList(data);
      externalInstances = Lists.newArrayListWithCapacity(data.getInfo().getNumDocuments());
      instances = new Instance[data.getInfo().getNumDocuments()];
      dataAlphabet = new Alphabet();
      dataAlphabet.startGrowth();
      targetAlphabet = new LabelAlphabet();
      targetAlphabet.startGrowth();
      
      // mallet alphabets
      for (DatasetInstance inst: data){
        // visit the data
        inst.asFeatureVector().visitSparseEntries(new EntryVisitor() {
          @Override
          public void visitEntry(int index, double value) {
            dataAlphabet.lookupIndex(index, true);
          }
        });
        if (inst.hasLabel()){ // ignore null label (but use hidden labels for label coverage--no cheating here)
          targetAlphabet.lookupLabel(inst.getLabel(), true);
        }
      }
      dataAlphabet.stopGrowth();
      targetAlphabet.stopGrowth();
      
      // convert instances
      int index=0;
      for (DatasetInstance inst: data){
        // convert feature vector
        instances[index] = MalletMaxentTrainer.convert(dataAlphabet, inst);
        // remember the original instance
        externalInstances.add(inst);
        index++;
      }

      // convert annotations
      int[][][] rawAnnotations = Datasets.compileDenseAnnotations(data);
      a = new int[instances.length][priors.getNumAnnotators()][data.getInfo().getNumClasses()];
      for (int i=0; i<rawAnnotations.length; i++){
        for (int j=0; j<priors.getNumAnnotators(); j++){
          for (int k=0; k<data.getInfo().getNumClasses(); k++){
            a[i][j][targetAlphabet.lookupIndex(k)] = rawAnnotations[i][j][k];
          }
        }
      }
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
    public RaykarModel build(){
      // convert the dataset to mallet
      initTrainingSet();

      //////////////////
      // Init (with soft majority vote labels + maxent training)
      //////////////////
      logger.info("Initializing EM (training model on majority vote labels)");
      double[][] softlabels = majorityVoteSoftLabels();
      double[][][] gammas = maxGammas(softlabels, a, priors.getNumAnnotators(), data.getInfo().getNumClasses(), priors);
      maxent = maxDataModel(softlabels,maxent,instances,dataAlphabet,targetAlphabet);
      
      // EM training
      double previousValue = Double.MIN_VALUE;
      double value = Double.MIN_VALUE;
      int iterations = 0;
      do{
        //////////////////
        // E - step (set softlabels)
        //////////////////
        logger.info(iterations+" E-step (calculating expected 'soft' labels)");
        softlabels = expectedSoftLabels(gammas, maxent, a, instances, semisupervised);
        
        //////////////////
        // M - step (maximize parameters with the soft labels determined during the E step)
        //////////////////
        logger.info(iterations+" M-step (maximizing parameters based on expected 'soft' labels)");
        gammas = maxGammas(softlabels, a, priors.getNumAnnotators(), data.getInfo().getNumClasses(), priors);
        maxent = maxDataModel(softlabels,maxent,instances,dataAlphabet,targetAlphabet);
        
        previousValue = value;
        value = logJoint(priors,gammas,softlabels,maxent,instances,a);

        logger.info(iterations+" EM value "+value+" (improvement of "+(value-previousValue)+")");
        iterations++;
      
      } while(iterations < 100 && value - previousValue > 1e-6);
      
      return new RaykarModel(maxent, gammas, softlabels, instances, externalInstances, a, dataAlphabet, targetAlphabet, value);
    }

    private static double logJoint(PriorSpecification priors, double[][][] gammas, double[][] softlabels, MaxEnt maxent, Instance[] instances, int[][][] a) {
      double logJoint = 0;
      double[][] gammaprior = Matrices.of(0, gammas[0].length, gammas[0].length);
      // gammas
      for (int j=0; j<gammas.length; j++){
        CrowdsourcingUtils.initializeConfusionMatrixWithPrior(gammaprior, priors.getBGamma(j), priors.getCGamma());
        for (int i=0; i<instances.length; i++){
          LabelVector labels = maxent.classify(instances[i]).getLabelVector();
          for (int k=0; k<gammas[j].length; k++){
            double logComponent = 0;
            // p(y|data)
            logComponent += Math.log(labels.value(k));
            // p(a|y)
            for (int kprime=0; kprime<gammas[j].length; kprime++){
              logComponent += Math.log(gammaprior[k][kprime] + gammas[j][k][kprime]) * a[i][j][kprime];
            }
            
            logJoint = Math2.logAddSloppy(logJoint, logComponent);
          }
        }
      }
      // data
      return logJoint;
    }

    private static double[] annotationEvidence(int[][] a, double[][][] gammas){
      int numLabels = gammas[0].length; // assume there is at least 1 annotator
      
      double[] evd = DoubleArrays.of(1, numLabels); 
      for (int j=0; j<gammas.length; j++){
        for (int k=0; k<numLabels; k++){
          for (int kprime=0; kprime<numLabels; kprime++){
            evd[k] *= Math.pow(gammas[j][k][kprime], a[j][kprime]);
          }
        }
      }
      
      return evd;
    }
    
    private static double[][] expectedSoftLabels(double[][][] gammas, MaxEnt maxent, int[][][] a, Instance[] instances, boolean includeUnannotated) {
      LabelAlphabet labelAlphabet = maxent.getLabelAlphabet();
      
      double softlabels[][] = new double[instances.length][labelAlphabet.size()];
      for (int i=0; i<instances.length; i++){
        double[] dataEvidence = maxent.classify(instances[i]).getLabelVector().getValues();
        double[] annotationEvidence = (a==null)? DoubleArrays.of(1, labelAlphabet.size()): annotationEvidence(a[i], gammas);
        for (int k=0; k<maxent.getLabelAlphabet().size(); k++){
          // p(y|x) * p(a|y)
          if (!includeUnannotated && DoubleArrays.sum(annotationEvidence)==labelAlphabet.size()){
            softlabels[i][k] = Double.NaN;
          }
          else{
            softlabels[i][k] = dataEvidence[k] * annotationEvidence[k];
          }
        }
        DoubleArrays.normalizeToSelf(softlabels[i]);
      }
      return softlabels;
    }

    private static double[][][] maxGammas(double[][] softlabels, int[][][] a, int numAnnotators, int numLabels, PriorSpecification priors) {
      double[][][] gammas = new double[numAnnotators][numLabels][numLabels];
      // add priors
      for (int j=0; j<numAnnotators; j++){
        CrowdsourcingUtils.initializeConfusionMatrixWithPrior(gammas[j], priors.getBGamma(j), priors.getCGamma());
      }
      
      for (int j=0; j<numAnnotators; j++){
        // aggregate all the labels this annotator annotated
        for (int i=0; i<a.length; i++){
          for (int k=0; k<numLabels; k++){
            for (int kprime=0; kprime<numLabels; kprime++){
              if (!Double.isNaN(softlabels[i][k])){ // ignore nans (instances with no annotations)
                gammas[j][k][kprime] += softlabels[i][k] * a[i][j][kprime];
              }
            }
          }
        }
        // normalize
        Matrices.normalizeRowsToSelf(gammas[j]);
      }
      return gammas;
    }

    private static MaxEnt maxDataModel(double[][] softlabels, MaxEnt previousModel, Instance[] instances, Alphabet dataAlphabet, LabelAlphabet targetAlphabet){
      // create a training set by adding each instance K times, each weighted by softlabels
      InstanceList trainingSet = new InstanceList(dataAlphabet, targetAlphabet);
      for (int i=0; i<instances.length; i++){
        for (int k=0; k<targetAlphabet.size(); k++){
          if (!Double.isNaN(softlabels[i][k])){ // ignore nans (instances with no annotations)
            // give this instance label k with weight softlabels[k]
            Instance inst = instances[i].shallowCopy();
            inst.setTarget(targetAlphabet.lookupLabel(k));
            trainingSet.add(inst, softlabels[i][k]);
          }
        }
      }
      // train
      return new MaxEntTrainer(previousModel).train(trainingSet);
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
    Instance convertedTestInstances = MalletMaxentTrainer.convert(dataAlphabet, inst);
    double[] testSoftLabels = ModelBuilder.expectedSoftLabels(null, maxent, null, new Instance[]{convertedTestInstances},true)[0];
    return softLabels2ExternalPredictions(testSoftLabels, targetAlphabet);
  }
  
  public Predictions predict(
      Dataset trainingInstances, 
      Dataset heldoutInstances){
    
    // em-derived labels
    List<Prediction> labeledPredictions = Lists.newArrayList();
    List<Prediction> unlabeledPredictions = Lists.newArrayList();
    for (int i=0; i<instances.length; i++){
      List<Integer> pi = softLabels2ExternalPredictions(softlabels[i], targetAlphabet);
      if (isAnnotated(a[i])){
        labeledPredictions.add(new BasicPrediction(pi, externalInstances.get(i)));
      }
      else{
        // make a prediction based soley on the data model
        List<Integer> predictionIndices = unannotatedPrediction(externalInstances.get(i), dataAlphabet, targetAlphabet, maxent);
        unlabeledPredictions.add(new BasicPrediction(predictionIndices, externalInstances.get(i)));
      }
    }
    
    // generalization labels
    List<Prediction> heldoutPredictions = Lists.newArrayList();
    for (DatasetInstance inst: heldoutInstances){
      List<Integer> predictionIndices = unannotatedPrediction(inst, dataAlphabet, targetAlphabet, maxent);
      heldoutPredictions.add(new BasicPrediction(predictionIndices, inst));
    }
    
    double[] annotatorAccuracies = getAnnotatorAccuracies(gammas);
    double[][][] annotatorConfusionMatrices = gammas;
    double machineAccuracy = -1;
    double[][] machineConfusionMatrix = null;
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

}
