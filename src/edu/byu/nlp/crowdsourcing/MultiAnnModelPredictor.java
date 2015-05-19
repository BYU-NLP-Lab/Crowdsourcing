/**
 * Copyright 2013 Brigham Young University
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
package edu.byu.nlp.crowdsourcing;

import java.util.List;

import com.google.common.collect.Lists;

import edu.byu.nlp.classify.eval.BasicPrediction;
import edu.byu.nlp.classify.eval.Prediction;
import edu.byu.nlp.classify.eval.Predictions;
import edu.byu.nlp.crowdsourcing.models.gibbs.BlockCollapsedMultiAnnModelMath;
import edu.byu.nlp.crowdsourcing.models.gibbs.BlockCollapsedMultiAnnModelMath.DiagonalizationMethod;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.DatasetInstance;
import edu.byu.nlp.dataset.Datasets;
import edu.byu.nlp.util.DoubleArrays;
import edu.byu.nlp.util.Enumeration;
import edu.byu.nlp.util.Iterables2;
import edu.byu.nlp.util.Matrices;

/**
 * Generate predictions using a private final MultiAnnModel
 * 
 * @author rah67
 *
 */
public class MultiAnnModelPredictor {

  private final MultiAnnModel model;
  private Dataset data;
  private boolean predictSingleLastSample;
  private int[] gold;
  private boolean diagonalizationWithFullConfusionMatrix;
  private DiagonalizationMethod diagonalizationMethod;
  private int goldInstancesForDiagonalization;
  
  /**
   * @param gold (optional) if not null, the predictor fixed label switching according to the confusion matrix of the labeled data 
   */

  public MultiAnnModelPredictor(MultiAnnModel model, Dataset data,
		boolean predictSingleLastSample,
		DiagonalizationMethod diagonalizationMethod,
		int goldInstancesForDiagonalization,
		boolean diagonalizationWithFullConfusionMatrix, int[] gold) {
    this.model=model;
    this.data=data;
    this.predictSingleLastSample=predictSingleLastSample;
    this.gold=gold;
    this.diagonalizationMethod=diagonalizationMethod;
    this.goldInstancesForDiagonalization=goldInstancesForDiagonalization;
    this.diagonalizationWithFullConfusionMatrix=diagonalizationWithFullConfusionMatrix;
  }


public MultiAnnState getFinalPredictiveParameters(){
    MultiAnnState sample = model.getCurrentState();
    
    if (!predictSingleLastSample){
      // use max marginal y and m values, and averaged marginal values for Mu
      // TODO (use marginals for gamma, theta, and phi)
      int[] marginalY = model.getMarginalYs().argmax();
      int[] marginalM = model.getMarginalMs().argmax();
      double[][] marginalMu = model.getMarginalYMs().average();
      Matrices.normalizeRowsToSelf(marginalMu);
      sample = new MultiAnnState.BasicMultiAnnState(marginalY, 
          marginalM, sample.getTheta(), sample.getMeanTheta(), sample.getLogPhi(), 
          sample.getMeanLogPhi(), sample.getMu(), marginalMu, sample.getAlpha(), sample.getMeanAlpha(),
          sample.getData(), sample.getInstanceIndices());
    }
    
    // fix label switching (optionally cheating to find a good diagonalization)
    return BlockCollapsedMultiAnnModelMath.fixLabelSwitching(sample, diagonalizationMethod, goldInstancesForDiagonalization, diagonalizationWithFullConfusionMatrix);
  }
  
  public Predictions predict(Dataset heldoutData) {
    
    List<Prediction> labeledPredictions = Lists.newArrayList();
    List<Prediction> unlabeledPredictions = Lists.newArrayList();
    
    MultiAnnState sample = getFinalPredictiveParameters();
    
//  // since we are measuring top-n guesses, I (pfelt) have commented out 
//  // this logic, which takes the final state of the model as the prediction. 
//  // Instead, parameter values are used to make all predictions; "labeled", "unlabeled", and "heldout" alike, 
//  // since this is the only way I can think of to get the model's topn predictions. 
//  // (if we were sampling, we would track variable distributions over time, but we aren't 
//  // sampling anymore!)
//    Map<String, Integer> instanceIndices = model.getInstanceIndices();
//    // read corpus predictions off of y
//    for (Prediction prediction : 
//      BlockCollapsedMultiAnnModelMath.predictions(data, sample.getY(), instanceIndices)) {
//      
//      if (prediction.getInstance().getAnnotations().size() > 0) {
//        labeledPredictions.add(prediction);
//      } else {
//        unlabeledPredictions.add(prediction);
//      }
//    }

    // corpus predictions 
    for (Prediction prediction : calculatePredictions(data, data.getInfo().getNumClasses(), data.getInfo().getNumDocuments(), sample)) {
      if (prediction.getInstance().hasAnnotations()) {
        labeledPredictions.add(prediction);
      } else {
        unlabeledPredictions.add(prediction);
      }
    }

    // out-of-corpus predictions
    int numInstances = 0;
    for (DatasetInstance e: heldoutData){
      numInstances++;
    }
    List<Prediction> heldoutPredictions = calculatePredictions(heldoutData, data.getInfo().getNumClasses(), numInstances, sample);
    
    // use means
    double[] annotatorAccuracies = getMeanAnnotatorAccuracies(sample);
    double[][][] annotatorConfusionMatrices = sample.getMeanAlpha();
    double machineAccuracy = getMeanMachineAccuracy(sample);
    double[][] machineConfusionMatrix = sample.getMeanMu();
    
    double logJoint = model.logJoint();
    return new Predictions(labeledPredictions,
                                                         unlabeledPredictions,
                                                         heldoutPredictions,
                                                         annotatorAccuracies,
                                                         annotatorConfusionMatrices,
                                                         machineAccuracy,
                                                         machineConfusionMatrix,
                                                         logJoint);
  }


  private static List<Prediction> calculatePredictions(
      Dataset dataset, int numLabels, int numInstances,
      MultiAnnState sample) {
    
    List<Prediction> predictions = Lists.newArrayList();
//    // FIXME(rhaertel): use the predictive distribution)
//    double[] theta = model.sampleTheta();
//    double[][] logPhi = model.logSamplePhi();
//    double[][] logMu = model.sampleMu();
    // FIXME(pfelt): for the time being, to reduce variance in answers
    // we are using parameter means to calculate predictions on heldout data.
    // In the future, do something more Bayesian that takes into account
    // the uncertainty, maybe using multiple chains.
    double[] logTheta = sample.getMeanTheta();
    logTheta = logTheta.clone();
    DoubleArrays.logToSelf(logTheta);
    double[][] logPhi = Matrices.clone(sample.getMeanLogPhi());
    double[][] logMu = Matrices.clone(sample.getMeanMu());
    Matrices.logToSelf(logMu);
    double[][][] logAlpha = Matrices.clone(sample.getMeanAlpha());
    Matrices.logToSelf(logAlpha);
    int[][][] annotations = Datasets.compileDenseAnnotations(dataset);
        
    
    for (Enumeration<DatasetInstance> entry : Iterables2.enumerate(dataset)) {
      DatasetInstance instance = entry.getElement();
      Integer index = entry.getIndex();
      
      // Calculate argmax_y p(y|X=x,theta,phi,mu) where X is set to the current instance
      // and under the assumption that there are no seen annotations A.
      //
      // We are using a fixed assignment to theta, phi, and mu derived 
      // from sampling. Problem is, theta, phi and mu are collapsed during 
      // sampling, so we use their mean values as stand-ins. We may be able to 
      // improve our estimate at the cost of efficiency by doing some 
      // stochastic integration using  samples from theta, phi, 
      // and mu to approximate p(y,X=x)

      // compute log p(y,m,X=x|a,theta,phi,mu) 
      double[][] logProb = new double[numLabels][numLabels];
      for (int y = 0; y < logProb.length; y++) {
        // log p(m|y)
        System.arraycopy(logMu[y], 0, logProb[y], 0, logMu.length);
        // log p(y,m) = log p(m|y) + log p(y)
        DoubleArrays.addToSelf(logProb[y], logTheta[y]);
        // log p(y,m,X=x) = log p(X=x|m) + log p(m|y)  
        instance.asFeatureVector().preMultiplyAsColumnAndAddTo(logPhi, logProb[y]);
        // log p(y,m,X=x,A=a) = log p(y,m,X=x) + log p(A=a|y)
        double logProbYGivenA = 0;
        for (int j=0; j<sample.getNumAnnotators(); j++){
          logProbYGivenA += DoubleArrays.dotProduct(logAlpha[j][y], annotations[index][j]);
        }
        DoubleArrays.addToSelf(logProb[y], logProbYGivenA);
      }
      // log normalize over all y,m values
      Matrices.logNormalizeToSelf(logProb);
      
      // Compute the marginal p(y,X=x|theta,phi,mu)
      Matrices.expToSelf(logProb);
      double[] probs = Matrices.sumOverSecond(logProb);
      
      // Report argmax_y p(y,X=x|theta,phi,mu) = argmax_y p(y|X=x,theta,phi,mu)
      predictions.add(new BasicPrediction(DoubleArrays.argMaxList(-1, probs), instance));
    }
    return predictions;
  }
  

  /**
   * Returns annotator accuracies based on a sample from alphas
   */
  public static double[] getAnnotatorAccuracies(MultiAnnState sample) {
    double[] theta = sample.getTheta();
    double[] accuracies = new double[sample.getNumAnnotators()];
    for (int l = 0; l < accuracies.length; l++) {
      double alpha[][] = sample.getAlpha()[l]; 
      accuracies[l] = CrowdsourcingUtils.getAccuracy(theta, alpha);
    }
    return accuracies;
  }
  
  /**
   * Returns annotator accuracies based on mean theta and alphas
   */
  public static double[] getMeanAnnotatorAccuracies(MultiAnnState sample) {
    double[] theta = sample.getMeanTheta();
    double[] accuracies = new double[sample.getNumAnnotators()];
    double[][][] meanAlpha = sample.getMeanAlpha();
    for (int l = 0; l < accuracies.length; l++) {
      double alpha[][] = meanAlpha[l];
      accuracies[l] = CrowdsourcingUtils.getAccuracy(theta, alpha);
    }
    return accuracies;
  }

  public static double getMachineAccuracy(MultiAnnState sample) {
    // getTheta() and getMu() are fixed across calls
    return CrowdsourcingUtils.getAccuracy(sample.getTheta(), sample.getMu());
  }

  public static double getMeanMachineAccuracy(MultiAnnState sample) {
    return CrowdsourcingUtils.getAccuracy(sample.getMeanTheta(), sample.getMeanMu());
  }
  
}
