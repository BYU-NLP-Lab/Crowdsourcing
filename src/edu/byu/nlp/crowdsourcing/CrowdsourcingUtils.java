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


/**
 * @author rah67
 * @author plf1
 *
 */
public class CrowdsourcingUtils {

  /**
   * Delegates to initializeConfusionMatrixWithPrior in order to create a 
   * confusion matrix for each annotator specified in the prior.
   */
  public static double[][][] annotatorConfusionMatricesFromPrior(PriorSpecification priors, int numClasses) {
    double[][][] gamma = new double[priors.getNumAnnotators()][numClasses][numClasses];
    for (int j=0; j<priors.getNumAnnotators(); j++){
      initializeConfusionMatrixWithPrior(gamma[j], priors.getBGamma(j), priors.getCGamma());
    }
    return gamma;
  }
  
  /**
   * Fills a two-dimensional array with prior counts.
   * Since mu_k ~ Dir(b_mu, c_mu). b_mu represents what we think the accuracy of the model is
   * regardless of class (future versions may allow for a per-class accuracy parameter to
   * distinguish difficult classes from easy ones). c_mu is a concentration parameter, roughly
   * equivalent to precision; large numbers represent high certainty in the accuracy (b_mu). We
   * assume (for now) that all other classes are equi-probable. We can therefore use a standard
   * Dirichlet parameterization for class 0 (wlog), by setting the parameter
   * alpha = cMu * (bMu, (1-bMu) / (numLabels - 1), ...).  
   */
  public static void initializeConfusionMatrixWithPrior(
      double[][] mu, double accuracy, double concentration) {
    int numLabels = mu.length;
    // Accuracy multiplied by the concentration parameter
    double diagonalParam = accuracy * concentration;
    // The diagonal is "accuracy"
    for (int k = 0; k < numLabels; k++) {
      mu[k][k] = diagonalParam;
    }
    double offDiagonalParam = (1.0 - accuracy) / (numLabels - 1) * concentration;
    for (int k = 0; k < numLabels; k++) {
      for (int kPrime = 0; kPrime < k; kPrime++) {
        mu[k][kPrime] = mu[kPrime][k] = offDiagonalParam;
      }
    }
  }

  /**
   * Calls getAccuracy once for each dimension of confusionProbs and returns 
   * results in an array
   */
  public static double[] getAccuracies(double[] theta, double[][][] confusionProbs) {
    double[] results = new double[confusionProbs.length];
    for (int i=0; i<confusionProbs.length; i++){
      results[i] = getAccuracy(theta, confusionProbs[i]);
    }
    return results;
  }
  
  /**
   * Summarizes annotator confusion matrices (gamma) in a single 
   * vector indicating how likely they are to give the right 
   * answer. Theta is an estimate of the probability of each class occurring; 
   * the diagonal of confusionProbs indicates how likely a given entity is to 
   * get the correct answer for each class. 
   * 
   * \sum_k p(Y = k) * alpha_lkk
   */
  public static double getAccuracy(double[] theta, double[][] confusionProbs) {
    double acc = 0.0;
    for (int k = 0; k < confusionProbs.length; k++) {
      acc += theta[k] * confusionProbs[k][k];
    }
    return acc;
  }
  
}
