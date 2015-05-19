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
package edu.byu.nlp.crowdsourcing.models.gibbs;

import java.util.Arrays;
import java.util.Iterator;
import java.util.Map;

import org.fest.util.VisibleForTesting;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;

import edu.byu.nlp.classify.eval.BasicPrediction;
import edu.byu.nlp.classify.eval.Prediction;
import edu.byu.nlp.crowdsourcing.CrowdsourcingUtils;
import edu.byu.nlp.crowdsourcing.MultiAnnState;
import edu.byu.nlp.crowdsourcing.MultiAnnState.BasicMultiAnnState;
import edu.byu.nlp.crowdsourcing.PriorSpecification;
import edu.byu.nlp.crowdsourcing.TrainableMultiAnnModel;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.DatasetInstance;
import edu.byu.nlp.data.types.SparseFeatureVector;
import edu.byu.nlp.data.types.SparseFeatureVector.EntryVisitor;
import edu.byu.nlp.dataset.Datasets;
import edu.byu.nlp.dataset.SparseFeatureVectors;
import edu.byu.nlp.math.GammaFunctions;
import edu.byu.nlp.math.Math2;
import edu.byu.nlp.math.SparseRealMatrices;
import edu.byu.nlp.util.DoubleArrays;
import edu.byu.nlp.util.IntArrays;
import edu.byu.nlp.util.Matrices;

/**
 * @author rah67
 * @author plf1
 * 
 */
public class BlockCollapsedMultiAnnModelMath {
  private static final Logger logger = LoggerFactory.getLogger(TrainableMultiAnnModel.class);
  
  /*
   * public int docIndexFor(FlatInstance<SparseFeatureVector, Integer> instance) {
   * Iterable<Enumeration<FlatInstance<SparseFeatureVector, Integer>>> enumeration =
   * Iterables2.enumerate(data.allInstances()); for (Enumeration<Instance<Integer,
   * SparseFeatureVector>> e : enumeration) { if (e.getElement() == instance) { return e.getIndex();
   * } } return -1; }
   * 
   * public void updateA(FlatInstance<SparseFeatureVector, Integer> instance, int annotator, int label)
   * { int docIndex = docIndexFor(instance); if (docIndex < 0) { throw new
   * IllegalArgumentException("Couldn't find instance"); } ++a[docIndex][annotator][label]; }
   */


  static boolean hasCorrectCounts(int[] y, int[] m, double[] logCountOfY, double[][][] countOfJYAndA,
      int[][][] a, double[][] numAnnsPerJAndY, int[][] docJCount,
      double[][] countOfMAndX, double[] numFeaturesPerM, double[] docSize,
      double[][] logCountOfYAndM, double[] logSumCountOfYAndM, 
      int numAnnotators, int numLabels, int numFeatures, Dataset data, PriorSpecification priors) {
    if (data==null || priors==null){
      logger.warn("Ignoring the hasCorrectCounts assertion because priors and/or data have not been set");
      return true;
    }
    double[] logCountOfYSanity = DoubleArrays.of(priors.getBTheta(), numLabels);
    for (int i = 0; i < y.length; i++) {
      ++logCountOfYSanity[y[i]];
    }
    DoubleArrays.logToSelf(logCountOfYSanity);
    if (!DoubleArrays.equals(logCountOfYSanity, logCountOfY, 1e-8)) {
      return false;
    }

    { // keeps the int i out of the scope of other loops
      double[][] countOfMAndXSanity = Matrices.of(priors.getBPhi(), numLabels, numFeatures);
      int i = 0;
      for (DatasetInstance instance : data) {
        for (SparseFeatureVector.Entry e : instance.asFeatureVector().sparseEntries()) {
          countOfMAndXSanity[m[i]][e.getIndex()] += e.getValue();
        }
        ++i;
      }
      if (!Matrices.equals(countOfMAndXSanity, countOfMAndX, 1e-8)) {
        return false;
      }
    }

    double[][] logCountOfYAndMSanity = new double[numLabels][numLabels];
    CrowdsourcingUtils.initializeConfusionMatrixWithPrior(logCountOfYAndMSanity, priors.getBMu(), priors.getCMu());
    for (int i = 0; i < y.length; i++) {
      ++logCountOfYAndMSanity[y[i]][m[i]];
    }
    Matrices.logToSelf(logCountOfYAndMSanity);
    if (!Matrices.equals(logCountOfYAndMSanity, logCountOfYAndM, 1e-8)) {
      return false;
    }

    // Assumes a[][][] is correct
    for (int j = 0; j < countOfJYAndA.length; j++) {
      double[][] countOfJYAndASanity = new double[numLabels][numLabels];
      CrowdsourcingUtils.initializeConfusionMatrixWithPrior(countOfJYAndASanity, priors.getBGamma(j),
          priors.getCGamma());
      for (int i = 0; i < y.length; i++) {
        for (int k = 0; k < numLabels; k++) {
          countOfJYAndASanity[y[i]][k] += a[i][j][k];
        }
      }
      if (!Matrices.equals(countOfJYAndASanity, countOfJYAndA[j], 1e-8)) {
        return false;
      }
    }

    return true;
  }



  @VisibleForTesting
  static double getMeanSquaredDistanceFrom01(double[][] mat) {
    double mse = 0;
    for (int r = 0; r < mat.length; r++) {
      double[] row = mat[r];
      for (int c = 0; c < row.length; c++) {
        double val = mat[r][c];
        assert 0 <= val && val <= 1;
        double dist = (val < .5) ? val : 1 - val;
        mse += dist * dist;
      }
    }
    return mse;
  }

  public static double[][] confusionMatrix(Boolean labeled, int[] gold, int[] guesses, int numLabels, Dataset data){
    Preconditions.checkArgument(!(data==null && labeled!=null));
    Iterator<DatasetInstance> itr = (data==null)? null : data.iterator();
    double[][] confusion = new double[numLabels][numLabels];
    
    for (int i=0; i<guesses.length; i++){
      // ignore truly unlabeled data (label=-1)
      if (gold[i]==-1){
        continue;
      }
      DatasetInstance inst = (itr==null)? null: itr.next();
      // "unlabeled" (labels exist, but are hidden)
      boolean hasAnnotations = SparseRealMatrices.sum(inst.getAnnotations().getLabelAnnotations())!=0;
      if (labeled==null || (!labeled && !hasAnnotations)){
        ++confusion[gold[i]][guesses[i]];
      }
      // labeled
      if (labeled==null || (labeled && hasAnnotations)){
        ++confusion[gold[i]][guesses[i]];
      }
    }
    return confusion;
  }

  public static enum DiagonalizationMethod {NONE,GOLD,AVG_GAMMA,MAX_GAMMA,RAND}
  
  /**
   * @param gold (optional) If the GOLD diagonalization method is to 
   * be used, gold standard labels must also be passed in.
   */

  public static MultiAnnState fixLabelSwitching(MultiAnnState sample,
		DiagonalizationMethod diagonalizationMethod,
		int goldInstancesForDiagonalization,
		boolean diagonalizationWithFullConfusionMatrix) {
//  public static MultiAnnSample fixLabelSwitching(MultiAnnSample sample, int[] gold, boolean labelSwitchingCheatUsesFullConfusionMatrix, Dataset data) {
    // manually re-order the labels according to y and then
    // m so that mu and alpha_j line up with their greatest entries
    // along the diagonal as much as possible.
    // This helps alleviate the problem of label switching.

    // Note that we are not changing the meaning of labels
    // globally (in the data elicited from annotators);
    // merely ensuring that the features-only-ML
    // will be most likely to assign a Y=0 assignment
    // the label 0; a Y=1 assignment the label 1, and
    // so forth (rather than learning to systematically
    // map Y=0 to 2, for example, which would be a setting
    // with just as much probability as the Y=0 to 0 setting.

    // It's important to NOT use the counts (e.g., logCountOfYAndM)
    // We need to look at actual normalized accuracies (e.g., mu).

    int[] y = sample.getY().clone();
    int[] m = sample.getM().clone();
    double[][] mu = Matrices.clone(sample.getMu());
    double[][] meanMu = Matrices.clone(sample.getMeanMu());
    double[][][] alpha = Matrices.clone(sample.getAlpha());
    double[][][] meanAlpha = Matrices.clone(sample.getMeanAlpha());
    double[] theta = sample.getTheta().clone();
    double[] meanTheta = sample.getMeanTheta().clone();
    double[][] logPhi = Matrices.clone(sample.getLogPhi());
    double[][] meanLogPhi = Matrices.clone(sample.getMeanLogPhi());

    // -------------- Fix Y ----------------------- //
    int[] yMap;
    int[] gold;
    switch (diagonalizationMethod) {
    case NONE:
      logger.info("Not Diagonalizing");
      // diagonal mapping (no change)
      yMap = IntArrays.sequence(0, mu.length);
        break;
    case RAND:
      logger.info("Diagonalizing randomly");
      // randomly shuffled mapping 
      yMap = IntArrays.shuffled(IntArrays.sequence(0, mu.length));
        break;
    case GOLD:
      logger.info("Diagonalizing based on gold 'heldout data'");
      // create a confusion matrix by comparing gold labels with model predictions (gold labels are constructed to match the model ordering) 
      Boolean useLabeledConfusionMatrix = diagonalizationWithFullConfusionMatrix? null: true;
      gold = Datasets.concealedLabels(sample.getData(), sample.getInstanceIndices());
      int numGoldInstances = goldInstancesForDiagonalization==-1? gold.length: goldInstancesForDiagonalization;
      gold = Arrays.copyOfRange(gold, 0, numGoldInstances);
      int[] guesses = Arrays.copyOfRange(sample.getY(), 0, numGoldInstances);
      double[][] confusions = confusionMatrix(useLabeledConfusionMatrix, gold, guesses, sample.getNumLabels(), sample.getData());
      // in a CONFUSION matrix, columns correspond to the latent variable y. So 
      // permute columns to find a good diagonalization
      yMap = Matrices.getColReorderingForStrongDiagonal(confusions);
      break;
    case AVG_GAMMA:
	  // in a gamma matrix, rows correspond to the latent variable y, so permute rows 
	  // to find a good diagonalization
      logger.info("Diagonalizing based on average alpha");
      double[][] cumulativeAlphaMean = new double[sample.getNumLabels()][sample.getNumLabels()];
      for (int j = 0; j < sample.getNumAnnotators(); j++) {
        double[][] alphaMean = meanAlpha[j];
        Matrices.addToSelf(cumulativeAlphaMean, alphaMean);
      }
      yMap = Matrices.getRowReorderingForStrongDiagonal(cumulativeAlphaMean);
      break;
    case MAX_GAMMA:
      logger.info("Diagonalizing based on most confident alpha");
      // (pfelt) Find the most definitive alpha matrix
      // (the one with entries that diverge least from 0 and 1)
      // We'll map that to be diagonal and then apply its mapping
      // to all of the other alphas, since alpha matrices
      // are constrained by the data to be coherent.
      double[][] bestAlphaMean = null;
      double min = Double.POSITIVE_INFINITY;
      for (int j = 0; j < sample.getNumAnnotators(); j++) {
        double[][] alphaMean = meanAlpha[j];
        double error = getMeanSquaredDistanceFrom01(alphaMean);
        if (error < min) {
          min = error;
          bestAlphaMean = alphaMean;
        }
      }
      yMap = Matrices.getNormalizedRowReorderingForStrongDiagonal(bestAlphaMean);
      break;
    default:
      throw new IllegalArgumentException("unknown diagonalization method: "
          + diagonalizationMethod.toString());
    }

    logger.info("Y-mapping="+IntArrays.toString(yMap));

    // fix alpha
    for (int j = 0; j < sample.getNumAnnotators(); j++) {
      Matrices.reorderRowsToSelf(yMap, alpha[j]);
      Matrices.reorderRowsToSelf(yMap, meanAlpha[j]);
    }
    // fix y
    for (int i = 0; i < y.length; i++) {
      y[i] = yMap[y[i]];
    }
    // fix theta
    Matrices.reorderElementsToSelf(yMap, theta);
    Matrices.reorderElementsToSelf(yMap, meanTheta);
    // fix mu
    Matrices.reorderRowsToSelf(yMap, mu);
    Matrices.reorderRowsToSelf(yMap, meanMu);
    
    // (pfelt) we don't need to update cached values anymore since we're 
    // operating on a sample and not in the context of a model being sampled
//    // fix logSumCountOfYAndM
//    Matrices.reorderElementsToSelf(yMap, logSumCountOfYAndM);
//    // fix numAnnsPerJAndY
//    Matrices.reorderColsToSelf(yMap, numAnnsPerJAndY);

    // -------------- Fix M ----------------------- //
    // (pfelt) We used to sample from mu (by calling mu())
    // to get a mu setting. I've changed this to use the params
    // of mu for two reasons:
    // 1) a small performance savings
    // 2) it's easier to test
    int[] mMap;
    try{
      mMap = Matrices.getColReorderingForStrongDiagonal(meanMu);
    }
    catch (IllegalArgumentException e){
      mMap = new int[meanMu.length];
      for (int i=0; i<mMap.length; i++){
        mMap[i] = i;
      }
      logger.warn("unable to diagonalize m, returning the identity mapping. "
          + "If this is itemresp or momresp, then this is fine. "
          + "If this is multiann, then there is a serious problem.");
    }

    // fix mu
    Matrices.reorderColsToSelf(mMap, mu);
    Matrices.reorderColsToSelf(mMap, meanMu);
    // fix m
    for (int i = 0; i < m.length; i++) {
      m[i] = mMap[m[i]];
    }
    // fix phi
    Matrices.reorderRowsToSelf(mMap, logPhi);
    Matrices.reorderRowsToSelf(mMap, meanLogPhi);
    
    // (pfelt) we don't need to update cached values anymore since we're 
    // operating on a sample and not in the context of a model being sampled
//    // fix numFeaturesPerM
//    Matrices.reorderElementsToSelf(mMap, numFeaturesPerM);

    return new BasicMultiAnnState(y, m, theta, meanTheta, logPhi, meanLogPhi, mu, meanMu, alpha, meanAlpha, sample.getData(), sample.getInstanceIndices());
  }




  static boolean isAnnotated(int docIndex, int[][][] a){
    int[][] docAnnotations = a[docIndex];
    for (int[] arr: docAnnotations){
      if (IntArrays.sum(arr)>0){
        return true;
      }
    }
    return false;
  }
  
  static double[] computeMSums(SparseFeatureVector instance, double docSize, 
      double[][] countOfMAndX, double[] numFeaturesPerM, int numLabels, double lambda) {
    assert instance.sum()*lambda-docSize < 1e-10;
    double[] mSums = new double[numLabels];
    for (int d = 0; d < mSums.length; d++) {
//      mSums[d] = computeMSum(instance, docSize, countOfMAndX[d], numFeaturesPerM[d]);
      // (pfelt): normalize and scale current document counts
      double scaledDocSize = docSize;
      SparseFeatureVector scaledInstance = instance;
      if (lambda>=0){
        scaledDocSize = docSize*lambda;
//        double scale = (1/docSize)*lambda;
//        scaledDocSize = lambda;
        scaledInstance = scaledInstance.copy();
        SparseFeatureVectors.multiplyToSelf(scaledInstance, lambda);
      }
      mSums[d] = computeMSum(scaledInstance, scaledDocSize, countOfMAndX[d], numFeaturesPerM[d]);
    }
    return mSums;
  }
  
  @VisibleForTesting
  static double computeMSum(SparseFeatureVector instance, double docSize, double[] countOfMAndX,
      double numFeaturesPerM) {
    // these assertions no long valid with fractional counts
//    assert Math.round(instance.sum()) == docSize;
//    assert Math2.doubleEquals(DoubleArrays.sum(countOfMAndX), numFeaturesPerM, 1e-10);
//    return CollapsedParameters.sumLogOfRisingFactorial(instance, countOfMAndX)
//      - GammaFunctions.logRisingFactorial(numFeaturesPerM, docSize);
    return sumLogOfRatioOfGammas(instance, countOfMAndX)
        - GammaFunctions.logRatioOfGammasByDifference(numFeaturesPerM, docSize);
  }
  
  @VisibleForTesting
  static double[][] computeYMSums(int docIndex,
      double[] logSumCountOfYAndM, double[][] logCountOfYAndM, int numLabels,
      double lambda) {
    double[][] sums = new double[numLabels][numLabels];
    
    for (int c=0; c<numLabels; c++){
//        assert BlockCollapsedMultiAnnModel.isValidLogSumCountOfYAndM(c, logCountOfYAndM, logSumCountOfYAndM);
      double logSumCountOfY = GammaFunctions.logRatioOfGammasByDifference(Math.exp(logSumCountOfYAndM[c]), lambda);
      for (int d=0; d<numLabels; d++){
        sums[c][d] = computeYMSum(logCountOfYAndM[c][d], lambda);
        sums[c][d] -= logSumCountOfY;
      }
    }
    
    return sums;
  }
  
  static double computeYMSum(double logCountOfYAndM, double lambda){
    return GammaFunctions.logRatioOfGammasByDifference(Math.exp(logCountOfYAndM), lambda);
  }

  static double[] computeYSums(int docIndex, Map<Integer,Integer> instanceLabels,
      double[] logCountOfY, double[][][] countOfJYAndA, 
      double[][] numAnnsPerJAndY, int[][][] a, int[][] docJCount,
      int numAnnotators, double lambda) {
  double[] ySums = DoubleArrays.of(0, logCountOfY.length);
  Integer label = instanceLabels!=null && instanceLabels.containsKey(docIndex)? instanceLabels.get(docIndex): null;
  
//  double[] ySums = logCountOfY.clone();
    for (int c = 0; c < ySums.length; c++) {
      // labeled item (uses delta function prob)
      if (label!=null){
        ySums[c] = c==label? 0: Double.NEGATIVE_INFINITY; // prob = 0/1
      }
      // no label (just theta prior and annotations) 
      else{
        // theta
        ySums[c] = GammaFunctions.logRatioOfGammasByDifference(Math.exp(logCountOfY[c]), lambda);
        
        // gamma
        for (int j = 0; j < numAnnotators; j++) {
          ySums[c] +=
              computeYSum(countOfJYAndA[j][c], numAnnsPerJAndY[j][c], a[docIndex][j],
                  docJCount[docIndex][j]);
        }
      }
    }
    return ySums;
  }

  // NOTE: EXCLUDES log(b_j + n_c) (see computeYSums)
  public static double computeYSum(double[] countOfJYAndA, double numAnnsPerJAndY, int[] a_ij,
      int docJCount) {
    assert Math2.doubleEquals(numAnnsPerJAndY, DoubleArrays.sum(countOfJYAndA), 1e-10);
    assert docJCount == IntArrays.sum(a_ij);
    double ySum = -GammaFunctions.logRisingFactorial(numAnnsPerJAndY, docJCount);
    for (int k = 0; k < a_ij.length; k++) {
      if (a_ij[k] > 0) {
        ySum += GammaFunctions.logRisingFactorial(countOfJYAndA[k], a_ij[k]);
      }
    }
    return ySum;
  }
 
// TODO: alter this function to return ranked predictions using sampling distributions.
  public static Iterable<Prediction> predictions(final Dataset data, final int y[], final Map<String,Integer> instanceMap) {
    return new Iterable<Prediction>() {

      @Override
      public Iterator<Prediction> iterator() {
        return new Iterator<Prediction>() {

          // only predict on unlabeled portion of dataset; the labeled portion is known
          private Iterator<DatasetInstance> it = data.iterator();

          @Override
          public boolean hasNext() {
            return it.hasNext();
          }

          @Override
          public Prediction next() {
            DatasetInstance instance = it.next();
            // prediction based on most recent sample
            Integer prediction = null;
            if (instanceMap.containsKey(instance.getInfo().getSource())){
              prediction = y[instanceMap.get(instance.getInfo().getSource())];
            }
            return new BasicPrediction(prediction, instance);
          }

          @Override
          public void remove() {
            throw new UnsupportedOperationException();
          }
        };
      }
    };
  }


  private static class SumLogOfRatioOfGammas implements EntryVisitor {

    private final double[] topicWordCounts;
    private double acc;
    
    public SumLogOfRatioOfGammas(double[] topicWordCounts) {
      this.topicWordCounts = topicWordCounts;
      this.acc = 0.0;
    }
    
    @Override
    public void visitEntry(int index, double value) {
      acc += GammaFunctions.logRatioOfGammas(topicWordCounts[index]+value, topicWordCounts[index]);
    }
    
    public double getSum() {
      return acc;
    }
    
  }

  public static double sumLogOfRatioOfGammas(SparseFeatureVector doc, double[] topicWordCounts) {
    SumLogOfRatioOfGammas visitor = new SumLogOfRatioOfGammas(topicWordCounts);
    doc.visitSparseEntries(visitor);
    return visitor.getSum();
  }
  
}
