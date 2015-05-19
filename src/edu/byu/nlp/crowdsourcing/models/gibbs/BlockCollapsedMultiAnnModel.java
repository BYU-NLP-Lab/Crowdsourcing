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

import java.io.PrintWriter;
import java.util.Map;

import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.special.Gamma;
import org.fest.util.VisibleForTesting;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.io.ByteStreams;

import edu.byu.nlp.classify.data.DatasetLabeler;
import edu.byu.nlp.classify.eval.Predictions;
import edu.byu.nlp.crowdsourcing.MultiAnnDatasetLabeler;
import edu.byu.nlp.crowdsourcing.MultiAnnModel;
import edu.byu.nlp.crowdsourcing.MultiAnnModelBuilders.AbstractMultiAnnModelBuilder;
import edu.byu.nlp.crowdsourcing.MultiAnnState;
import edu.byu.nlp.crowdsourcing.MultiAnnState.CollapsedMultiAnnState;
import edu.byu.nlp.crowdsourcing.PriorSpecification;
import edu.byu.nlp.crowdsourcing.TrainableMultiAnnModel;
import edu.byu.nlp.crowdsourcing.models.gibbs.BlockCollapsedMultiAnnModelMath.DiagonalizationMethod;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.DatasetInstance;
import edu.byu.nlp.data.types.SparseFeatureVector;
import edu.byu.nlp.math.Math2;
import edu.byu.nlp.stats.RandomGenerators;
import edu.byu.nlp.util.DoubleArrays;
import edu.byu.nlp.util.IntArrayCounter;
import edu.byu.nlp.util.Matrices;
import edu.byu.nlp.util.MatrixAverager;

/**
 * @author rah67  
 * docWeight
 */
public class BlockCollapsedMultiAnnModel extends TrainableMultiAnnModel {

  private static final boolean USE_LOG_JOINT_FOR_COEFFS = false;

  private static final Logger logger = LoggerFactory.getLogger(BlockCollapsedMultiAnnModel.class);

  private final PriorSpecification priors;

  // Reference to data
  final Dataset data;

  // Assignment counter
  private final IntArrayCounter yMarginals;
  private final IntArrayCounter mMarginals;
  private final MatrixAverager ymMarginals;

  // Assignments
  private final int[] y; // inferred 'true' label assignments
  // len(y)=numdocs, type(y[i])=label
  private final int[] m; // features-only-ML label assignments
  // len(m)=numdocs, type(m[i])=label

  // Sufficient statistics
  private final double[] logCountOfY; // (replaces theta)
  private final double[][] countOfMAndX; // (replaces phi)
  private final double[][] logCountOfYAndM; // (replaces mu)
  private final double[][][] countOfJYAndA; // (replaces alpha)

  // Cached values for efficiency
  private final double[] logSumCountOfYAndM; // log \sum_k countOfYAndM[y][k]
  private final double[] numFeaturesPerM; // \sum_f b_\phi + x[i][f]
  private final double[][] numAnnsPerJAndY; // [annotator][y]; \sum_k' countOfJYAndA[j][y][k']
                                            // (similar to numFeaturesPerM)

  // Also cached, but part of the data (no need to update)
  private final double[] docSize;
  private final int[][][] a; // count of annotations [doc][annotator][label]
  private final int[][] docJCount; // [doc][annotator]; num of anns per annotator per doc; (similar
                                   // to docSize)

  private double temp;
  private double[] lambdas;

  private final RandomGenerator rnd;

  // cached between drawing new samples
  private MultiAnnState currentSample = null;
  
  // true labels for debugging
  int[] gold;

  private Map<String,Integer> instanceIndices;

  private Map<Integer, Integer> instanceLabels;

  // Builder pattern
  public static class ModelBuilder extends AbstractMultiAnnModelBuilder{
    @Override
    protected MultiAnnModel build(PriorSpecification priors, Dataset data,
        Map<String,Integer> instanceIndices, Map<Integer,Integer> instanceLabels, int[][][] a,
        int[] y, int[] m, double[] logCountOfY, double[][] logCountOfYAndM,
        double[][] countOfMAndX, double[][][] countOfJYAndA,
        double[] logSumCountOfYAndM, double[] numFeaturesPerM,
        double[] docSize, double[][] numAnnsPerJAndY, int[][] docJCount,
        double initialTemp, double[] lambdas, int[] gold,
        RandomGenerator rnd) {
      return new BlockCollapsedMultiAnnModel(priors, data, instanceIndices, instanceLabels, a, y, m, logCountOfY,
          logCountOfYAndM, countOfMAndX, countOfJYAndA, logSumCountOfYAndM,
          numFeaturesPerM, docSize, numAnnsPerJAndY,
          docJCount, initialTemp, lambdas, gold, rnd);
    }
  }

  @VisibleForTesting
  BlockCollapsedMultiAnnModel(PriorSpecification priors, Dataset data, Map<String,Integer> instanceIndices, 
      Map<Integer,Integer> instanceLabels, int[][][] a, int[] y, int[] m, 
      double[] logCountOfY, double[][] logCountOfYAndM, double[][] countOfMAndX,
      double[][][] countOfJYAndAnn, double[] logSumCountOfYAndM, double[] numFeaturesPerM,
      double[] docSize, double[][] numAnnsPerJAndY, int[][] docJCount, 
      double initialTemp, double[] lambdas, int[] gold, RandomGenerator rnd) {
    this.priors = priors;
    this.data = data;
    this.instanceIndices=instanceIndices;
    this.instanceLabels=instanceLabels;
    this.a = a;
    this.logCountOfYAndM = logCountOfYAndM;
    this.countOfJYAndA = countOfJYAndAnn;
    this.y = y;
    this.m = m;
    this.logCountOfY = logCountOfY;
    this.countOfMAndX = countOfMAndX;
    this.logSumCountOfYAndM = logSumCountOfYAndM;
    this.numFeaturesPerM = numFeaturesPerM;
    this.docSize = docSize;
    this.numAnnsPerJAndY = numAnnsPerJAndY;
    this.docJCount = docJCount;
    this.temp = initialTemp;
    this.lambdas=lambdas;
    this.gold=gold;
    this.rnd = rnd;
    // marginal counters
    this.yMarginals = new IntArrayCounter(y.length, logCountOfY.length);
    this.mMarginals = new IntArrayCounter(m.length, logCountOfY.length);
    this.ymMarginals = new MatrixAverager(logCountOfY.length, logCountOfY.length);
    marginalCountersIncrement(); // include initial values
  }
  
  public double getDocumentWeight(int docIndex){
    return lambdas==null? 1: lambdas[docIndex]; 
  }

  public int numAnnotators() {
    return countOfJYAndA.length;
  }

  public int numLabels() {
    return data.getInfo().getNumClasses();
  }
  
  public MultiAnnState getCurrentState(){
    if (currentSample==null){
      currentSample = new CollapsedMultiAnnState(y, m, logCountOfY, countOfMAndX, logCountOfYAndM, countOfJYAndA, data, instanceIndices, rnd);
    }
    return currentSample;
  }

  public IntArrayCounter getMarginalYs(){
    return yMarginals;
  }

  public IntArrayCounter getMarginalMs(){
    return mMarginals;
  }

  public MatrixAverager getMarginalYMs(){
    return ymMarginals;
  }

  public double[][] confusionMatrixY(){
    return BlockCollapsedMultiAnnModelMath.confusionMatrix(
        null, gold, y, numLabels(), data);
  }
  public double[][] confusionMatrixM(){
    return BlockCollapsedMultiAnnModelMath.confusionMatrix(
        null, gold, m, numLabels(), data);
  }

  
  public double getTemp() {
    return temp;
  }

  public void setTemp(double temp) {
    this.temp = temp;
    // this indicates a change in sampling settings and triggers a history reset
    marginalCountersReset();
    marginalCountersIncrement(); // initialize with y state
  }
  
  private void marginalCountersReset(){
    yMarginals.reset();
    mMarginals.reset();
    ymMarginals.reset();
  }
  
  private void marginalCountersIncrement(){
    yMarginals.increment(y);
    mMarginals.increment(m);
    ymMarginals.increment(Matrices.exp(logCountOfYAndM));
  }

  @VisibleForTesting
  double[] getLogCountOfY() {
    return logCountOfY;
  }

  @VisibleForTesting
  double[][] getCountOfMAndX() {
    return countOfMAndX;
  }

  @VisibleForTesting
  double[][] getLogCountOfYAndM() {
    return logCountOfYAndM;
  }

  @VisibleForTesting
  double[][][] getCountOfJYAndA() {
    return countOfJYAndA;
  }

  @VisibleForTesting
  double[] getLogSumCountOfYAndM() {
    return logSumCountOfYAndM; 
  }

  @VisibleForTesting
  double[] getNumFeaturesPerM() {
    return numFeaturesPerM;
  }

  @VisibleForTesting
  double[][] getNumAnnsPerJAndY() {
    return numAnnsPerJAndY;
  }

  @VisibleForTesting
  int[][][] getA() {
    return a;
  }

  @VisibleForTesting
  int[] getY(){
    return y;
  }
  
  @VisibleForTesting
  int[] getM(){
    return m;
  }
  
  @VisibleForTesting
  double[] docSize() {
    return docSize;
  }

  @VisibleForTesting
  int[][] docJCount() {
    return docJCount; 
  }

  public PriorSpecification getPriors() {
    return priors;
  }
  
  public void sample() {
    currentSample = null; // invalidate cached values
    // enumerate instances
    int docIndex = 0;
    for (DatasetInstance instance : data) {
      blockSampleYAndM(docIndex, instance.asFeatureVector());
      ++docIndex;
    }
    // average over marginal dist for y, m, and mu
    marginalCountersIncrement();
  }
  public void blockSampleYAndM(int docIndex, SparseFeatureVector instance) {
    double lambda = getDocumentWeight(docIndex);
    decrementCounts(docIndex, instance, lambda);
    double[] coeffs = computeCoefficients(docIndex, instance, numAnnotators(), numLabels());
    if (temp != 1.0) {
      DoubleArrays.divideToSelf(coeffs, temp);
    }
    int choice = RandomGenerators.nextIntUnnormalizedLogProbs(rnd, coeffs);
    int nextY = choice / numLabels();
    int nextM = choice % numLabels();
    incrementCounts(docIndex, instance, nextY, nextM, lambda);
    assert BlockCollapsedMultiAnnModelMath.hasCorrectCounts(y, m, logCountOfY, countOfJYAndA,
        a, numAnnsPerJAndY, docJCount,
        countOfMAndX, numFeaturesPerM, docSize,
        logCountOfYAndM, logSumCountOfYAndM, 
        numAnnotators(), numLabels(), data.getInfo().getNumFeatures(), data, priors);
  }

  /**
   * sample from y, marginalizing out m
   */
  public void sampleY() {
    currentSample = null; // invalidate cached values
    // enumerate instances
    int docIndex = 0;
    for (DatasetInstance instance : data) {
      sampleY(docIndex, instance.asFeatureVector());
      ++docIndex;
    }
    // average over marginal dist for y, m, and mu
    marginalCountersIncrement();
  }
  public void sampleY(int docIndex, SparseFeatureVector instance) {
    double lambda = getDocumentWeight(docIndex);
    decrementCounts(docIndex, instance, lambda);
    double[] coeffs = computeCoefficients(docIndex, instance, numAnnotators(), numLabels());
    if (temp != 1.0) {
      DoubleArrays.divideToSelf(coeffs, temp);
    }
    
    // marginalize m
    DoubleArrays.logNormalizeToSelf(coeffs);
    DoubleArrays.expToSelf(coeffs);
    double[] yPosterior = new double[numLabels()];
    int cIndex = 0;
    for (int k = 0; k < yPosterior.length; k++) {
      for (int kPrime = 0; kPrime < yPosterior.length; kPrime++) {
        yPosterior[k] += coeffs[cIndex++];
      }
    }
    
    int nextY = RandomGenerators.nextIntUnnormalizedProbs(rnd, yPosterior);
    int nextM = m[docIndex];
    incrementCounts(docIndex, instance, nextY, nextM, lambda);
    assert BlockCollapsedMultiAnnModelMath.hasCorrectCounts(y, m, logCountOfY, countOfJYAndA,
        a, numAnnsPerJAndY, docJCount,
        countOfMAndX, numFeaturesPerM, docSize,
        logCountOfYAndM, logSumCountOfYAndM, 
        numAnnotators(), numLabels(), data.getInfo().getNumFeatures(), data, priors);
  }
  

  /**
   * sample from m, marginalizing out y
   */
  public void sampleM() {
    currentSample = null; // invalidate cached values
    // enumerate instances
    int docIndex = 0;
    for (DatasetInstance instance : data) {
      sampleM(docIndex, instance.asFeatureVector());
      ++docIndex;
    }
    // average over marginal dist for y, m, and mu
    marginalCountersIncrement();
  }
  public void sampleM(int docIndex, SparseFeatureVector instance) {
    double lambda = getDocumentWeight(docIndex);
    decrementCounts(docIndex, instance, lambda);
    double[] coeffs = computeCoefficients(docIndex, instance, numAnnotators(), numLabels());
    if (temp != 1.0) {
      DoubleArrays.divideToSelf(coeffs, temp);
    }
    
    // marginalize m
    DoubleArrays.logNormalizeToSelf(coeffs);
    DoubleArrays.expToSelf(coeffs);
    double[] mPosterior = new double[numLabels()];
    int cIndex = 0;
    for (int k = 0; k < mPosterior.length; k++) {
      for (int kPrime = 0; kPrime < mPosterior.length; kPrime++) {
        mPosterior[kPrime] += coeffs[cIndex++];
      }
    }
    
    int nextY = y[docIndex];
    int nextM = RandomGenerators.nextIntUnnormalizedProbs(rnd, mPosterior);
    incrementCounts(docIndex, instance, nextY, nextM, lambda);
    assert BlockCollapsedMultiAnnModelMath.hasCorrectCounts(y, m, logCountOfY, countOfJYAndA,
        a, numAnnsPerJAndY, docJCount,
        countOfMAndX, numFeaturesPerM, docSize,
        logCountOfYAndM, logSumCountOfYAndM, 
        numAnnotators(), numLabels(), data.getInfo().getNumFeatures(), data, priors);
  }


  /**
   * Does the same thing as sample, but instead of sampling from each complete conditional in turn,
   * takes the max
   */
  public void maximize() {
    currentSample = null; // invalidate cached values
    // enumerate instances
    int docIndex = 0;
    for (DatasetInstance instance : data) {
      blockMaximizeYAndM(docIndex, instance.asFeatureVector());
      ++docIndex;
    }
  }
  public void blockMaximizeYAndM(int docIndex, SparseFeatureVector instance) {
    double lambda = getDocumentWeight(docIndex);
    decrementCounts(docIndex, instance, lambda);
    
    double[] coeffs = computeCoefficients(docIndex, instance, numAnnotators(), numLabels());
    
    int choice = DoubleArrays.argMax(coeffs);
    int nextY = choice / numLabels();
    int nextM = choice % numLabels();
    incrementCounts(docIndex, instance, nextY, nextM, lambda);
  }

  /**
   * Finds the argmax assignment to M (marginalizing over Y).
   * 
   * NOTE: this may need mathematical justification.
   */
  public void maximizeM() {
    currentSample = null; // invalidate cached values
    // enumerate instances
    int docIndex = 0;
    for (DatasetInstance instance : data) {
      maximizeM(docIndex, instance.asFeatureVector());
      ++docIndex;
    }
  }
  public void maximizeM(int docIndex, SparseFeatureVector instance) {
    double lambda = getDocumentWeight(docIndex);
    
    decrementCounts(docIndex, instance, lambda);
    double[] coeffs = computeCoefficients(docIndex, instance, 
        numAnnotators(), numLabels());
    
    DoubleArrays.logNormalizeToSelf(coeffs);
    DoubleArrays.expToSelf(coeffs);
    
    double[] mPosterior = new double[numLabels()];
    int cIndex = 0;
    for (int k = 0; k < mPosterior.length; k++) {
      for (int kPrime = 0; kPrime < mPosterior.length; kPrime++) {
        mPosterior[kPrime] += coeffs[cIndex++];
      }
    }
    
    int nextY = y[docIndex]; 
    int nextM = DoubleArrays.argMax(mPosterior);
    incrementCounts(docIndex, instance, nextY, nextM, lambda);
  }
  
  /**
   * Finds the argmax assignment to Y (marginalizing over M).
   * 
   * NOTE: this may need mathematical justification.
   */
  public void maximizeY() {
    currentSample = null; // invalidate cached values
    // enumerate instances
    int docIndex = 0;
    for (DatasetInstance instance : data) {
      maximizeY(docIndex, instance.asFeatureVector());
      ++docIndex;
    }
  }
  public void maximizeY(int docIndex, SparseFeatureVector instance) {
    double lambda = getDocumentWeight(docIndex);
    
    decrementCounts(docIndex, instance, lambda);
    double[] coeffs = computeCoefficients(docIndex, instance, 
        numAnnotators(), numLabels());
    
    DoubleArrays.logNormalizeToSelf(coeffs);
    DoubleArrays.expToSelf(coeffs);
    
    double[] yPosterior = new double[numLabels()];
    int cIndex = 0;
    for (int k = 0; k < yPosterior.length; k++) {
      for (int kPrime = 0; kPrime < yPosterior.length; kPrime++) {
        yPosterior[k] += coeffs[cIndex++];
      }
    }
    
    int nextY = DoubleArrays.argMax(yPosterior);
    int nextM = m[docIndex];
    incrementCounts(docIndex, instance, nextY, nextM, lambda);
  }
  



  /**
   * For some normalization constant c, computes log p(y, m) + c for all y and m, the result of
   * which is stored in a linearized array s.t. for any given value of y, all values of m are stored
   * contiguously and sequentially, i.e. y=0, m=2 is indexed at location 2; y=0, m=3 is indexed at
   * location 3, etc.
   */
  @VisibleForTesting
  double[] computeCoefficients(int docIndex, SparseFeatureVector instance, int numAnnotators, int numLabels) {
    
    if (USE_LOG_JOINT_FOR_COEFFS) {
      return computeCoefficientsUsingJoint(instance, docIndex);
    }
    double lambda = getDocumentWeight(docIndex); 
    double[] ySums = BlockCollapsedMultiAnnModelMath.computeYSums(docIndex, instanceLabels, logCountOfY, countOfJYAndA, 
        numAnnsPerJAndY, a, docJCount, numAnnotators, lambda);
    double[] mSums = BlockCollapsedMultiAnnModelMath.computeMSums(instance, docSize[docIndex], 
        countOfMAndX, numFeaturesPerM, numLabels, lambda);
    double[][] ymSums = BlockCollapsedMultiAnnModelMath.computeYMSums(docIndex, 
        logSumCountOfYAndM, logCountOfYAndM, numLabels, lambda);
    // double[] coeff1 = computeCoefficientsUsingJoint(instance, docIndex);
    // double[] coeff2 = computeCoefficients(ySums, mSums, logCountOfYAndM, logSumCountOfYAndM,
    // instance, docIndex);
    // DoubleArrays.logNormalizeToSelf(coeff1);
    // DoubleArrays.expToSelf(coeff1);
    // DoubleArrays.logNormalizeToSelf(coeff2);
    // DoubleArrays.expToSelf(coeff2);
    // if (!DoubleArrays.equals(coeff1, coeff2, 1e-8)) {
    // System.out.println(Arrays.toString(coeff1));top
    // System.out.println(Arrays.toString(coeff2));
    // logger.severe("Not the same!");
    // } else {
    // logger.info("Same!");
    // }
    return computeCoefficients(ySums, mSums, ymSums, instance, docIndex);
  }

  @VisibleForTesting
  static double[] computeCoefficients(double[] ySums, double[] mSums, double[][] ymSums, 
      SparseFeatureVector instance, int i) {
    double[] coeffs = new double[ySums.length * mSums.length];
    int coeffIdx = 0;
    for (int c = 0; c < ySums.length; c++) {
      for (int d = 0; d < mSums.length; d++) {
        // n.b. no -logSumCountOfYAndM[c] here, because we take care of that in the ysums
        coeffs[coeffIdx++] = ySums[c] + mSums[d] + ymSums[c][d]; 
      }
    }
    return coeffs;
  }
  
  double[] computeCoefficientsUsingJoint(SparseFeatureVector instance, int i) {
    double lambda = getDocumentWeight(i);
    int numLabels = numLabels();
    double[] coeffs = new double[numLabels * numLabels];
    int coeffIdx = 0;
    for (int c = 0; c < numLabels; c++) {
      for (int d = 0; d < numLabels; d++) {
        incrementCounts(i, instance, c, d, lambda);
        coeffs[coeffIdx++] = logJoint();
        decrementCounts(i, instance, lambda);
      }
    }
    return coeffs;
  }

  public double logJoint() {
    double logJoint = 0.0;
    /* theta */
    // the following term is constant and may be dropped (and is in the complete conditionals, 
    // so it must be dropped here, too, if they are to match)
    // logJoint -= Gamma.logGamma(DoubleArrays.sum(DoubleArrays.exp(logCountOfY))); 
    for (int k = 0; k < logCountOfY.length; k++) {
      logJoint += Gamma.logGamma(Math.exp(logCountOfY[k]));
    }

    /* mu */
    // TODO(rhaertel): could use GammaFuntions.logBeta here and elsewhere. Note,
    // however that we may have some values that we've cached for the complete
    // conditionals that make "manual" computation faster. If, however, we are only
    // caching such values for this computation, it is probably not worth the
    // maintenance burden.
    for (int k = 0; k < logCountOfYAndM.length; k++) {
      logJoint -= Gamma.logGamma(Math.exp(logSumCountOfYAndM[k]));
      for (int kPrime = 0; kPrime < logCountOfYAndM.length; kPrime++) {
        logJoint += Gamma.logGamma(Math.exp(logCountOfYAndM[k][kPrime]));
      }
    }

    /* phi */
    for (int k = 0; k < countOfMAndX.length; k++) {
      logJoint -= Gamma.logGamma(numFeaturesPerM[k]);
      for (int f = 0; f < countOfMAndX[k].length; f++) {
        logJoint += Gamma.logGamma(countOfMAndX[k][f]);
      }
    }

    /* alpha */
    for (int j = 0; j < countOfJYAndA.length; j++) {
      for (int k = 0; k < countOfJYAndA[j].length; k++) {
        logJoint -= Gamma.logGamma(numAnnsPerJAndY[j][k]);
        for (int kPrime = 0; kPrime < countOfJYAndA[j][k].length; kPrime++) {
          logJoint += Gamma.logGamma(countOfJYAndA[j][k][kPrime]);
        }
      }
    }

    return logJoint;
  }
  
  // Important : also updates y[docIndex] and m[docIndex]
  void incrementCounts(int docIndex, SparseFeatureVector doc, int nextY, int nextM, double lambda) {
    y[docIndex] = nextY;
    int numAnnotators = numAnnotators();
    int numLabels = numLabels();
    // "before" sanity check
    assert isValidLogSumCountOfYAndM(nextY, logCountOfYAndM, logSumCountOfYAndM);

    logCountOfY[nextY] = Math.log(Math.exp(logCountOfY[nextY]) + lambda);
    for (int j = 0; j < numAnnotators; j++) {
      for (int k = 0; k < numLabels; k++) {
        assert lambda==1: "lambdas should only be non-1 for unannotated documents, but is "+lambda+". This should never happen.";
        countOfJYAndA[j][nextY][k] += a[docIndex][j][k];
      }
      numAnnsPerJAndY[j][nextY] += docJCount[docIndex][j];
    }

    m[docIndex] = nextM;
    doc.scaleAndAddTo(countOfMAndX[nextM], lambda);
    numFeaturesPerM[nextM] += docSize[docIndex] * lambda; // neither instances nor docsizes are scaled 
    logCountOfYAndM[nextY][nextM] = Math.log(Math.exp(logCountOfYAndM[nextY][nextM]) + lambda);
    logSumCountOfYAndM[nextY] = Math.log(Math.exp(logSumCountOfYAndM[nextY]) + lambda);
    // "after" sanity check + 1
    assert isValidLogSumCountOfYAndM(nextY, logCountOfYAndM, logSumCountOfYAndM);
  }

  @VisibleForTesting
  void decrementCounts(int docIndex, SparseFeatureVector doc, double lambda) {
    int curY = y[docIndex];
    int numAnnotators = numAnnotators();
    int numLabels = numLabels();
    // "before" sanity check
    assert isValidLogSumCountOfYAndM(curY, logCountOfYAndM, logSumCountOfYAndM);

    logCountOfY[curY] = Math.log(Math.exp(logCountOfY[curY]) - lambda);
    for (int j = 0; j < numAnnotators; j++) {
      for (int k = 0; k < numLabels; k++) {
        assert lambda==1: "lambdas should only be non-1 for unannotated documents, but is "+lambda+". This should never happen.";
        countOfJYAndA[j][curY][k] -= a[docIndex][j][k];
      }
      numAnnsPerJAndY[j][curY] -= docJCount[docIndex][j];
    }

    int curM = m[docIndex];
    doc.scaleAndSubtractFrom(countOfMAndX[curM], lambda);
    numFeaturesPerM[curM] -= docSize[docIndex] * lambda;
    logCountOfYAndM[curY][curM] = Math.log(Math.exp(logCountOfYAndM[curY][curM]) - lambda);
    logSumCountOfYAndM[curY] = Math.log(Math.exp(logSumCountOfYAndM[curY]) - lambda);
    // "after" sanity check
    assert isValidLogSumCountOfYAndM(curY, logCountOfYAndM, logSumCountOfYAndM);
  }

  private static boolean isValidLogSumCountOfYAndM(int c, double[][] logCountOfYAndM, double[] logSumCountOfYAndM) {
    double q1 = Math.exp(logSumCountOfYAndM[c]);
    double q2 = DoubleArrays.sum(DoubleArrays.exp(logCountOfYAndM[c])); 
    return Math2.doubleEquals(q1, q2, 1e-9);
  }

  /** {@inheritDoc} */
  @Override
  public Map<String,Integer> getInstanceIndices() {
    return instanceIndices;
  }

  /** {@inheritDoc} */
  @Override
  public DatasetLabeler getIntermediateLabeler() {
    final MultiAnnModel thisModel = this;
    return new DatasetLabeler() {
      @Override
      public Predictions label(Dataset trainingInstances, Dataset heldoutInstances) {
        return new MultiAnnDatasetLabeler(thisModel, 
            new PrintWriter(ByteStreams.nullOutputStream()), 
            true, DiagonalizationMethod.NONE, false, 0, trainingInstances, rnd).label(trainingInstances, heldoutInstances);
      }
    };
  }


  
}
