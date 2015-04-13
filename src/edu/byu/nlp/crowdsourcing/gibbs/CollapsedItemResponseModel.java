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
package edu.byu.nlp.crowdsourcing.gibbs;

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
import edu.byu.nlp.crowdsourcing.CrowdsourcingUtils;
import edu.byu.nlp.crowdsourcing.MultiAnnDatasetLabeler;
import edu.byu.nlp.crowdsourcing.MultiAnnModel;
import edu.byu.nlp.crowdsourcing.MultiAnnModelBuilders;
import edu.byu.nlp.crowdsourcing.MultiAnnModelBuilders.AbstractMultiAnnModelBuilder;
import edu.byu.nlp.crowdsourcing.MultiAnnState;
import edu.byu.nlp.crowdsourcing.MultiAnnState.CollapsedItemResponseState;
import edu.byu.nlp.crowdsourcing.gibbs.BlockCollapsedMultiAnnModelMath.DiagonalizationMethod;
import edu.byu.nlp.crowdsourcing.PriorSpecification;
import edu.byu.nlp.crowdsourcing.TrainableMultiAnnModel;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.DatasetInstance;
import edu.byu.nlp.data.types.SparseFeatureVector;
import edu.byu.nlp.math.optimize.ConvergenceCheckers;
import edu.byu.nlp.math.optimize.IterativeOptimizer;
import edu.byu.nlp.math.optimize.IterativeOptimizer.ReturnType;
import edu.byu.nlp.math.optimize.ValueAndObject;
import edu.byu.nlp.stats.RandomGenerators;
import edu.byu.nlp.stats.SymmetricDirichletMultinomialDiagonalMatrixMAPOptimizable;
import edu.byu.nlp.stats.SymmetricDirichletMultinomialMatrixMAPOptimizable;
import edu.byu.nlp.util.DoubleArrays;
import edu.byu.nlp.util.IntArrayCounter;
import edu.byu.nlp.util.Matrices;
import edu.byu.nlp.util.MatrixAverager;
import edu.byu.nlp.util.Pair;

/**
 * @author pfelt
 * 
 */
public class CollapsedItemResponseModel extends TrainableMultiAnnModel {

  private static final boolean USE_LOG_JOINT_FOR_COEFFS = false;

  private static final Logger logger = LoggerFactory.getLogger(CollapsedItemResponseModel.class);

  private final PriorSpecification priors;

  // Reference to data
  final Dataset data;

  // Assignment counter
  private final IntArrayCounter yMarginals;

  // Assignments
  private final int[] y; // inferred 'true' label assignments
  // len(y)=numdocs, type(y[i])=label

  // Sufficient statistics
  private final double[] logCountOfY; // (replaces theta)
  private final double[][][] countOfJYAndA; // (replaces alpha)

  // Cached values for efficiency
  private double[][] numAnnsPerJAndY; // [annotator][y]; \sum_k' countOfJYAndA[j][y][k']
                                            // (similar to numFeaturesPerM)

  // Also cached, but part of the data (no need to update)
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
    protected void hookPostInitYandM(int[] y, int[] m) {
      // ensure that y and m are perfectly the same so that derived
      // counts are consistent when we remove m
      for (int i=0; i<y.length; i++){
        m[i] = y[i];
      }
    }

    /** {@inheritDoc} */
    @Override
    protected MultiAnnModel build(PriorSpecification priors, Dataset data,
        Map<String, Integer> instanceIndices,
        Map<Integer, Integer> instanceLabels, int[][][] a, int[] y, int[] m,
        double[] logCountOfY, double[][] logCountOfYAndM,
        double[][] countOfMAndX, double[][][] countOfJYAndA,
        double[] logSumCountOfYAndM, double[] numFeaturesPerM,
        double[] docSize, double[][] numAnnsPerJAndY, int[][] docJCount,
        double initialTemp, double[] lambdas, int[] gold,
        RandomGenerator rnd) {
      return new CollapsedItemResponseModel(priors, data, instanceIndices, instanceLabels, a, y, logCountOfY,
          countOfJYAndA, numAnnsPerJAndY, docJCount, initialTemp, lambdas, gold, rnd);
    }


  }

  @VisibleForTesting
  CollapsedItemResponseModel(PriorSpecification priors, Dataset data, Map<String,Integer> instanceIndices, 
      Map<Integer, Integer> instanceLabels, int[][][] a, int[] y,
      double[] logCountOfY, 
      double[][][] countOfJYAndAnn, 
      double[][] numAnnsPerJAndY, int[][] docJCount, 
      double initialTemp, double[] lambdas, int[] gold, RandomGenerator rnd) {
    this.priors = priors;
    this.data = data;
    this.instanceIndices=instanceIndices;
    this.instanceLabels=instanceLabels;
    this.a = a;
    this.countOfJYAndA = countOfJYAndAnn;
    this.y = y;
    this.yMarginals = new IntArrayCounter(y.length, logCountOfY.length);
    this.yMarginals.increment(y); // include initial values
    this.logCountOfY = logCountOfY;
    this.numAnnsPerJAndY = numAnnsPerJAndY;
    this.docJCount = docJCount;
    this.temp = initialTemp;
    this.lambdas=lambdas;
    this.gold=gold;
    this.rnd = rnd;
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
      currentSample = new CollapsedItemResponseState(y, logCountOfY, countOfJYAndA, data, instanceIndices, rnd);
    }
    return currentSample;
  }

  public IntArrayCounter getMarginalYs(){
    return yMarginals;
  }

  public double[][] confusionMatrixY(){
    return BlockCollapsedMultiAnnModelMath.confusionMatrix(
        null, gold, y, numLabels(), data);
  }

  public double getTemp() {
    return temp;
  }

  public void setTemp(double temp) {
    this.temp = temp;
    // this indicates a change in sampling settings and triggers a history reset
    yMarginals.reset();
    yMarginals.increment(y); // initialize with y state
  }

  @VisibleForTesting
  double[] getLogCountOfY() {
    return logCountOfY;
  }

  @VisibleForTesting
  double[][][] getCountOfJYAndA() {
    return countOfJYAndA;
  }

  @VisibleForTesting
  double[][] getNumAnnsPerJAndY() {
    return numAnnsPerJAndY;
  }

  @VisibleForTesting
  int[][][] getA() {
    return a;
  }

  public PriorSpecification getPriors() {
    return priors;
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
      maximizeY(docIndex, instance.asFeatureVector());
      ++docIndex;
    }
    
    // do hyper updates
    if (priors.getInlineHyperparamTuning()){
      updateBTheta();
      updateBGamma();
    }
  }
  
  @Override
  public void maximizeY() {
    maximize();
  }

  @Override
  public void maximizeM() {
    // ignore
  }

  
  public void sample() {
    currentSample = null; // invalidate cached values
    // enumerate instances
    int docIndex = 0;
    for (DatasetInstance instance : data) {
      sampleY(docIndex, instance.asFeatureVector());
      ++docIndex;
    }
    yMarginals.increment(y);
    
    // do hyper updates
    if (priors.getInlineHyperparamTuning()){
      updateBTheta();
      updateBGamma();
    }
  }

  @Override
  public void sampleY() {
    sample();
  }

  @Override
  public void sampleM() {
    // ignore
  }

  public void sampleY(int docIndex, SparseFeatureVector instance) {
    double lambda = getDocumentWeight(docIndex);
    decrementCounts(docIndex, instance, lambda);

    double[] coeffs = computeCoefficients(docIndex, instance, 
        numAnnotators(), numLabels());
    if (temp != 1.0) {
      DoubleArrays.divideToSelf(coeffs, temp);
    }

    int nextY = RandomGenerators.nextIntUnnormalizedLogProbs(rnd, coeffs);
    incrementCounts(docIndex, instance, nextY, lambda);
  }

  public void maximizeY(int docIndex, SparseFeatureVector instance) {
    double lambda = getDocumentWeight(docIndex);
    
    decrementCounts(docIndex, instance, lambda);
    
    double[] coeffs = computeCoefficients(docIndex, instance, numAnnotators(), numLabels());
    
    int nextY = DoubleArrays.argMax(coeffs);
    incrementCounts(docIndex, instance, nextY, lambda);
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
    double lambda = lambdas==null? 1: lambdas[docIndex]; 
    return BlockCollapsedMultiAnnModelMath.computeYSums(docIndex, instanceLabels, logCountOfY, countOfJYAndA, 
        numAnnsPerJAndY, a, docJCount, numAnnotators, lambda);

  }

  double[] computeCoefficientsUsingJoint(SparseFeatureVector instance, int i) {
    double lambda = getDocumentWeight(i);
    int numLabels = numLabels();
    double[] coeffs = new double[numLabels];
    int coeffIdx = 0;
    for (int c = 0; c < numLabels; c++) {
      incrementCounts(i, instance, c, lambda);
      coeffs[coeffIdx++] = logJoint();
      decrementCounts(i, instance, lambda);
    }
    return coeffs;
  }

  public double logJoint() {
    double logJoint = 0.0;
    /* theta */
    for (int k = 0; k < logCountOfY.length; k++) {
      logJoint += Gamma.logGamma(Math.exp(logCountOfY[k]));
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

  // Important : also updates y[docIndex] 
  void incrementCounts(int docIndex, SparseFeatureVector doc, int nextY, double lambda) {
    
    y[docIndex] = nextY;
    int numAnnotators = numAnnotators();
    int numLabels = numLabels();

    logCountOfY[nextY] = Math.log(Math.exp(logCountOfY[nextY]) + lambda);
    for (int j = 0; j < numAnnotators; j++) {
      for (int k = 0; k < numLabels; k++) {
        countOfJYAndA[j][nextY][k] += a[docIndex][j][k];
      }
      numAnnsPerJAndY[j][nextY] += docJCount[docIndex][j];
    }

  }

  @VisibleForTesting
  void decrementCounts(int docIndex, SparseFeatureVector doc, double lambda) {
    
    int curY = y[docIndex];
    int numAnnotators = numAnnotators();
    int numLabels = numLabels();

    logCountOfY[curY] = Math.log(Math.exp(logCountOfY[curY]) - lambda);
    for (int j = 0; j < numAnnotators; j++) {
      for (int k = 0; k < numLabels; k++) {
        countOfJYAndA[j][curY][k] -= a[docIndex][j][k];
      }
      numAnnsPerJAndY[j][curY] -= docJCount[docIndex][j];
    }

  }


  @Override
  public Map<String,Integer> getInstanceIndices() {
    return instanceIndices;
  }

  /** {@inheritDoc} */
  @Override
  public IntArrayCounter getMarginalMs() {
    return getMarginalYs(); // m is equal to y in this model
  }

  /** {@inheritDoc} */
  @Override
  public MatrixAverager getMarginalYMs() {
    MatrixAverager diagonMatrixAverager = new MatrixAverager(numLabels(), numLabels());
    diagonMatrixAverager.increment(getCurrentState().getMu()); // mus is a dummy diagonal value
    return diagonMatrixAverager;
  }
  
  private void updateBTheta() {
    logger.debug("optimizing btheta in light of most recent topic assignments");
    subtractPriorsFromCounts();
    
    double oldValue = priors.getBTheta();
    double[][] thetaData = new double[][]{DoubleArrays.exp(logCountOfY)};
    IterativeOptimizer optimizer = new IterativeOptimizer(ConvergenceCheckers.relativePercentChange(PriorSpecification.HYPERPARAM_LEARNING_CONVERGENCE_THRESHOLD));
    SymmetricDirichletMultinomialMatrixMAPOptimizable o = SymmetricDirichletMultinomialMatrixMAPOptimizable.newOptimizable(thetaData,2,2);
    ValueAndObject<Double> optimum = optimizer.optimize(o, ReturnType.HIGHEST, true, oldValue);
    priors.setBTheta(optimum.getObject());
    
    addPriorsToCounts();
    logger.info("new btheta="+priors.getBTheta()+" old btheta="+oldValue);
  }
  
  private void updateBGamma() {
    int numClasses = data.getInfo().getNumClasses();
    logger.debug("optimizing bgamma in light of most recent document labels");
    subtractPriorsFromCounts();

    double[][] gammaHyperParams = calculateGammaHyperParams();
    Pair<Double,Double> oldValue = Pair.of(gammaHyperParams[0][0], gammaHyperParams[0][1]);
    IterativeOptimizer optimizer = new IterativeOptimizer(ConvergenceCheckers.relativePercentChange(PriorSpecification.HYPERPARAM_LEARNING_CONVERGENCE_THRESHOLD));
    SymmetricDirichletMultinomialDiagonalMatrixMAPOptimizable o = SymmetricDirichletMultinomialDiagonalMatrixMAPOptimizable.newOptimizable(countOfJYAndA, 2, 2);
    ValueAndObject<Pair<Double,Double>> optimum = optimizer.optimize(o, ReturnType.HIGHEST, true, oldValue);
    double newDiag = optimum.getObject().getFirst();
    double newOffDiag = optimum.getObject().getSecond();
    // setting priors here means the new values will be used to increment counts below
    double newCGamma = newDiag + (numClasses-1)*newOffDiag;
    priors.setBGamma(newDiag/newCGamma);
    priors.setCGamma(newCGamma);
    
    addPriorsToCounts();
    logger.info("new bgamma="+optimum.getObject()+" old bgamma="+oldValue);
  }

  private void addPriorsToCounts(){
    // btheta
    DoubleArrays.expToSelf(logCountOfY);
    DoubleArrays.addToSelf(logCountOfY, priors.getBTheta());
    DoubleArrays.logToSelf(logCountOfY);

    // bgamma 
    double[][] gammaHyperParams = calculateGammaHyperParams();
    for (int j=0; j<data.getInfo().getNumAnnotators(); j++){
      Matrices.addToSelf(countOfJYAndA[j], gammaHyperParams);
    }
    updateDerivedCounts(); // pre-compute some derived counts 
  }
  
  private void subtractPriorsFromCounts(){
    DoubleArrays.expToSelf(logCountOfY);
    DoubleArrays.subtractToSelf(logCountOfY, priors.getBTheta());
    for (int i=0; i<logCountOfY.length; i++){
      logCountOfY[i] = Math.max(0, logCountOfY[i]); // ensure no floating point error "negatives" mess us up 
    }
    DoubleArrays.logToSelf(logCountOfY);

    // bgamma 
    double[][] gammaHyperParams = calculateGammaHyperParams();
    for (int j=0; j<data.getInfo().getNumAnnotators(); j++){
      Matrices.subtractFromSelf(countOfJYAndA[j], gammaHyperParams);
    }
    updateDerivedCounts(); // pre-compute some derived counts 
  }
  
  private void updateDerivedCounts(){
    this.numAnnsPerJAndY = MultiAnnModelBuilders.numAnnsPerJAndY(this.countOfJYAndA);
  }
  
  private double[][] calculateGammaHyperParams(){
    int numClasses = data.getInfo().getNumClasses();
    double[][] gammaHyperParams = new double[numClasses][numClasses];
    CrowdsourcingUtils.initializeConfusionMatrixWithPrior(gammaHyperParams, priors.getBGamma(), priors.getCGamma());
    return gammaHyperParams;
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
