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
package edu.byu.nlp.crowdsourcing.meanfield;

import java.util.Map;

import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.special.Gamma;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cc.mallet.types.Dirichlet;
import edu.byu.nlp.crowdsourcing.CrowdsourcingUtils;
import edu.byu.nlp.crowdsourcing.MultiAnnModel;
import edu.byu.nlp.crowdsourcing.MultiAnnModelBuilders.AbstractMultiAnnModelBuilder;
import edu.byu.nlp.crowdsourcing.MultiAnnState;
import edu.byu.nlp.crowdsourcing.PriorSpecification;
import edu.byu.nlp.crowdsourcing.TrainableMultiAnnModel;
import edu.byu.nlp.crowdsourcing.gibbs.CollapsedItemResponseModel;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.DatasetInstance;
import edu.byu.nlp.dataset.Datasets;
import edu.byu.nlp.math.GammaFunctions;
import edu.byu.nlp.math.optimize.ConvergenceCheckers;
import edu.byu.nlp.math.optimize.IterativeOptimizer;
import edu.byu.nlp.math.optimize.ValueAndObject;
import edu.byu.nlp.math.optimize.IterativeOptimizer.ReturnType;
import edu.byu.nlp.stats.SymmetricDirichletMultinomialMLEOptimizable;
import edu.byu.nlp.util.DoubleArrays;
import edu.byu.nlp.util.IntArrayCounter;
import edu.byu.nlp.util.IntArrays;
import edu.byu.nlp.util.Matrices;
import edu.byu.nlp.util.MatrixAverager;

/**
 * @author pfelt
 */
public class MeanFieldItemRespModel extends TrainableMultiAnnModel implements MeanFieldMultiAnnModel {

  private static final Logger logger = LoggerFactory.getLogger(CollapsedItemResponseModel.class);

  private static double INITIALIZATION_SMOOTHING = 1e-6; 
  
//  private static double LOG_CATEGORICAL_SMOOTHING = Math.log(1e-100);
  
  PriorSpecification priors;
  double[][] muParams;
  double[][][] gammaParams;
  
  private Dataset data;
  private int[][][] a; // annotations dim=NxJxK
  private Map<String, Integer> instanceIndices;
  private RandomGenerator rnd;
  public VariationalParams vars, newvars;

  // cached values
  private double[] digammaOfPis;
  private double[][][] digammaOfNus;
  private double digammaOfSummedPis;
  private double[][] digammaOfSummedNus;
  
  static class VariationalParams{
    double[][] logg; // g(y) dim=NxK
    double[] pi;  // pi(theta) dim=J
    double[][][] nu; // nu(gamma) dim=JxKxK
    public VariationalParams(int numClasses, int numAnnotators, int numInstances){
      this.logg = new double[numInstances][numClasses];
      this.pi = new double[numClasses];
      this.nu = new double[numAnnotators][numClasses][numClasses];
    }
    public void clonetoSelf(VariationalParams other){
      this.logg = Matrices.clone(other.logg);
      this.pi = other.pi.clone();
      this.nu = Matrices.clone(other.nu);
    }
  }
  
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
      
      // priors
      double[][] muParams = new double[logCountOfY.length][logCountOfY.length];
      CrowdsourcingUtils.initializeConfusionMatrixWithPrior(muParams, priors.getBMu(), priors.getCMu());
      double[][][] gammaParams = new double[countOfJYAndA.length][logCountOfY.length][logCountOfY.length];
      for (int j=0; j<countOfJYAndA.length; j++){
        CrowdsourcingUtils.initializeConfusionMatrixWithPrior(gammaParams[j], priors.getBGamma(j), 1);
      }
      
      // create model and initialize with empirical counts
      MeanFieldItemRespModel model = new MeanFieldItemRespModel(priors,a,muParams,gammaParams,instanceIndices,data,rnd);
      model.empiricalFit();
      
      return model;
    }

  }

  
  public MeanFieldItemRespModel(PriorSpecification priors, int[][][] a,  
      double[][] muParams, double[][][] gammaParams, Map<String,Integer> instanceIndices, Dataset data, RandomGenerator rnd) {
    this.priors=priors;
    this.a=a;
    this.data=data;
    this.muParams=muParams;
    this.gammaParams=gammaParams;
    this.instanceIndices=instanceIndices;
    this.rnd=rnd;
    this.vars = new VariationalParams(muParams.length,gammaParams.length,a.length);
    this.newvars = new VariationalParams(muParams.length,gammaParams.length,a.length);
  }

  
  public void empiricalFit(){
    // initialize g with (smoothed?) empirical distribution
    for (int i=0; i<a.length; i++){
      // g = empirical fit
      for (int j=0; j<a[i].length; j++){
        for (int k=0; k<a[i][j].length; k++){
          vars.logg[i][k] += a[i][j][k] + INITIALIZATION_SMOOTHING; 
        }
      }
      DoubleArrays.normalizeAndLogToSelf(vars.logg[i]);
    }
    
    // init params based on g and h
    fitPi(vars.pi);
    fitNu(vars.nu);
    
    // make sure both sets of parameter values match
    newvars.clonetoSelf(vars);
  }
  
  /** {@inheritDoc} */
  @Override
  public MultiAnnState getCurrentState() {

    return new MeanFieldMultiAnnState(
        Matrices.exp(vars.logg), 
        Matrices.exp(vars.logg), // h 
        vars.pi, 
        vars.nu, 
        Matrices.of(1, numClasses(), numClasses()), // tau 
        Matrices.of(1, numClasses(), numFeatures()), // lambda
        data, instanceIndices);
    
  }



  /** {@inheritDoc} */
  @Override
  public void maximize() {
    fitG(newvars.logg);
    fitPi(newvars.pi);
    fitNu(newvars.nu);
    
    // swap in new values
    VariationalParams tmpvars = this.vars;
    this.vars = this.newvars;
    this.newvars = tmpvars;
    
    // optimize hyperparams
    fitBTheta();
  }

  private static double HYPERPARAM_LEARNING_CONVERGENCE_THRESHOLD = 0.1;
  private void fitBTheta() {
    logger.info("optimizing btheta in light of most recent posterior assignments");
    double oldValue = priors.getBTheta();
    IterativeOptimizer optimizer = new IterativeOptimizer(ConvergenceCheckers.relativePercentChange(HYPERPARAM_LEARNING_CONVERGENCE_THRESHOLD));
    double perDocumentClassCounts[][] = Matrices.exp(vars.logg);
    SymmetricDirichletMultinomialMLEOptimizable o = SymmetricDirichletMultinomialMLEOptimizable.newOptimizable(perDocumentClassCounts,2,2);
    ValueAndObject<Double> optimum = optimizer.optimize(o, ReturnType.HIGHEST, true, oldValue);
    double newValue = optimum.getObject();
    priors.setBTheta(newValue);
    logger.info("new btheta="+newValue+" old btheta="+oldValue);
  }
  
  public void fitPi(double[] pi) {
    double[][] g = Matrices.exp(vars.logg);
    
    double[] summedClasses = Matrices.sumOverFirst(g);
    DoubleArrays.addToSelf(summedClasses, priors.getBTheta());
    for (int k=0; k<pi.length; k++){
      pi[k] = summedClasses[k];
    }
  }
  public void fitNu(double[][][] nu) {
    double[][] g = Matrices.exp(vars.logg);
    
    for (int j=0; j<numAnnotators(); j++){
      for (int k1=0; k1<numClasses(); k1++){
        for (int k2=0; k2<numClasses(); k2++){
          double nu1 = gammaParams[j][k1][k2];
          // TODO: can this be done more efficiently?
          double nu2 = 0;
          for (int i=0; i<numInstances(); i++){
            nu2 += a[i][j][k2] * g[i][k1];
          }
          
          nu[j][k1][k2] = nu1 + nu2; 
        }
      }
    }
  }
  public void fitG(double[][] logg) {
    
    // precalculate 
    double[] digammaOfPis = MeanFieldMultiRespModel.digammasOfArray(vars.pi);
    double digammaOfSummedPis = MeanFieldMultiRespModel.digammaOfSummedArray(vars.pi);
    double[][][] digammaOfNus = MeanFieldMultiRespModel.digammasOfTensor(vars.nu);
    double[][] digammaOfSummedNus = MeanFieldMultiRespModel.digammasOfArraysSummedOverLast(vars.nu);
    
    for (int i=0; i<numInstances(); i++){
      fitG_i(logg[i],a[i],digammaOfPis,digammaOfSummedPis,digammaOfNus,digammaOfSummedNus);
    }
  }
  public void fitG_i(double[] logg_i, int[][] a_i, double[] digammaOfPis, double digammaOfSummedPis, 
      double[][][] digammaOfNus, double[][] digammaOfSummedNus) {
    for (int k=0; k<numClasses(); k++){
      
      double term1 = 0;
      term1 += digammaOfPis[k] - digammaOfSummedPis;
      
      double term2 = 0;
      for (int j=0; j<numAnnotators(); j++){
        for (int k2=0; k2<numClasses(); k2++){
          term2 += a_i[j][k2] * (digammaOfNus[j][k][k2] - digammaOfSummedNus[j][k]);
        }
      }
      
      logg_i[k] = term1 + term2; 
    }
    DoubleArrays.logNormalizeToSelf(logg_i);
  }
  /** {@inheritDoc} */
  @Override
  public double[] fitOutOfCorpusInstance(DatasetInstance instance) {

    // precalculate 
    if (digammaOfPis==null){
      digammaOfPis = MeanFieldMultiRespModel.digammasOfArray(vars.pi);
      digammaOfSummedPis = MeanFieldMultiRespModel.digammaOfSummedArray(vars.pi);
      digammaOfNus = MeanFieldMultiRespModel.digammasOfTensor(vars.nu);
      digammaOfSummedNus = MeanFieldMultiRespModel.digammasOfArraysSummedOverLast(vars.nu);
    }
    
    // annotations
    int[][] a_i = Datasets.compileDenseAnnotations(instance, numClasses(), numAnnotators());

    double[] logg_i = new double[numClasses()];
    fitG_i(logg_i, a_i, digammaOfPis, digammaOfSummedPis, digammaOfNus, digammaOfSummedNus);
    return DoubleArrays.exp(logg_i);
  }
  /** {@inheritDoc} */
  @Override
  public double logJoint() {
    double[][] g = Matrices.exp(vars.logg);
    double[] summedG = Matrices.sumOverFirst(g);
    double[] digammaOfPis = MeanFieldMultiRespModel.digammasOfArray(vars.pi);
    double digammaOfSummedPis = Dirichlet.digamma(DoubleArrays.sum(vars.pi));
    double[][][] digammaOfNus = MeanFieldMultiRespModel.digammasOfTensor(vars.nu);
    double[][] digammaOfSummedNus = MeanFieldMultiRespModel.digammasOfArraysSummedOverLast(vars.nu);

    // *********
    // term 1
    // *********
    
    // term 1 - theta normalizer
    double t1thetanorm = GammaFunctions.logBetaSymmetric(priors.getBTheta(), numClasses());
    
    // term 1 - theta
    double t1thetaterm = 0;
    for (int k=0; k<numClasses(); k++){
      double theta1 = priors.getBTheta() + summedG[k] - 1;
      double theta2 = digammaOfPis[k] - digammaOfSummedPis;
      t1thetaterm += theta1 * theta2;
    }

    // term 1 - gamma normalizer
    double t1gammanorm = 0;
    for (int j=0; j<numAnnotators(); j++){
      for (int k=0; k<numClasses(); k++){
        t1gammanorm += GammaFunctions.logBeta(gammaParams[j][k]);
      }
    }
    
    // term 1 - gamma
    double t1gammaterm = 0;
    for (int j=0; j<numAnnotators(); j++){
      for (int k=0; k<numClasses(); k++){
        for (int k2=0; k2<numClasses(); k2++){
          double agterm = 0;
          for (int i=0; i<numInstances(); i++){
            agterm += this.a[i][j][k2] * g[i][k];
          }
          
          double gamma1 = this.gammaParams[j][k][k2] + agterm - 1;
          double gamma2 = digammaOfNus[j][k][k2] - digammaOfSummedNus[j][k];
          t1gammaterm += gamma1 * gamma2;
        }
      }
    }

    // term 1 - a 
    double t1aterm = 0;
    for (int i=0; i<numInstances(); i++){
      for (int j=0; j<numAnnotators(); j++){
        double a1 = Gamma.logGamma(IntArrays.sum(this.a[i][j]) + 1);
        double a2 = 0;
        for (int k=0; k<numClasses(); k++){
          a2 += Gamma.logGamma(this.a[i][j][k] + 1);
        }
        t1aterm += a1 - a2;
      }
    }
    
    double term1 = t1thetaterm + t1gammaterm - t1thetanorm - t1gammanorm + t1aterm;

    // *********
    // term 2
    // *********
    
    // term 2 - theta normalizer
    double t2thetanorm = GammaFunctions.logBeta(vars.pi);
    
    // term 2 - theta
    double t2thetaterm = 0;
    for (int k=0; k<numClasses(); k++){
      double theta1 = vars.pi[k] - 1;
      double theta2 = digammaOfPis[k] - digammaOfSummedPis;
      
      t2thetaterm += theta1 * theta2;
    }
    
    // term 2 - gamma normalizer
    double t2gammanorm = 0;
    for (int j=0; j<numAnnotators(); j++){
      for (int k=0; k<numClasses(); k++){
        t2gammanorm += GammaFunctions.logBeta(vars.nu[j][k]);
      }
    }
    
    // term 2 - gamma
    double t2gammaterm = 0;
    for (int j=0; j<numAnnotators(); j++){
      for (int k=0; k<numClasses(); k++){
        for (int k2=0; k2<numClasses(); k2++){
          double gamma1 = vars.nu[j][k][k2] - 1;
          double gamma2 = digammaOfNus[j][k][k2] - digammaOfSummedNus[j][k];
          
          t2gammaterm += gamma1 * gamma2;
        }
      }
    }
    
    // term 2 - g 
    double t2gterm = 0;
    for (int i=0; i<numInstances(); i++){
      double gterm = 0;
      for (int k=0; k<numClasses(); k++){
        gterm += g[i][k] * vars.logg[i][k];
      }
      
      t2gterm += gterm;
    }
    
    double term2 = t2thetaterm + t2gammaterm + t2gterm - t2thetanorm - t2gammanorm;
    
    // debugging the validity of the variational bound
    double div1 = (t2thetaterm + t2gterm - t2thetanorm) - (t1thetaterm - t1thetanorm);
    double div2 = (t2gammaterm - t2gammanorm) - (t1gammaterm - t1gammanorm);
    if (div1<0 || div2<0){
      throw new RuntimeException("invalid variational bound. All divergences should be non-negative, "
          + "but divergence1="+div1+" divergence2="+div2);
    }
    
    // ********
    // total
    // ********
    return term1 - term2;
  }

  

  
  /** {@inheritDoc} */
  @Override
  public Map<String, Integer> getInstanceIndices() {
    return instanceIndices;
  }
  /** {@inheritDoc} */
  @Override
  public double getDocumentWeight(int docIndex) {
    return 1;
  }



  
  
  
  
  
  private int numAnnotators(){
    return vars.nu.length;
  }
  private int numInstances(){
    return vars.logg.length;
  }
  private int numClasses(){
    return vars.pi.length;
  }
  private int numFeatures() {
    return data.getInfo().getNumFeatures();
  }
  
  
  
  
  /** {@inheritDoc} */
  @Override
  public void setTemp(double temp) {
    if (temp!=1){
      throw new UnsupportedOperationException("not implemented");
    }
  }
  
  /** {@inheritDoc} */
  @Override
  public void maximizeY() {
    logger.warn("maximizeY() defaults to maximize() for variational models.");
    maximize();
  }

  /** {@inheritDoc} */
  @Override
  public void maximizeM() {
    logger.warn("maximizeM() defaults to maximize() for variational models.");
    maximize();
  }

  /** {@inheritDoc} */
  @Override
  public void sample() {
    logger.warn("Sampling not available for variational model. Ignoring...");
  }

  /** {@inheritDoc} */
  @Override
  public void sampleY() {
    logger.warn("Sampling not available for variational model. Ignoring...");
  }

  /** {@inheritDoc} */
  @Override
  public void sampleM() {
    logger.warn("Sampling not available for variational model. Ignoring...");
  }


  public static void main(String[] args){
    System.out.println(Dirichlet.digamma(0));
    System.out.println(Dirichlet.digamma(0.0001));
    System.out.println(Dirichlet.digamma(0.001));
    System.out.println(Dirichlet.digamma(0.01));
    System.out.println(Dirichlet.digamma(0.1));
    System.out.println(Dirichlet.digamma(1));
    System.out.println(Dirichlet.digamma(10));
    System.out.println(Dirichlet.digamma(100));
    System.out.println(Dirichlet.digamma(1000));
  }


  /** {@inheritDoc} */
  @Override
  public IntArrayCounter getMarginalMs() {
    throw new UnsupportedOperationException("not implemented");
  }


  /** {@inheritDoc} */
  @Override
  public MatrixAverager getMarginalYMs() {
    throw new UnsupportedOperationException("not implemented");
  }

  /** {@inheritDoc} */
  @Override
  public IntArrayCounter getMarginalYs() {
    throw new UnsupportedOperationException("not implemented");
  }
  
  
}
