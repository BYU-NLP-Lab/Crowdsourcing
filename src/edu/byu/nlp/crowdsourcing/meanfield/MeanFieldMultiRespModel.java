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

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.special.Gamma;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cc.mallet.types.Dirichlet;

import com.google.common.collect.Lists;

import edu.byu.nlp.classify.data.DatasetLabeler;
import edu.byu.nlp.classify.eval.Predictions;
import edu.byu.nlp.crowdsourcing.CrowdsourcingUtils;
import edu.byu.nlp.crowdsourcing.MultiAnnModel;
import edu.byu.nlp.crowdsourcing.MultiAnnModelBuilders.AbstractMultiAnnModelBuilder;
import edu.byu.nlp.crowdsourcing.MultiAnnState;
import edu.byu.nlp.crowdsourcing.PriorSpecification;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.DatasetInstance;
import edu.byu.nlp.data.types.SparseFeatureVector.EntryVisitor;
import edu.byu.nlp.dataset.Datasets;
import edu.byu.nlp.math.GammaFunctions;
import edu.byu.nlp.math.optimize.ConvergenceCheckers;
import edu.byu.nlp.math.optimize.IterativeOptimizer;
import edu.byu.nlp.math.optimize.IterativeOptimizer.ReturnType;
import edu.byu.nlp.math.optimize.ValueAndObject;
import edu.byu.nlp.stats.DirichletDistribution;
import edu.byu.nlp.stats.SymmetricDirichletMultinomialDiagonalMatrixMAPOptimizable;
import edu.byu.nlp.stats.SymmetricDirichletMultinomialMatrixMAPOptimizable;
import edu.byu.nlp.util.DoubleArrays;
import edu.byu.nlp.util.IntArrayCounter;
import edu.byu.nlp.util.IntArrays;
import edu.byu.nlp.util.Matrices;
import edu.byu.nlp.util.MatrixAverager;
import edu.byu.nlp.util.Pair;

/**
 * @author pfelt
 */
public class MeanFieldMultiRespModel extends AbstractMeanFieldMultiAnnModel {

  private static final Logger logger = LoggerFactory.getLogger(MeanFieldMultiRespModel.class);

  private static double INITIALIZATION_SMOOTHING = 1e-6; 
  
//  private static double LOG_CATEGORICAL_SMOOTHING = Math.log(1e-100);
  
  PriorSpecification priors;
  double[][] muParams;
  double[][][] gammaParams;
  
  private Dataset data;
  private ArrayList<DatasetInstance> instances;
  private int[][][] a; // annotations dim=NxJxK
  double[] docSizes;
  private Map<String, Integer> instanceIndices;
  private RandomGenerator rnd;
  public VariationalParams vars, newvars;

  // cached values
  private double[] digammaOfPis;
  private double digammaOfSummedPis;
  private double[][] digammaOfTaus;
  private double[][][] digammaOfNus;
  private double[] digammaOfSummedTaus;
  private double[][] digammaOfSummedNus;
  private double[][] digammaOfLambdas;
  private double[] digammaOfSummedLambda;

  
  static class VariationalParams{
    double[][] logg; // g(y) dim=NxK
    double[][] logh; // h(m) dim=NxK
    double[] pi;  // pi(theta) dim=J
    double[][][] nu; // nu(gamma) dim=JxKxK
    double[][] tau; // tau(mu) dim=KxK
    double[][] lambda; // lambda(phi) dim=KxF
    public VariationalParams(int numClasses, int numAnnotators, int numInstances, int numFeatures){
      this.logg = new double[numInstances][numClasses];
      this.logh = new double[numInstances][numClasses];
      this.pi = new double[numClasses];
      this.nu = new double[numAnnotators][numClasses][numClasses];
      this.tau = new double[numClasses][numClasses];
      this.lambda = new double[numClasses][numFeatures];
    }
    public void clonetoSelf(VariationalParams other){
      this.logg = Matrices.clone(other.logg);
      this.logh = Matrices.clone(other.logh);
      this.pi = other.pi.clone();
      this.nu = Matrices.clone(other.nu);
      this.tau = Matrices.clone(other.tau);
      this.lambda = Matrices.clone(other.lambda);
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

      // data counts
      double[][] countOfXAndF = Datasets.toFeatureArray(data);

      // data
      List<Map<Integer, Double>> x = Datasets.toSparseFeatureArray(data);
      
      // priors
      double[][] muParams = new double[logCountOfY.length][logCountOfY.length];
      CrowdsourcingUtils.initializeConfusionMatrixWithPrior(muParams, priors.getBMu(), priors.getCMu());
      double[][][] gammaParams = new double[countOfJYAndA.length][logCountOfY.length][logCountOfY.length];
      for (int j=0; j<countOfJYAndA.length; j++){
        CrowdsourcingUtils.initializeConfusionMatrixWithPrior(gammaParams[j], priors.getBGamma(j), 1);
      }
      
      // create model and initialize with empirical counts
      MeanFieldMultiRespModel model = new MeanFieldMultiRespModel(priors,a,countOfXAndF,muParams,gammaParams,instanceIndices,data,rnd);
      model.empiricalFit();
      
      return model;
    }

  }

  
  public MeanFieldMultiRespModel(PriorSpecification priors, int[][][] a, double[][] countOfXAndF,  
      double[][] muParams, double[][][] gammaParams, Map<String,Integer> instanceIndices, Dataset data, RandomGenerator rnd) {
    this.priors=priors;
    this.a=a;
    this.data=data;
    this.instances = Lists.newArrayList(data);
    this.docSizes = Datasets.countDocSizes(data);
    this.muParams=muParams;
    this.gammaParams=gammaParams;
    this.instanceIndices=instanceIndices;
    this.rnd=rnd;
    this.vars = new VariationalParams(muParams.length,gammaParams.length,a.length,countOfXAndF[0].length);
    this.newvars = new VariationalParams(muParams.length,gammaParams.length,a.length,countOfXAndF[0].length);
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
    // initialize h
    for (int i=0; i<vars.logg.length; i++){
//      vars.logh[i] = DirichletDistribution.logSample(DoubleArrays.exp(vars.logg[i]), rnd);
      vars.logh[i] = DirichletDistribution.logSample(DoubleArrays.of(1, numClasses()), rnd);
      DoubleArrays.normalizeAndLogToSelf(vars.logh[i]);
//      vars.logh[i] = vars.logg[i].clone();
    }
    // init params based on g and h
    fitPi(vars.pi);
    fitNu(vars.nu);
    fitTau(vars.tau);
    fitLambda(vars.lambda);
    
    // make sure both sets of parameter values match
    newvars.clonetoSelf(vars);
  }
  
  /** {@inheritDoc} */
  @Override
  public MultiAnnState getCurrentState() {
    
    return new MeanFieldMultiAnnState(
        Matrices.exp(vars.logg), 
        Matrices.exp(vars.logh),  
        vars.pi, 
        vars.nu, 
        vars.tau, // tau 
        vars.lambda, // lambda
        data, instanceIndices);
  }

  /** {@inheritDoc} */
  @Override
  public void maximize() {
    fitG(newvars.logg);
    fitH(newvars.logh);
    fitPi(newvars.pi);
    fitNu(newvars.nu);
    fitTau(newvars.tau);
    fitLambda(newvars.lambda);
    
    // swap in new values
    VariationalParams tmpvars = this.vars;
    this.vars = this.newvars;
    this.newvars = tmpvars;
    
    // optimize hyperparameters
    if (priors.getInlineHyperparamTuning()){
      fitBTheta();
      fitBPhi();
      fitBGamma();
      // TODO: fit mu hypers?
    }
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
  public void fitTau(double[][] tau) {
    double[][] g = Matrices.exp(vars.logg);
    double[][] h = Matrices.exp(vars.logh);
    
    for (int k1=0; k1<numClasses(); k1++){
      for (int k2=0; k2<numClasses(); k2++){
        tau[k1][k2] = muParams[k1][k2];
        // TODO: can this be done more efficiently?
        for (int i=0; i<numInstances(); i++){
          tau[k1][k2] += g[i][k1] * h[i][k2];
        }
      }
    }
  }
  public void fitLambda(final double[][] lambda) {
    double[][] h = Matrices.exp(vars.logh);
    
//    // this form implements the math clearly, but is inefficient 
//    // TODO: iterate sparsely through non-zero features
//    for (int k=0; k<numClasses(); k++){
//      for (int f=0; f<numFeatures(); f++){
//        lambda[k][f] = priors.getBPhi();
//        for (int i=0; i<numInstances(); i++){
//          double xval = (x.get(i).containsKey(f))? x.get(i).get(f): 0;
//          lambda[k][f] += xval * h[i][k];
//        }
//      }
//    }

  // this version reverses the loops over i and f in order to loop sparsely over the 
  // non-zero features of the data. 
  // since this version is hard to read, I'm leaving the version above commented out.
  for (int k=0; k<numClasses(); k++){

    // initialize
    for (int f=0; f<numFeatures(); f++){
      lambda[k][f] = priors.getBPhi();
    }
    
    for (int i=0; i<numInstances(); i++){
      final double hik = h[i][k];
      final int kk = k;
      
      // for each feature f
      instances.get(i).asFeatureVector().visitSparseEntries(new EntryVisitor() {
        @Override
        public void visitEntry(int f, double x_if) {
          lambda[kk][f] += x_if * hik;
        }
      });
    }
  }
    
  }
  public void fitG(double[][] logg) {
    double[][] h = Matrices.exp(vars.logh);
    
    // precalculate 
    double[] digammaOfPis = digammasOfArray(vars.pi);
    double digammaOfSummedPis = digammaOfSummedArray(vars.pi);
    double[][] digammaOfTaus = digammasOfMatrix(vars.tau);
    double[] digammaOfSummedTaus = digammasOfArraysSummedOverSecond(vars.tau);
    double[][][] digammaOfNus = digammasOfTensor(vars.nu);
    double[][] digammaOfSummedNus = digammasOfArraysSummedOverLast(vars.nu);
    
    for (int i=0; i<numInstances(); i++){
      fitG_i(logg[i],a[i],h[i],digammaOfPis,digammaOfSummedPis,digammaOfTaus,digammaOfSummedTaus,digammaOfNus,digammaOfSummedNus);
    }
  }
  /**
   * This part is factored out of the main fitG method to make generalization inference easier
   * (fit an arbitrary g given annotation info and all other params in a model)
   */
  public void fitG_i(double[] logg_i, int[][] a_i, double[] h_i, 
      double[] digammaOfPis, double digammaOfSummedPis, double[][] digammaOfTaus, double[] digammaOfSummedTaus, 
      double[][][] digammaOfNus, double[][] digammaOfSummedNus) {
    for (int k=0; k<numClasses(); k++){
      
      double term1 = 0;
      term1 += digammaOfPis[k] - digammaOfSummedPis;
      
      double term2 = 0;
      for (int k2=0; k2<numClasses(); k2++){
        term2 += h_i[k2] * (digammaOfTaus[k][k2] - digammaOfSummedTaus[k]);
      }
      
      double term3 = 0;
      for (int j=0; j<numAnnotators(); j++){
        for (int k2=0; k2<numClasses(); k2++){
          term3 += a_i[j][k2] * (digammaOfNus[j][k][k2] - digammaOfSummedNus[j][k]);
        }
      }
      
      logg_i[k] = term1 + term2 + term3; 
    }
    DoubleArrays.logNormalizeToSelf(logg_i);
  }
  public void fitH(double[][] logh) {
    // precalculate 
    double[][] digammaOfTaus = digammasOfMatrix(vars.tau);
    double[] digammaOfSummedTaus = digammasOfArraysSummedOverSecond(vars.tau);
    double[][] digammaOfLambdas = digammasOfMatrix(vars.lambda);
    double[] digammaOfSummedLambda = digammasOfArraysSummedOverSecond(vars.lambda);
    double[][] g = Matrices.exp(vars.logg);
    
    for (int i=0; i<numInstances(); i++){
      fitH_i(logh[i],instances.get(i),g[i],digammaOfTaus,digammaOfSummedTaus,digammaOfLambdas,digammaOfSummedLambda);
    }
  }
  public void fitH_i(double[] logh_i, DatasetInstance instance, double[] g_i, double[][] digammaOfTaus, double[] digammaOfSummedTaus, 
      final double[][] digammaOfLambdas, final double[] digammaOfSummedLambda){
    for (int k2=0; k2<numClasses(); k2++){
      
      double term1 = 0;
      for (int k=0; k<numClasses(); k++){
        term1 += g_i[k] * (digammaOfTaus[k][k2] - digammaOfSummedTaus[k]);
      }
      
      // original version: more clear but less efficient
//      double term2 = 0;
//      for (int f=0; f<numFeatures(); f++){
//        double phi1 = (x_i.containsKey(f))? x_i.get(f): 0;
//        double phi2 = digammaOfLambdas[k2][f] - digammaOfSummedLambda[k2];
//        term2 += phi1 * phi2;
//      }

      final double[] term2 = new double[]{0}; // this is only an array so it can be final AND mutable
      final int kk22 = k2;
      instance.asFeatureVector().visitSparseEntries(new EntryVisitor() {
        @Override
        public void visitEntry(int f, double x_if) {
          double phi1 = x_if;
          double phi2 = digammaOfLambdas[kk22][f] - digammaOfSummedLambda[kk22];
          term2[0] += phi1 * phi2;
        }
      });

      logh_i[k2] = term1 + term2[0]; 
    }
    DoubleArrays.logNormalizeToSelf(logh_i);
  }
  /** {@inheritDoc} */
  @Override
  public double[] fitOutOfCorpusInstance(DatasetInstance instance) {
    // precalculate
    if (digammaOfPis==null){
      digammaOfPis = MeanFieldMultiRespModel.digammasOfArray(vars.pi);
      digammaOfSummedPis = MeanFieldMultiRespModel.digammaOfSummedArray(vars.pi);
      digammaOfTaus = MeanFieldMultiRespModel.digammasOfMatrix(vars.tau);
      digammaOfSummedTaus = MeanFieldMultiRespModel.digammasOfArraysSummedOverSecond(vars.tau);
      digammaOfNus = MeanFieldMultiRespModel.digammasOfTensor(vars.nu);
      digammaOfSummedNus = MeanFieldMultiRespModel.digammasOfArraysSummedOverLast(vars.nu);
      digammaOfLambdas = MeanFieldMultiRespModel.digammasOfMatrix(vars.lambda);
      digammaOfSummedLambda = MeanFieldMultiRespModel.digammasOfArraysSummedOverSecond(vars.lambda);
    }

    // annotations
    int[][] a_i = Datasets.compileDenseAnnotations(instance, numClasses(), numAnnotators());
    
    double[] logg_i = DoubleArrays.of(1, numClasses());
    double[] logh_i = DoubleArrays.of(1, numClasses());
    
//    // fit g_i and h_i iteratively (doing it right)
//    int iterations = 0;
//    double curr = Double.MIN_VALUE, improvement = Double.MIN_VALUE;
//    do {
//      fitG_i(logg_i, a_i, DoubleArrays.exp(logh_i), digammaOfPis, digammaOfSummedPis, digammaOfTaus, digammaOfSummedTaus, digammaOfNus, digammaOfSummedNus);
//      fitH_i(logh_i, instance, DoubleArrays.exp(logg_i), digammaOfTaus, digammaOfSummedTaus, digammaOfLambdas, digammaOfSummedLambda);
//      if (iterations%MultiAnnModelTraining.MAXIMIZE_BATCH_SIZE==0){
//        double next = logJoint();
//        improvement = next-curr;
//        curr = next;
//        logger.info("iteration "+iterations+" out-of-corpus bound "+curr);
//      }
//    } while(improvement>MultiAnnModelTraining.MAXIMIZE_IMPROVEMENT_THRESHOLD && iterations<MultiAnnModelTraining.MAXIMIZE_MAX_ITERATIONS);

    // fit g_i and h_i iteratively (doing it quick)
    for (int i=0; i<10; i++){
      fitG_i(logg_i, a_i, DoubleArrays.exp(logh_i), digammaOfPis, digammaOfSummedPis, digammaOfTaus, digammaOfSummedTaus, digammaOfNus, digammaOfSummedNus);
      fitH_i(logh_i, instance, DoubleArrays.exp(logg_i), digammaOfTaus, digammaOfSummedTaus, digammaOfLambdas, digammaOfSummedLambda);
    }
    
    return DoubleArrays.exp(logg_i);
  }
  

  private void fitBTheta() {
    logger.info("optimizing btheta in light of most recent posterior assignments");
    double oldValue = priors.getBTheta();
    IterativeOptimizer optimizer = new IterativeOptimizer(ConvergenceCheckers.relativePercentChange(PriorSpecification.HYPERPARAM_LEARNING_CONVERGENCE_THRESHOLD));
    double perDocumentClassCounts[][] = Matrices.exp(vars.logg);
    double[][] dat = new double[][]{Matrices.sumOverFirst(perDocumentClassCounts)};
    SymmetricDirichletMultinomialMatrixMAPOptimizable o = SymmetricDirichletMultinomialMatrixMAPOptimizable.newOptimizable(dat,2,2);
    ValueAndObject<Double> optimum = optimizer.optimize(o, ReturnType.HIGHEST, true, oldValue);
    double newValue = optimum.getObject();
    priors.setBTheta(newValue);
    logger.info("new btheta="+newValue+" old btheta="+oldValue);
  }
  
  private void fitBPhi() {
    logger.info("optimizing bphi in light of most recent posterior assignments");
    double oldValue = priors.getBPhi();
    IterativeOptimizer optimizer = new IterativeOptimizer(ConvergenceCheckers.relativePercentChange(PriorSpecification.HYPERPARAM_LEARNING_CONVERGENCE_THRESHOLD));
    // TODO: here we are tying ALL bphi hyperparams (even across classes). In this case, inference actually doesn't matter
    // Alternatively, we could fit each class symmetric dirichlet separately. Or even fit each individual parameter (maybe w/ gamma prior).
    double[][] perClassVocabCounts = perClassVocab(data, instances, vars.logg);
    SymmetricDirichletMultinomialMatrixMAPOptimizable o = SymmetricDirichletMultinomialMatrixMAPOptimizable.newOptimizable(perClassVocabCounts,2,2);
    ValueAndObject<Double> optimum = optimizer.optimize(o, ReturnType.HIGHEST, true, oldValue);
    double newValue = optimum.getObject();
    priors.setBPhi(newValue);
    logger.info("new bphi="+newValue+" old bphi="+oldValue);
  }
  private double[][][] annotatorConfusions;
  private void fitBGamma() {
	if (annotatorConfusions==null){
		// track empirical annotator confusion according to inferred correct classes
	    this.annotatorConfusions = new double[gammaParams.length][gammaParams[0].length][gammaParams[0].length]; 
	}
	calculateCurrentAnnotatorConfusions(annotatorConfusions, a, vars.logg);
    logger.info("optimizing bgamma in light of most recent topic assignments");
    Pair<Double,Double> oldValue = Pair.of(gammaParams[0][0][0], gammaParams[0][0][1]);
    IterativeOptimizer optimizer = new IterativeOptimizer(ConvergenceCheckers.relativePercentChange(PriorSpecification.HYPERPARAM_LEARNING_CONVERGENCE_THRESHOLD));
    SymmetricDirichletMultinomialDiagonalMatrixMAPOptimizable o = SymmetricDirichletMultinomialDiagonalMatrixMAPOptimizable.newOptimizable(annotatorConfusions, 2, 2);
    ValueAndObject<Pair<Double,Double>> optimum = optimizer.optimize(o, ReturnType.HIGHEST, true, oldValue);
    double newDiag = optimum.getObject().getFirst();
    double newOffDiag = optimum.getObject().getSecond();
    for (int j=0; j<numAnnotators(); j++){
    	for (int k=0; k<numClasses(); k++){
    		for (int kprime=0; kprime<numClasses(); kprime++){
    			gammaParams[j][k][kprime] = (k==kprime)? newDiag: newOffDiag;
    		}
    	}
    }
    // setting priors allows driver class to report settled-on values
    double newCGamma = newDiag + (numClasses()-1)*newOffDiag;
    priors.setBGamma(newDiag/newCGamma);
    priors.setCGamma(newCGamma);
    logger.info("new bgamma="+optimum.getObject()+" old bgamma="+oldValue);
  }
  
  /** {@inheritDoc} */
  @Override
  public double logJoint() {
    double[][] g = Matrices.exp(vars.logg);
    double[][] h = Matrices.exp(vars.logh);
    double[] summedG = Matrices.sumOverFirst(g);
    double[] digammaOfPis = digammasOfArray(vars.pi);
    double digammaOfSummedPis = Dirichlet.digamma(DoubleArrays.sum(vars.pi));
    double[][] digammaOfTaus = digammasOfMatrix(vars.tau);
    double[] digammaOfSummedTaus = digammasOfArraysSummedOverSecond(vars.tau);
    double[][][] digammaOfNus = digammasOfTensor(vars.nu);
    double[][] digammaOfSummedNus = digammasOfArraysSummedOverLast(vars.nu);
    double[][] digammaOfLambdas = digammasOfMatrix(vars.lambda);
    double[] digammaOfSummedLambda = digammasOfArraysSummedOverSecond(vars.lambda);

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

    // term 1 - mu normalizer
    double t1munorm = 0;
    for (int k=0; k<numClasses(); k++){
      t1munorm += GammaFunctions.logBeta(muParams[k]);
    }
    
    // term 1 - mu
    double t1muterm = 0;
    for (int k=0; k<numClasses(); k++){
      for (int k2=0; k2<numClasses(); k2++){
        double ghterm = 0;
        for (int i=0; i<numInstances(); i++){
          ghterm += g[i][k] * h[i][k2];
        }
        
        double mu1 = muParams[k][k2] + ghterm - 1;
        double mu2 = digammaOfTaus[k][k2] - digammaOfSummedTaus[k];
        t1muterm += mu1 * mu2;
      }
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

    // term 1 - phi normalizer
    double t1phinorm = 0;
    for (int k=0; k<numClasses(); k++){
      t1phinorm += GammaFunctions.logBetaSymmetric(priors.getBPhi(), numFeatures());
    }
    
    // term 1 - phi
    double t1phiterm = 0;
    for (int k=0; k<numClasses(); k++){
//      // slow but easy to read
//      for (int f=0; f<numFeatures(); f++){
//        double xhterm = 0;
//        for (int i=0; i<numInstances(); i++){
//          // TODO: very slow
//          double xval = (x.get(i).containsKey(f))? x.get(i).get(f): 0;
//          xhterm += xval * h[i][k];
//        }
//        
//        double phi1 = this.priors.getBPhi() + xhterm - 1;
//        double phi2 = digammaOfLambdas[k][f] - digammaOfSummedLambda[k];
//        t1phiterm += phi1 * phi2;
//      }
      
      // faster but hard to read (distributive law, regroup, reverse sums over i and f)
      // compute phi expectations and deal with (prior-1)
      final double[] expected_phi = new double[numFeatures()];
      for (int f=0; f<numFeatures(); f++){
        expected_phi[f] = digammaOfLambdas[k][f] - digammaOfSummedLambda[k];
        t1phiterm += (this.priors.getBPhi() - 1) * expected_phi[f];
      }
      for (int i=0; i<numInstances(); i++){
        final double[] phidataterm = new double[]{0}; // only an array to be final AND mutable 
        instances.get(i).asFeatureVector().visitSparseEntries(new EntryVisitor() {
          @Override
          public void visitEntry(int f, double x_if) {
            phidataterm[0] += x_if * expected_phi[f];
          }
        });
        
        t1phiterm += h[i][k] * phidataterm[0];
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
    
    // term 1 - x
    double t1xterm = 0;
    for (int i=0; i<numInstances(); i++){
      double x1 = Gamma.logGamma(docSizes[i] + 1);
//      // easier to read (dense) version for x2
//      double x2 = 0;
//      for (int f=0; f<numFeatures(); f++){
//        x2 += Gamma.logGamma(x.get(i).get(f) + 1);
//      }
      // harder to read (sparse) version for x2
      final double[] x2 = new double[]{0}; // this is an array only so it can be final AND mutable
      instances.get(i).asFeatureVector().visitSparseEntries(new EntryVisitor() {
        @Override
        public void visitEntry(int index, double value) {
          x2[0] += Gamma.logGamma(value + 1);
        }
      });
      t1xterm += x1 - x2[0];
    }
    
    double term1 = t1thetaterm + t1muterm + t1gammaterm + t1phiterm - t1thetanorm - t1munorm - t1gammanorm - t1phinorm + t1aterm + t1xterm;

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
    
    // term 2 - mu normalizer
    double t2munorm = 0;
    for (int k=0; k<numClasses(); k++){
      t2munorm += GammaFunctions.logBeta(vars.tau[k]);
    }
    
    // term 2 - mu
    double t2muterm = 0;
    for (int k=0; k<numClasses(); k++){
      for (int k2=0; k2<numClasses(); k2++){
        double mu1 = vars.tau[k][k2] - 1;
        double mu2 = digammaOfTaus[k][k2] - digammaOfSummedTaus[k];
        
        t2muterm += mu1 * mu2;
      }
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
    
    // term 2 - phi normalizer
    double t2phinorm = 0;
    for (int k=0; k<numClasses(); k++){
      t2phinorm += GammaFunctions.logBeta(vars.lambda[k]);
    }
    
    // term 2 - phi
    double t2phiterm = 0;
    for (int k=0; k<numClasses(); k++){
      for (int f=0; f<numFeatures(); f++){
        double phi1 = vars.lambda[k][f] - 1;
        double phi2 = digammaOfLambdas[k][f] - digammaOfSummedLambda[k];
        
        t2phiterm += phi1 * phi2;
      }
    }
    
    // term 2 - g and h
    double t2gterm = 0;
    double t2hterm = 0;
    for (int i=0; i<numInstances(); i++){
      double gterm = 0;
      for (int k=0; k<numClasses(); k++){
        gterm += g[i][k] * vars.logg[i][k];
      }
      
      double hterm = 0;
      for (int k=0; k<numClasses(); k++){
        hterm += h[i][k] * vars.logh[i][k];
      }
      
      t2gterm += gterm;
      t2hterm += hterm;
    }
    
    double term2 = t2thetaterm + t2muterm + t2gammaterm + t2phiterm + t2gterm + t2hterm - t2thetanorm - t2gammanorm - t2munorm - t2phinorm;

    // debugging the validity of the variational bound
    double div1 = (t2thetaterm + t2gterm - t2thetanorm) - (t1thetaterm - t1thetanorm);
    double div2 = (t2gammaterm - t2gammanorm) - (t1gammaterm - t1gammanorm);
    double div3 = (t2phiterm - t2phinorm) - (t1phiterm - t1phinorm);
    double div4 = (t2muterm + t2hterm - t2munorm) - (t1muterm - t1munorm);
    if (div1<0 || div2<0 || div3<0 || div4<0){
      throw new RuntimeException("invalid variational bound. All divergences should be non-negative, but "
          + "divergence1="+div1+" divergence2="+div2+" divergence3="+div3+" divergence4="+div4);
    }
    
    // ********
    // total
    // ********
    return term1 - term2;
  }

  

  public static double[][][] digammasOfTensor(double[][][] orig){
    double[][][] digammas = new double[orig.length][orig[0].length][orig[0][0].length];
    for (int i=0; i<orig.length; i++){
      digammas[i] = digammasOfMatrix(orig[i]);
    }
    return digammas;
  }
  public static double[][] digammasOfMatrix(double[][] orig){
    double[][] digammas = new double[orig.length][orig[0].length];
    for (int i=0; i<orig.length; i++){
      digammas[i] = digammasOfArray(orig[i]);
    }
    return digammas;
  }
  public static double[] digammasOfArray(double[] orig){
    double[] digammas = new double[orig.length];
    for (int i=0; i<orig.length; i++){
      digammas[i] = Dirichlet.digamma(orig[i]);
    }
    return digammas;
  }
  public static double digammaOfSummedArray(double[] orig){
    return Dirichlet.digamma(DoubleArrays.sum(orig));
  }
  public static double[] digammasOfArraysSummedOverSecond(double[][] orig){
    double[] summedArrays = Matrices.sumOverSecond(orig);
    return digammasOfArray(summedArrays);
  }
  public static double[][] digammasOfArraysSummedOverLast(double[][][] orig){
    double[][] summedArrays = new double[orig.length][orig[0].length];
    for (int i=0; i<summedArrays.length; i++){
      summedArrays[i] = digammasOfArraysSummedOverSecond(orig[i]);
    }
    return summedArrays;
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
  private int numFeatures(){
    return vars.lambda[0].length;
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
  public IntArrayCounter getMarginalYs() {
    throw new UnsupportedOperationException("not implemented");
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
  public DatasetLabeler getIntermediateLabeler() {
    final MultiAnnModel thisModel = this;
    return new DatasetLabeler() {
      @Override
      public Predictions label(Dataset trainingInstances, Dataset heldoutInstances) {
        return new MeanFieldMultiAnnLabeler(thisModel).label(trainingInstances, heldoutInstances);
      }
    };
  }
  

  
}
