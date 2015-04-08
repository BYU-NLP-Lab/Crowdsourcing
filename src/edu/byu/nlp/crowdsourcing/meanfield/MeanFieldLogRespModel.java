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
import java.util.Map;

import org.apache.commons.math3.random.RandomGenerator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cc.mallet.classify.MaxEnt;
import cc.mallet.types.Dirichlet;

import com.google.common.collect.Lists;

import edu.byu.nlp.classify.MalletMaxentTrainer;
import edu.byu.nlp.crowdsourcing.CrowdsourcingUtils;
import edu.byu.nlp.crowdsourcing.MultiAnnModel;
import edu.byu.nlp.crowdsourcing.MultiAnnModelBuilders.AbstractMultiAnnModelBuilder;
import edu.byu.nlp.crowdsourcing.MultiAnnState;
import edu.byu.nlp.crowdsourcing.PriorSpecification;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.DatasetInstance;
import edu.byu.nlp.data.types.SparseFeatureVector.EntryVisitor;
import edu.byu.nlp.dataset.Datasets;
import edu.byu.nlp.math.optimize.ConvergenceCheckers;
import edu.byu.nlp.math.optimize.IterativeOptimizer;
import edu.byu.nlp.math.optimize.IterativeOptimizer.ReturnType;
import edu.byu.nlp.math.optimize.ValueAndObject;
import edu.byu.nlp.stats.SymmetricDirichletMultinomialDiagonalMatrixMAPOptimizable;
import edu.byu.nlp.stats.SymmetricDirichletMultinomialMatrixMAPOptimizable;
import edu.byu.nlp.util.DoubleArrays;
import edu.byu.nlp.util.IntArrayCounter;
import edu.byu.nlp.util.Matrices;
import edu.byu.nlp.util.MatrixAverager;
import edu.byu.nlp.util.Pair;

/**
 * @author pfelt
 */
public class MeanFieldLogRespModel extends AbstractMeanFieldMultiAnnModel {

  private static final Logger logger = LoggerFactory.getLogger(MeanFieldLogRespModel.class);

  private static double INITIALIZATION_SMOOTHING = 1e-6; 
  
//  private static double LOG_CATEGORICAL_SMOOTHING = Math.log(1e-100);
  
  PriorSpecification priors;
  double[][][] gammaParams;
  
  private Dataset data;
  private ArrayList<DatasetInstance> instances;
  private int[][][] a; // annotations dim=NxJxK
  double[] docSizes;
  private Map<String, Integer> instanceIndices;
  private RandomGenerator rnd;
  public VariationalParams vars, newvars;
  private MalletMaxentTrainer trainer;

  // cached values
  private double[][][] digammaOfNus;
  private double[][] digammaOfSummedNus;
  private double[][] maxLambda;

  
  static class VariationalParams{
    double[][] logg; // g(y) dim=NxK
    double[][][] nu; // nu(gamma) dim=JxKxK
    private MaxEnt maxent;
    public VariationalParams(int numClasses, int numAnnotators, int numInstances, int numFeatures){
      this.logg = new double[numInstances][numClasses];
      this.nu = new double[numAnnotators][numClasses][numClasses];
    }
    public void clonetoSelf(VariationalParams other){
      this.logg = Matrices.clone(other.logg);
      this.nu = Matrices.clone(other.nu);
      this.maxent = other.maxent; // TODO need to clone?
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
      double[][][] gammaParams = new double[countOfJYAndA.length][logCountOfY.length][logCountOfY.length];
      for (int j=0; j<countOfJYAndA.length; j++){
        CrowdsourcingUtils.initializeConfusionMatrixWithPrior(gammaParams[j], priors.getBGamma(j), 1);
      }
      
      // create model and initialize with empirical counts
      MalletMaxentTrainer trainer = MalletMaxentTrainer.build(data);
      MeanFieldLogRespModel model = new MeanFieldLogRespModel(trainer, priors,a,gammaParams,instanceIndices,data,rnd);
      model.empiricalFit();
      
      return model;
    }

  }

  
  public MeanFieldLogRespModel(MalletMaxentTrainer trainer, PriorSpecification priors, int[][][] a,  
      double[][][] gammaParams, Map<String,Integer> instanceIndices, Dataset data, RandomGenerator rnd) {
    this.trainer=trainer;
    this.priors=priors;
    this.a=a;
    this.data=data;
    this.docSizes = Datasets.countDocSizes(data);
    this.instances = Lists.newArrayList(data);
    this.gammaParams=gammaParams;
    this.instanceIndices=instanceIndices;
    this.rnd=rnd;
    this.vars = new VariationalParams(gammaParams[0].length,gammaParams.length,a.length,data.getInfo().getNumFeatures());
    this.newvars = new VariationalParams(gammaParams[0].length,gammaParams.length,a.length,data.getInfo().getNumFeatures());
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
    fitNu(vars.nu);
    vars.maxent = trainer.maxDataModel(Matrices.exp(vars.logg), vars.maxent);
    
    // make sure both sets of parameter values match
    newvars.clonetoSelf(vars);
  }
  
  /** {@inheritDoc} */
  @Override
  public MultiAnnState getCurrentState() {
    
    return new MeanFieldMultiAnnState(
        Matrices.exp(vars.logg), 
        Matrices.exp(vars.logg), // h  
        null, // pi
        vars.nu, 
        Matrices.of(1, numClasses(), numClasses()), // tau 
        null, // lambda
        vars.maxent,
        data, instanceIndices);
  }

  /** {@inheritDoc} */
  @Override
  public void maximize() {
    fitG(newvars.logg);
    fitNu(newvars.nu);
    newvars.maxent = trainer.maxDataModel(Matrices.exp(vars.logg), vars.maxent);
    
    // swap in new values
    VariationalParams tmpvars = this.vars;
    this.vars = this.newvars;
    this.newvars = tmpvars;
    
    // optimize hypers
    if (priors.getInlineHyperparamTuning()){
      fitBPhi();
      fitBGamma();
    }
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
    double[][][] digammaOfNus = MeanFieldMultiRespModel.digammasOfTensor(vars.nu);
    double[][] digammaOfSummedNus = MeanFieldMultiRespModel.digammasOfArraysSummedOverLast(vars.nu);
    double[][] maxLambda = trainer.maxWeights(vars.maxent, numClasses(), numFeatures());
    
    for (int i=0; i<numInstances(); i++){
      fitG_i(logg[i],a[i],instances.get(i),digammaOfNus,digammaOfSummedNus,maxLambda);
    }
  }
  public void fitG_i(double[] logg_i, int[][] a_i, DatasetInstance instance, 
      double[][][] digammaOfNus, double[][] digammaOfSummedNus, final double[][] maxLambda) {
    for (int k=0; k<numClasses(); k++){
      
      double term1 = 0;
      for (int j=0; j<numAnnotators(); j++){
        for (int k2=0; k2<numClasses(); k2++){
          term1 += a_i[j][k2] * (digammaOfNus[j][k][k2] - digammaOfSummedNus[j][k]);
        }
      }

      final double[] term2 = new double[]{0}; // this is only an array so it can be final AND mutable
      final int kk = k;
      term2[0] += maxLambda[kk][numFeatures()]; // class bias term
      instance.asFeatureVector().visitSparseEntries(new EntryVisitor() {
        @Override
        public void visitEntry(int f, double x_if) {
          double phi1 = x_if;
          double phi2 = maxLambda[kk][f];
          term2[0] += phi1 * phi2;
//          System.out.println("x_i"+f+"("+phi1+")"+" w_"+kk+f+"("+phi2+")");
        }
      });
      
      logg_i[k] = term1 + term2[0]; 
    }
//    // FIXME: DEBUG stuff
//    if (IntArrays.sum(a_i)==0){
//      System.out.println("my unnormalized log prediction: "+DoubleArrays.toString(logg_i));
//      Classification pred = vars.maxent.classify(MalletMaxentTrainer.convert(vars.maxent.getAlphabet(), vars.maxent.getLabelAlphabet(), instance));
//      LabelVector predvec = pred.getLabelVector();
//      double[] values = new double[numClasses()];
//      for (int l=0; l<values.length; l++){
//        values[l] = predvec.value(l);
//      }
//      
//      DoubleArrays.logNormalizeToSelf(logg_i);
//      Preconditions.checkState(Math.abs(DoubleArrays.sum(values)-1)<1e-6);
//      System.out.println(DoubleArrays.toString(values)+" vs \n"+DoubleArrays.toString(DoubleArrays.exp(logg_i))+"\n");
////      System.out.println("maxent getparams:");
////      System.out.println(DoubleArrays.toString(vars.maxent.getParameters()));
////      System.out.println("my params:");
////      System.out.println(Matrices.toString(maxLambda));
////      System.out.println("");
//    }
    
    DoubleArrays.logNormalizeToSelf(logg_i);

    
  }
  /** {@inheritDoc} */
  @Override
  public double[] fitOutOfCorpusInstance(DatasetInstance instance) {

    // precalculate 
    if (digammaOfNus==null){
      digammaOfNus = MeanFieldMultiRespModel.digammasOfTensor(vars.nu);
      digammaOfSummedNus = MeanFieldMultiRespModel.digammasOfArraysSummedOverLast(vars.nu);
      maxLambda = trainer.maxWeights(vars.maxent, numClasses(), numFeatures());
    }
    
    // annotations
    int[][] a_i = Datasets.compileDenseAnnotations(instance, numClasses(), numAnnotators());
    
//  // sanity check: mallet's classification code should return the *same* thing as our code on unanntoated examples. 
//    return vars.maxent.classify(MalletMaxentTrainer.convert(vars.maxent.getAlphabet(), vars.maxent.getLabelAlphabet(), instance)).getLabelVector().getValues();
    
    double[] logg_i = new double[numClasses()];
    fitG_i(logg_i, a_i, instance, digammaOfNus, digammaOfSummedNus, maxLambda);
    
    return DoubleArrays.exp(logg_i);
  }
  
  // Hack alert: I'm not sure what the bound should look like in this case, 
  // where we are half variational and half EM. 
  private static double jnt = 1; 
  @Override
  public double logJoint() {
    if (jnt<100){
      jnt++;
    }
    return jnt; 
    
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
    return data.getInfo().getNumClasses();
  }
  private int numFeatures(){
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
