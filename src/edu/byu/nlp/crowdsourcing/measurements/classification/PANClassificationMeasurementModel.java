/**
 * Copyright 2015 Brigham Young University
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
package edu.byu.nlp.crowdsourcing.measurements.classification;

import java.util.Map;

import org.apache.commons.math3.random.RandomGenerator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cc.mallet.types.Dirichlet;

import com.google.common.base.Preconditions;

import edu.byu.nlp.classify.data.DatasetLabeler;
import edu.byu.nlp.classify.eval.Predictions;
import edu.byu.nlp.crowdsourcing.PriorSpecification;
import edu.byu.nlp.crowdsourcing.measurements.MeasurementModelBuilder;
import edu.byu.nlp.crowdsourcing.measurements.MeasurementExpectation;
import edu.byu.nlp.crowdsourcing.measurements.classification.ClassificationMeasurementExpectations.ScaledMeasurementExpectation;
import edu.byu.nlp.crowdsourcing.models.meanfield.MeanFieldMultiRespModel;
import edu.byu.nlp.data.measurements.ClassificationMeasurements.ClassificationAnnotationMeasurement;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.DatasetInstance;
import edu.byu.nlp.data.types.Measurement;
import edu.byu.nlp.math.GammaFunctions;
import edu.byu.nlp.util.DoubleArrays;
import edu.byu.nlp.util.Matrices;

/**
 * @author plf1
 *
 */
public class PANClassificationMeasurementModel implements ClassificationMeasurementModel{
  private static final Logger logger = LoggerFactory.getLogger(PANClassificationMeasurementModel.class);
  private static boolean SCALE_MEASUREMENTS = true;
  
  // Main experiment prior
  // defines an almost perfectly trusted annotator (for the proportion measurements of the main experiment)
  public static final double TRUSTED_ALPHA = 1e6, TRUSTED_BETA = 1.1;
  // Qualitative example prior (alice, bob, carol, dave)
//  // defines an almost perfectly trusted annotator (for "dave" in the artificial experiment with alice/bob/carol/dave)
//  public static final double TRUSTED_ALPHA = 2, TRUSTED_BETA = 1.1;

  private State state;
  private RandomGenerator rnd;

  
  public PANClassificationMeasurementModel(State state, RandomGenerator rnd) {
    this.state=state;
    this.rnd=rnd;
  }
  

  /** {@inheritDoc} */
  @Override
  public Double sample(String variableName, int iteration, String[] args) {
    throw new IllegalStateException("Sampling not implemented for measurement models");
  }

  
  /** {@inheritDoc} */
  @Override
  public Double maximize(String variableName, int iteration, String[] args) {
    // ignore variables--we only do joint maximization in variational
    if (!variableName.toLowerCase().equals("all")){
      logger.warn("Ignoring request to maximize "+variableName+". Instead maximizing 'all'");
    }
    
    // update the state in place, variable by variable 
    ClassificationMeasurementModelExpectations expectations = ClassificationMeasurementModelExpectations.from(state);
//    logger.info("LB after initialization: "+lowerBound(state,expectations));
    fitNuTheta(state);
//    logger.info("LB after fitNuTheta: "+lowerBound(state,expectations));
    fitNuSigma2(state, expectations);
//    logger.info("LB after fitNuSigma2: "+lowerBound(state,expectations));
    fitLogNuY(state, expectations);
//    logger.info("LB after fitLogNuY: "+lowerBound(state,expectations));
    
//    logger.info("nuTheta="+DoubleArrays.toString(state.getNuTheta()));
//    logger.info("nuSigma2=\n"+Matrices.toString(state.getNuSigma2()));
////    logger.info("logNuY=\n"+DoubleArrays.toString(state.getLogNuY()[0]));
//    logger.info("logNuY=\n"+Matrices.toString(state.getLogNuY()));

    // optimize hyperparams
    if (state.getPriors().getInlineHyperparamTuning()){
      fitBTheta();
      fitBSigma2();
    }

    return lowerBound(expectations);
  }




  /* ************************************** */
  /* ***** Parameter Optimization ********* */
  /* ************************************** */

  private void fitNuTheta(State state) {
    double[] nuTheta = state.getNuTheta();
    double[] classCounts = Matrices.sumOverFirst(Matrices.exp(state.getLogNuY()));
    
    for (int c=0; c<state.getNumClasses(); c++){
      // Dirichlet for each class
      nuTheta[c] = 
          state.getPriors().getBTheta() // symmetric class prior
          + classCounts[c]; // count
    }
    Preconditions.checkState(DoubleArrays.isFinite(nuTheta), "fitNuTheta() resulting in infinite parameter value"+nuTheta);
  }

  public static double getPriorAlpha(State state, int annotator){
    return annotator==state.getTrustedAnnotator()? TRUSTED_ALPHA: state.getPriors().getBGamma();
  }

  public static double getPriorBeta(State state, int annotator){
    return annotator==state.getTrustedAnnotator()? TRUSTED_BETA: state.getPriors().getCGamma();
  }

  private void fitNuSigma2(State state, ClassificationMeasurementModelExpectations expectations) {
    double[][] nuSigma2 = state.getNuSigma2();
    
    for (int j=0; j<state.getNumAnnotators(); j++){
      double priorAlpha = getPriorAlpha(state, j), priorBeta = getPriorBeta(state, j);
      
      // each inverse gamma distributed sigma2_j has two variational parameters: shape (nuSigma2[j][0]) and scale (nuSigma2[j][1]).

      // posterior alpha
      nuSigma2[j][0] = (state.getStaticCounts().getPerAnnotatorMeasurementCounts().getCount(j) / 2.0) + priorAlpha;
      Preconditions.checkState(Double.isFinite(nuSigma2[j][0]));
      
      // posterior beta
      // error sum
      double errorSum = 0;
      for (MeasurementExpectation<Integer> expectation: expectations.getExpectationsForAnnotator(j)){
        double tau_jk = expectation.getMeasurement().getValue(); 
        if (SCALE_MEASUREMENTS){
          // empirical measurements are often already scaled between 0 and 1 
          // (because annotators aren't good at guessing what the normalizer should be)
          if (!state.getMeasurementsPreScaled()){ 
            double range = expectation.getRange().upperEndpoint() - expectation.getRange().lowerEndpoint();
            tau_jk = expectation.getMeasurement().getValue() / range; // scale the observation
          }
          expectation = ScaledMeasurementExpectation.from(expectation); // scale the expectation quantities
        }
        errorSum += Math.pow(tau_jk, 2);
        errorSum -= 2 * tau_jk * expectation.sumOfExpectedValuesOfSigma();
        errorSum += Math.pow(expectation.sumOfExpectedValuesOfSigma(), 2);
        errorSum -= expectation.piecewiseSquaredSumOfExpectedValuesOfSigma();
        errorSum += expectation.sumOfExpectedValuesOfSquaredSigma();
        Preconditions.checkState(Double.isFinite(errorSum));
      }
      
      nuSigma2[j][1] = priorBeta + (0.5 * errorSum); 
      Preconditions.checkState(Double.isFinite(nuSigma2[j][1]));
    }
  }


  private void fitLogNuY(State state, ClassificationMeasurementModelExpectations expectations) {
    // pre-calculate
    double[] digammaOfNuThetas = MeanFieldMultiRespModel.digammasOfArray(state.getNuTheta());
    double digammaOfSummedNuThetas = MeanFieldMultiRespModel.digammaOfSummedArray(state.getNuTheta());
    
    for (int i=0; i<state.getNumDocuments(); i++){
      
      for (int c=0; c<state.getNumClasses(); c++){
        // part 1 (identical to first part of meanfielditemresp.fitg
        double t1 = digammaOfNuThetas[c] - digammaOfSummedNuThetas;
        
        double t2 = 0;
        for (int j=0; j<state.getNumAnnotators(); j++){
          double postAlpha = state.getNuSigma2()[j][0], postBeta = state.getNuSigma2()[j][1];
          
          double t3 = postAlpha / (2 * postBeta);
          
          double t4 = 0;
          for (MeasurementExpectation<Integer> expectation: expectations.getExpectationsForAnnotatorInstanceAndLabel(j, i, c)){
            double error = 0;
//            Integer rawLabel = ((ClassificationMeasurement)expectation.getMeasurement()).getLabel();
            
            double tau_jk = expectation.getMeasurement().getValue(); 
            if (SCALE_MEASUREMENTS){
              // empirical measurements are often already scaled between 0 and 1 
              // (because annotators aren't good at guessing what the normalizer should be)
              if (!state.getMeasurementsPreScaled()){ 
                double range = expectation.getRange().upperEndpoint() - expectation.getRange().lowerEndpoint();
                tau_jk = expectation.getMeasurement().getValue() / range; // scale the observation
              }
              expectation = ScaledMeasurementExpectation.from(expectation); // scale the expectation quantities
            }
            
            double sigma_jk = expectation.featureValue(i, c);
            
            // -2 tau sigma(x,y)
            error -= 2 * tau_jk * sigma_jk;
            
            // sigma^2
            error += Math.pow(sigma_jk, 2);
            
            // 2 sigma sum_i!=i E[sigma]
            expectation.setSummandVisible(i, false);
            error += 2 * sigma_jk * expectation.sumOfExpectedValuesOfSigma();
            expectation.setSummandVisible(i, true);
            
            t4 += error;
          }

          t2 -= t3*t4;
        }

        state.getLogNuY()[i][c] = t1 + t2; 
        
      }
      DoubleArrays.logNormalizeToSelf(state.getLogNuY()[i]);
      Preconditions.checkState(DoubleArrays.isFinite(state.getLogNuY()[i]), "produced infinite logNuY: "+state.getLogNuY()[i]);
      expectations.updateLogNuY_i(i, state.getLogNuY()[i]);
    }
  }



  public double lowerBound(ClassificationMeasurementModelExpectations expectations) {
    return lowerBound(state, expectations);
  }

  public static double expectedLogP(State state, ClassificationMeasurementModelExpectations expectations) {

    // precalculate values
    double[] digammaOfNuThetas = MeanFieldMultiRespModel.digammasOfArray(state.getNuTheta());
    double digammaOfSummedNuThetas = MeanFieldMultiRespModel.digammaOfSummedArray(state.getNuTheta());
    double[][] nuY = Matrices.exp(state.getLogNuY());
    double[] classCounts = Matrices.sumOverFirst(nuY);
    double priorDelta = state.getPriors().getBTheta();
    
    // part1 - term 1
    double part1t1 = state.getStaticCounts().getLogLowerBoundConstant();
    
    // part1 - term 2
    double part1t2 = 0;
    for (int c=0; c<state.getNumClasses(); c++){
      double t1 = digammaOfNuThetas[c] - digammaOfSummedNuThetas;
      double t2 = priorDelta + classCounts[c] - 1;
      part1t2 += t1 * t2;
    }
    
    // part1 - terms 3 
    double part1t3 = 0;
    for (int j=0; j<state.getNumAnnotators(); j++){
      double priorAlpha = getPriorAlpha(state,j), priorBeta = getPriorBeta(state,j);
      double postAlpha = state.getNuSigma2()[j][0], postBeta = state.getNuSigma2()[j][1]; // IG variational params
      
      // part 1 - term 3
      double part1t3a = (-(priorAlpha + (state.getStaticCounts().getPerAnnotatorMeasurementCounts().getCount(j) / 2.0))) - 1;
      double part1t3b = Math.log(postBeta) - Dirichlet.digamma(postAlpha);
      part1t3 += part1t3a * part1t3b;
    
    }

    // part1 - term 4
    double part1t4 = 0;
    for (int j=0; j<state.getNumAnnotators(); j++){
      double priorBeta = getPriorBeta(state, j);
      double postAlpha = state.getNuSigma2()[j][0], postBeta = state.getNuSigma2()[j][1]; // IG variational params
      
      // error sum
      double errorSum = 0;
      for (MeasurementExpectation<Integer> expectation: expectations.getExpectationsForAnnotator(j)){
        double tau_jk = expectation.getMeasurement().getValue(); 
        if (SCALE_MEASUREMENTS){
          // empirical measurements are often already scaled between 0 and 1 
          // (because annotators aren't good at guessing what the normalizer should be)
          if (!state.getMeasurementsPreScaled()){ 
            double range = expectation.getRange().upperEndpoint() - expectation.getRange().lowerEndpoint();
            tau_jk = expectation.getMeasurement().getValue() / range; // scale the observation
          }
          expectation = ScaledMeasurementExpectation.from(expectation); // scale the expectation quantities
        }
        
        errorSum += Math.pow(tau_jk, 2);
        errorSum -= 2 * tau_jk * expectation.sumOfExpectedValuesOfSigma();
        errorSum += Math.pow(expectation.sumOfExpectedValuesOfSigma(), 2);
        errorSum -= expectation.piecewiseSquaredSumOfExpectedValuesOfSigma();
        errorSum += expectation.sumOfExpectedValuesOfSquaredSigma();
      }

      double part1t4a = -1 * postAlpha/postBeta; 
      double part1t4b = priorBeta + (0.5 * errorSum);
      part1t4 += part1t4a * part1t4b;
    
    }
    
    return part1t1 + part1t2 + part1t3 + part1t4;
  }
  
  public static double lowerBound(State state, ClassificationMeasurementModelExpectations expectations) {

    // precalculate values
    double[] digammaOfNuThetas = MeanFieldMultiRespModel.digammasOfArray(state.getNuTheta());
    double digammaOfSummedNuThetas = MeanFieldMultiRespModel.digammaOfSummedArray(state.getNuTheta());
    double[][] nuY = Matrices.exp(state.getLogNuY());
    double[] classCounts = Matrices.sumOverFirst(nuY);
    double priorDelta = state.getPriors().getBTheta();
    
    // part1 - term 1
    double part1t1 = state.getStaticCounts().getLogLowerBoundConstant();
    
    // part1 - term 2
    double part1t2 = 0;
    for (int c=0; c<state.getNumClasses(); c++){
      double t1 = digammaOfNuThetas[c] - digammaOfSummedNuThetas;
      double t2 = priorDelta + classCounts[c] - 1;
      part1t2 += t1 * t2;
    }
    
    // part1 - terms 3 
    double part1t3 = 0;
    for (int j=0; j<state.getNumAnnotators(); j++){
      double priorAlpha = getPriorAlpha(state, j), priorBeta = getPriorBeta(state, j);
      double postAlpha = state.getNuSigma2()[j][0], postBeta = state.getNuSigma2()[j][1]; // IG variational params
      
      // part 1 - term 3
      double part1t3a = (-(priorAlpha + (state.getStaticCounts().getPerAnnotatorMeasurementCounts().getCount(j) / 2.0))) - 1;
      double part1t3b = Math.log(postBeta) - Dirichlet.digamma(postAlpha);
      part1t3 += part1t3a * part1t3b;
    
    }

    // part1 - term 4
    double part1t4 = 0;
    for (int j=0; j<state.getNumAnnotators(); j++){
      double priorBeta = getPriorBeta(state, j);
      double postAlpha = state.getNuSigma2()[j][0], postBeta = state.getNuSigma2()[j][1]; // IG variational params
      
      // error sum
      double errorSum = 0;
      for (MeasurementExpectation<Integer> expectation: expectations.getExpectationsForAnnotator(j)){
        double tau_jk = expectation.getMeasurement().getValue(); 
        if (SCALE_MEASUREMENTS){
          // empirical measurements are often already scaled between 0 and 1 
          // (because annotators aren't good at guessing what the normalizer should be)
          if (!state.getMeasurementsPreScaled()){ 
            double range = expectation.getRange().upperEndpoint() - expectation.getRange().lowerEndpoint();
            tau_jk = expectation.getMeasurement().getValue() / range; // scale the observation
          }
          expectation = ScaledMeasurementExpectation.from(expectation); // scale the expectation quantities
        }
        
        errorSum += Math.pow(tau_jk, 2);
        errorSum -= 2 * tau_jk * expectation.sumOfExpectedValuesOfSigma();
        errorSum += Math.pow(expectation.sumOfExpectedValuesOfSigma(), 2);
        errorSum -= expectation.piecewiseSquaredSumOfExpectedValuesOfSigma();
        errorSum += expectation.sumOfExpectedValuesOfSquaredSigma();
      }

      double part1t4a = -1 * postAlpha/postBeta; 
      double part1t4b = priorBeta + (0.5 * errorSum);
      part1t4 += part1t4a * part1t4b;
    
    }
    
    double expQlogP = part1t1 + part1t2 + part1t3 + part1t4;

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////

    // part 2 - term 1
    double part2t1 = - GammaFunctions.logBeta(state.getNuTheta());

    // part 2 - term 2
    double part2t2 = 0;
    for (int c=0; c<state.getNumClasses(); c++){
      double part2t2a = digammaOfNuThetas[c] - digammaOfSummedNuThetas;
      double part2t2b = state.getNuTheta()[c] - 1;
      part2t2 += part2t2a * part2t2b;
    }

    // part 2 - term 3
    double part2t3 = 0;
    for (int j=0; j<state.getNumAnnotators(); j++){
      double postAlpha = state.getNuSigma2()[j][0], postBeta = state.getNuSigma2()[j][1];
      
      part2t3 -= Dirichlet.logGamma(postAlpha)
          - Dirichlet.digamma(postAlpha) * (postAlpha + 1)
          + Math.log(postBeta)
          + postAlpha
      ;
    }

    // y
    double part2t4 = 0;
    for (int i=0; i<state.getNumDocuments(); i++){
      for (int k=0; k<state.getNumClasses(); k++){
        part2t4 += nuY[i][k] * state.getLogNuY()[i][k];
      }
    }
    
    double expQlogQ = part2t1 + part2t2 + part2t3 + part2t4;
    
    double elbo = expQlogP - expQlogQ;
    
    // Sanity checks. We can calculate various DL divergences from 
    // the terms above. Because ANY valid divergence should be >0, 
    // this gives us a way of sanity-checking some of the intermediate 
    // terms
    Preconditions.checkState(expQlogQ - expQlogP >0, "Sanity test failed! -ELBO is a KL divergence and must be >0");
    // paul: not sure these are valid KL divergences
//    Preconditions.checkState((part2t1 + part2t2) - expQlogP > 0, "Sanity test failed! All KL divergence and must be >0");
//    Preconditions.checkState(part2t3 - expQlogP > 0, "Sanity test failed! All KL divergence and must be >0");
//    Preconditions.checkState(part2t4 - expQlogP > 0, "Sanity test failed! All KL divergence and must be >0");
    
    return elbo;
  }

  
  /**
   * Set the ys to a smoothed empirical fit derived by iterating over all 
   * of the 'annotation' type measurements.
   */
  public void empiricalFit(){
    Dataset data = state.getData();
    
    // empirical 'annotation' distribution is derived by treating measurement values 
    // as log values and normalizing. A lack of annotations is interpreted as a value 
    // of log 1 = 0, so we are effectively doing add-1 smoothing on all counts.
    // Interpreting values as log values allows us to use negative and positive values.
    for (DatasetInstance instance: data){
      int docIndex = getInstanceIndices().get(instance.getInfo().getRawSource());
      
      // clear values (just in case they aren't already 0)
      for (int c=0; c<state.getNumClasses(); c++){
        state.getLogNuY()[docIndex][c] = 0;
      }
      
    }
    // increment with measured annotation values
    for (Measurement measurement: data.getMeasurements()){
      // empirical
      if (measurement instanceof ClassificationAnnotationMeasurement){
        ClassificationAnnotationMeasurement classificationMeasurement = (ClassificationAnnotationMeasurement) measurement;
        int docIndex = getInstanceIndices().get(classificationMeasurement.getDocumentSource());
        state.getLogNuY()[docIndex][classificationMeasurement.getLabel()] += classificationMeasurement.getValue();
//        // RANDOM initialization (for debugging)
//        state.getLogNuY()[docIndex][rnd.nextInt(state.getNumClasses())] += 1;
      }
    }
    Matrices.logNormalizeRowsToSelf(state.getLogNuY());
    
    // now set theta and sigma2 by fit
    ClassificationMeasurementModelExpectations expectations = ClassificationMeasurementModelExpectations.from(state);
    fitNuTheta(state);
    fitNuSigma2(state, expectations);
    
  }

  /* ************************************** */
  /* **** Hyperparameter Optimization ***** */
  /* ************************************** */
  private void fitBSigma2() {
    // TODO Auto-generated method stub
    
  }


  private void fitBTheta() {
    // TODO Auto-generated method stub
    
  }
  

  /* ************************************** */
  /* ***** Model Output ******************* */
  /* ************************************** */
  /** {@inheritDoc} */
  @Override
  public DatasetLabeler getIntermediateLabeler() {
    final ClassificationMeasurementModel thisModel = this;
    return new DatasetLabeler() {
      @Override
      public Predictions label(Dataset trainingInstances, Dataset heldoutInstances) {
        return new ClassificationMeasurementModelLabeler(thisModel).label(trainingInstances, heldoutInstances);
      }
    };
  }
  

  /** {@inheritDoc} */
  @Override
  public State getCurrentState() {
    return state;
  }


  /** {@inheritDoc} */
  @Override
  public Map<String, Integer> getInstanceIndices() {
    return state.getInstanceIndices();
  }

  
  /** {@inheritDoc} */
  @Override
  public double[] fitOutOfCorpusInstance(DatasetInstance instance) {
    return null; // NA for this model
  }
  
  
  
  public static class Builder extends MeasurementModelBuilder{
    /** {@inheritDoc} */
    @Override
    protected ClassificationMeasurementModel buildModel(PriorSpecification priors, Dataset data, int[] y,
        StaticMeasurementModelCounts staticCounts, Map<String, Integer> instanceIndices, 
        boolean measurementsPreScaled, int trustedMeasurementAnnotator, RandomGenerator rnd) {
      double[] nuTheta = new double[data.getInfo().getNumClasses()];
      double[][] nuSigma2 = new double[data.getInfo().getNumAnnotators()][2];
      double[][] logNuY = new double[data.getInfo().getNumDocuments()][data.getInfo().getNumClasses()];
      ClassificationMeasurementModel.State state = 
          new ClassificationMeasurementModel.State.Builder()
            .setData(data)
            .setPriors(priors)
            .setInstanceIndices(instanceIndices)
            .setStaticCounts(staticCounts)
            .setNuTheta(nuTheta)
            .setNuSigma2(nuSigma2)
            .setLogNuY(logNuY)
            .setTrustedAnnotator(trustedMeasurementAnnotator)
            .setMeasurementsPreScaled(measurementsPreScaled)
            .build()
            ;
      // create model and initialize variational parameters with an empirical fit
      PANClassificationMeasurementModel model = new PANClassificationMeasurementModel(state,rnd); 
      model.empiricalFit();
      
      return model; 
    }

  }


}
