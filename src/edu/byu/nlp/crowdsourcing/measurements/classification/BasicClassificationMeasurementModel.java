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
import edu.byu.nlp.classify.data.DatasetLabeler;
import edu.byu.nlp.classify.eval.Predictions;
import edu.byu.nlp.crowdsourcing.PriorSpecification;
import edu.byu.nlp.crowdsourcing.measurements.AbstractMeasurementModelBuilder;
import edu.byu.nlp.crowdsourcing.measurements.MeasurementExpectation;
import edu.byu.nlp.crowdsourcing.models.meanfield.MeanFieldMultiRespModel;
import edu.byu.nlp.data.measurements.ClassificationMeasurements.ClassificationAnnotationMeasurement;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.DatasetInstance;
import edu.byu.nlp.data.types.Measurement;
import edu.byu.nlp.util.Matrices;

/**
 * @author plf1
 *
 */
public class BasicClassificationMeasurementModel implements ClassificationMeasurementModel{
  private static final Logger logger = LoggerFactory.getLogger(BasicClassificationMeasurementModel.class);

  private State state;  

  
  public BasicClassificationMeasurementModel(State state) {
    this.state=state;
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
    
    // construct an updated state (without changing the old state yet)
    State newstate = state.copy();
    ClassificationMeasurementModelCounts counts = ClassificationMeasurementModelCounts.from(state);
    System.out.println("LB after initialization: "+lowerBound(newstate,counts));
//    for (int j=0; j<state.getNumAnnotators(); j++){
//      newstate.getNuSigma2()[j][0] = (new Random().nextDouble()+100)*10;
//      newstate.getNuSigma2()[j][1] = (new Random().nextDouble()+100)*10; 
//    }
//    System.out.println("LB after randomization of sigma2: "+lowerBound(newstate,counts));
    fitNuTheta(newstate.getNuTheta(), counts);
    System.out.println("LB after futNuTheta: "+lowerBound(newstate,counts));
    fitNuSigma2(newstate.getNuSigma2(), counts);
    System.out.println("LB after fitNuSigma2: "+lowerBound(newstate,counts));
    fitLogNuY(newstate.getLogNuY(), counts);
    System.out.println("LB after fitLogNuY: "+lowerBound(newstate,counts));

    
    // swap in new state all at once
    this.state = newstate;
    
    // optimize hyperparams
    if (state.getPriors().getInlineHyperparamTuning()){
      fitBTheta();
      fitBSigma2();
    }

    return lowerBound(counts);
  }




  /* ************************************** */
  /* ***** Parameter Optimization ********* */
  /* ************************************** */

  private void fitNuTheta(double[] nuTheta, ClassificationMeasurementModelCounts counts) {
    
    double[] classCounts = Matrices.sumOverFirst(Matrices.exp(state.getLogNuY()));
    
    for (int c=0; c<state.getNumClasses(); c++){
      // Dirichlet for each class
      nuTheta[c] = 
          state.getPriors().getBTheta() // symmetric class prior
          + classCounts[c]; // count
    }
  }

  
  private void fitNuSigma2(double[][] nuSigma2, ClassificationMeasurementModelCounts counts) {
    // alpha is shoe-horned into priors.bgamma; beta into priors.cgamma
    double alpha = state.getPriors().getBGamma(), beta = state.getPriors().getCGamma();
    for (int j=0; j<state.getNumAnnotators(); j++){
      // each inverse gamma distributed sigma2_j has two variational parameters: shape (nuSigma2[j][0]) and scale (nuSigma2[j][1]).

      // variational posterior shape parameter
      nuSigma2[j][0] = (state.getStaticCounts().getPerAnnotatorMeasurements().getCount(j) / 2.0) + alpha;
      
      // variational posterior scale parameter
      double summedExpectationError = 0;
      for (MeasurementExpectation<Integer> expectation: counts.getExpectationsForAnnotator(j)){
        summedExpectationError += Math.pow(expectation.getMeasurement().getValue() - expectation.expectedValue(), 2);
      }
      nuSigma2[j][1] = beta + 0.5 * summedExpectationError; 
    }
  }
  
  
  private void fitLogNuY(double[][] logNuY, ClassificationMeasurementModelCounts counts) {
    // pre-calculate
    double[] digammaOfNuThetas = MeanFieldMultiRespModel.digammasOfArray(state.getNuTheta());
    double digammaOfSummedNuThetas = MeanFieldMultiRespModel.digammaOfSummedArray(state.getNuTheta());
    double priorBeta = state.getPriors().getCGamma();
    
    for (int i=0; i<state.getNumDocuments(); i++){
      for (int c=0; c<state.getNumClasses(); c++){
        // part 1 (identical to first part of meanfielditemresp.fitg
        double t1 = digammaOfNuThetas[c] - digammaOfSummedNuThetas;
        
        double t2 = 0;
        for (int j=0; j<state.getNumAnnotators(); j++){
          double postAlpha = state.getNuSigma2()[j][0], postBeta = state.getNuSigma2()[j][1];
          double t3 = 0;
          for (MeasurementExpectation<Integer> expectation: counts.getExpectationsForAnnotatorInstanceAndLabel(j, i, c)){
            // for the purposes of this calculation, 'remove' all expectations that depend on 
            // y_i by setting y_i to 0 (then resetting after)
            expectation.setSummandVisible(i,false);
            t3 += Math.pow(expectation.getMeasurement().getValue() - expectation.expectedValue() - expectation.featureValue(i, c), 2);
            expectation.setSummandVisible(i,true);
          }
          double t4 = postBeta/(postAlpha-1); // E[sigma2]
          
          t2 = priorBeta + (0.5 * t3) / t4;
        }
         
        logNuY[i][c] = t1 - t2;
      }
    }
    Matrices.logNormalizeRowsToSelf(logNuY);
  }


  public double lowerBound(ClassificationMeasurementModelCounts counts) {
    return lowerBound(state, counts);
  }
  
  public static double lowerBound(State state, ClassificationMeasurementModelCounts counts) {
    double elbo = state.getStaticCounts().getLogLowerBoundConstant();
    
    // precalculate values
    double[] digammaOfNuThetas = MeanFieldMultiRespModel.digammasOfArray(state.getNuTheta());
    double digammaOfSummedNuThetas = MeanFieldMultiRespModel.digammaOfSummedArray(state.getNuTheta());
    double[] classCounts = Matrices.sumOverFirst(Matrices.exp(state.getLogNuY()));
    
    // part 1
    for (int c=0; c<state.getNumClasses(); c++){
      double t1 = digammaOfNuThetas[c] - digammaOfSummedNuThetas;
      double t2 = state.getPriors().getBTheta() + classCounts[c] - 1;
      elbo += t1*t2;
    }
    
    // part 2
    double priorAlpha = state.getPriors().getBGamma(), priorBeta = state.getPriors().getCGamma(); // shoe-horned IG prior params
    for (int j=0; j<state.getNumAnnotators(); j++){
      double postAlpha = state.getNuSigma2()[j][0], postBeta = state.getNuSigma2()[j][1]; // IG variational params
      // part 2a
      double t1 = -( (state.getStaticCounts().getPerAnnotatorMeasurements().getCount(j) / 2.0) + priorAlpha) - 1;
      double t2 = Math.log(postBeta) - Dirichlet.digamma(postAlpha);
      elbo += t1*t2;
    
      // part 2b
      double t3 = 0;
      for (MeasurementExpectation<Integer> expectation: counts.getExpectationsForAnnotator(j)){
        t3 += Math.pow(expectation.getMeasurement().getValue() - expectation.expectedValue(), 2);
      }
      double t4 = postBeta/(postAlpha-1); // E[sigma2]
      elbo -= (priorBeta + (0.5 * t3)) / t4;
    }
    
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
      
      // increment with measured annotation values
      for (Measurement measurement: instance.getAnnotations().getMeasurements()){
        if (measurement instanceof ClassificationAnnotationMeasurement){
          ClassificationAnnotationMeasurement annotation = (ClassificationAnnotationMeasurement) measurement;
          state.getLogNuY()[docIndex][annotation.getLabel()] += annotation.getValue();
        }
      }
    }
    Matrices.logNormalizeRowsToSelf(state.getLogNuY());
    
    // now set theta and sigma2 by fit
    ClassificationMeasurementModelCounts counts = ClassificationMeasurementModelCounts.from(state);
    fitNuTheta(state.getNuTheta(), counts);
    fitNuSigma2(state.getNuSigma2(), counts);
    
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
  
  
  
  public static class Builder extends AbstractMeasurementModelBuilder{
    /** {@inheritDoc} */
    @Override
    protected ClassificationMeasurementModel buildModel(PriorSpecification priors, Dataset data, int[] y,
        StaticMeasurementModelCounts staticCounts, Map<String, Integer> instanceIndices, RandomGenerator rnd) {
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
            .build()
            ;
      
      // create model and initialize variational parameters with an empirical fit
      BasicClassificationMeasurementModel model = new BasicClassificationMeasurementModel(state); 
      model.empiricalFit();
      
      return model; 
    }

  }


}
