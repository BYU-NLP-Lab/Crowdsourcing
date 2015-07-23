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

import edu.byu.nlp.classify.data.DatasetLabeler;
import edu.byu.nlp.classify.eval.Predictions;
import edu.byu.nlp.crowdsourcing.PriorSpecification;
import edu.byu.nlp.crowdsourcing.measurements.AbstractMeasurementModelBuilder;
import edu.byu.nlp.crowdsourcing.measurements.AbstractMeasurementModelBuilder.StaticMeasurementModelCounts;
import edu.byu.nlp.crowdsourcing.measurements.classification.ClassificationMeasurementModel.State;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.DatasetInstance;

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
    fitNuTheta(newstate.getTheta(), counts);
    fitNuSigma2(newstate.getSigma2(), counts);
    fitLogNuY(newstate.getLogNuY(), counts);

    
    // swap in new state all at once
    this.state = newstate;
    
    // optimize hyperparams
    if (state.getPriors().getInlineHyperparamTuning()){
      fitBTheta();
      fitBSigma2();
    }

    return logJoint();
  }




  /* ************************************** */
  /* ***** Parameter Optimization ********* */
  /* ************************************** */

  private void fitNuTheta(double[] nuTheta, ClassificationMeasurementModelCounts counts) {
    for (int c=0; c<state.getNumClasses(); c++){
      // Dirichlet for each class
      nuTheta[c] = 
          state.getPriors().getBTheta() // symmetric class prior
          + Math.exp(counts.getLogNuY(c)); // count
    }
  }

  
  private void fitNuSigma2(double[][] nuSigma2, ClassificationMeasurementModelCounts counts) {
    for (int j=0; j<state.getNumAnnotators(); j++){
      // each inverse gamma distributed sigma2_j has two variational parameters: shape (nuSigma2[j][0]) and scale (nuSigma2[j][1]).
      // These variational posteriors parameters are influence by their corresponding prior hyperparameters 
      // alpha (shoe-horned into priors.bgamma) and beta (priors.cgamma)
      nuSigma2[j][0] = 
          (state.getStaticCounts().getPerAnnotatorMeasurements()[j] / 2.0)
          - state.getPriors().getBGamma();
      nuSigma2[j][1] = 
          state.getPriors().getCGamma()
          ; // beta
    }
  }
  
  
  private void fitLogNuY(double[][] nuY, ClassificationMeasurementModelCounts counts) {
    for (int i=0; i<state.getNumDocuments(); i++){
    }
  }


  public double logJoint() {
    // TODO Auto-generated method stub
    return 0;
  }

  
  public void empiricalFit(){
//    // initialize logNuY with (smoothed?) empirical distribution
//    for (int i=0; i<a.length; i++){
//      // logNuY = empirical fit
//      for (int j=0; j<a[i].length; j++){
//        for (int k=0; k<a[i][j].length; k++){
//          state.getLogNuY()[i][k] += a[i][j][k] + INITIALIZATION_SMOOTHING; 
//        }
//      }
//      DoubleArrays.normalizeAndLogToSelf(vars.logg[i]);
//    }
//    
//    // init params based on g and h
//    fitPi(vars.pi);
//    fitNu(vars.nu);
//    
//    // make sure both sets of parameter values match
//    newvars.clonetoSelf(vars);
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
    protected ClassificationMeasurementModel initializeModel(PriorSpecification priors, Dataset data,
        int[] y, StaticMeasurementModelCounts staticCounts, Map<String, Integer> instanceIndices, RandomGenerator rnd) {
      ClassificationMeasurementModel.State state = 
          new ClassificationMeasurementModel.State()
            .setData(data)
            .setPriors(priors)
            .setInstanceIndices(instanceIndices)
            .setStaticCounts(staticCounts)
            ;
      
      return new BasicClassificationMeasurementModel(state);
    }
  }




}
