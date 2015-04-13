/**
 * Copyright 2014 Brigham Young University
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
package edu.byu.nlp.crowdsourcing.em;

import org.apache.commons.math3.random.RandomGenerator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;

import edu.byu.nlp.classify.data.DatasetLabeler;
import edu.byu.nlp.classify.eval.Predictions;
import edu.byu.nlp.classify.util.ModelTraining;
import edu.byu.nlp.classify.util.ModelTraining.SupportsTrainingOperations;
import edu.byu.nlp.crowdsourcing.ModelInitialization.AssignmentInitializer;
import edu.byu.nlp.crowdsourcing.ModelInitialization.MatrixAssignmentInitializer;
import edu.byu.nlp.crowdsourcing.PriorSpecification;
import edu.byu.nlp.crowdsourcing.em.ConfusedSLDADiscreteModel.State;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.util.Matrices;

/**
 * @author plf1
 *
 * A model where LDA vectors are trained as a preprocessing step, after 
 * which LogResp is run using topic features for data. This can be 
 * implemented as a special case of the csLDA model. 
 */
public class LogRespLDAModel {
  private static final Logger logger = LoggerFactory.getLogger(LogRespLDAModel.class);
  
  
  //////////////////////////////////////////////
  // Model Code
  //////////////////////////////////////////////
  private State state;
  
  public LogRespLDAModel(State state) {
    this.state=state;
  }

  public Dataset getData() {
    return this.state.data;
  }
  
  public int[] getY(){
    return this.state.y.clone();
  }

  public int[][] getZ(){
    return Matrices.clone(this.state.z);
  }
  
  public State getState(){
    return this.state;
  }

  public Predictions predict(Dataset trainingInstances, Dataset heldoutInstances, RandomGenerator rnd){
    return ConfusedSLDADiscreteModel.predict(state, trainingInstances, heldoutInstances, rnd);
  }

  //////////////////////////////////////////////
  // Model Builder
  //////////////////////////////////////////////
  public static class ModelBuilder {
    
    ConfusedSLDADiscreteModel.ModelBuilder delegate;
    
    private String trainingOps;
    private RandomGenerator rnd;

    public ModelBuilder(Dataset dataset){
      this.delegate = new ConfusedSLDADiscreteModel.ModelBuilder(dataset);
    }

    public ModelBuilder setZInitializer(MatrixAssignmentInitializer zInitializer){
      this.delegate.setZInitializer(zInitializer);
      return this;
    }
    
    public ModelBuilder setYInitializer(AssignmentInitializer yInitializer){
      this.delegate.setYInitializer(yInitializer);
      return this;
    }

    public ModelBuilder setNumTopics(int numTopics){
      this.delegate.setNumTopics(numTopics);
      return this;
    }
    
    public ModelBuilder setPriors(PriorSpecification priors){
      this.delegate.setPriors(priors);
      return this;
    }

    public ModelBuilder setTrainingOps(String trainingOps){
      this.trainingOps=trainingOps;
      this.delegate.setTrainingOps(trainingOps);
      return this;
    }

    public ModelBuilder setRandomGenerator(RandomGenerator rnd){
      this.rnd=rnd;
      this.delegate.setRandomGenerator(rnd);
      return this;
    }
    
    protected LogRespLDAModel build() {

      ConfusedSLDADiscreteModel delegateModel = this.delegate.build(false);
      State state = delegateModel.getState();

      LogRespLDAModel model = new LogRespLDAModel(state);
      
      ////////////////////
      // train model 
      ////////////////////
      ModelTrainer trainer = new ModelTrainer(state);
      ModelTraining.doOperations(trainingOps, trainer);

      logger.info("Training finished with log joint="+ConfusedSLDADiscreteModel.unnormalizedLogJoint(state));
      logger.info("Final topics");
      ConfusedSLDADiscreteModel.logTopNWordsPerTopic(state, 10);
      
      return model;
      
    }

    private class ModelTrainer implements SupportsTrainingOperations{
      private State state;
      public ModelTrainer(State state){
        this.state=state;
      }
      /** {@inheritDoc} */
      @Override
      public Double sample(String variableName, int iteration, String[] args) {
        // in this model, we ignore the connection between topics z and 
        // inferred labels y when updating z (vanilla LDA model)
        // We do pay attention to it when updating y (logresp w/ lda features)
        // we do this by turning off includeMetadataSupervision before 
        // updating z and turning it on again after
        state.includeMetadataSupervision = true;
        
        Preconditions.checkNotNull(variableName);
        
        // Joint
        if (variableName.equals("all")){
          logger.debug("maximizing log-linear weights B iteration "+iteration);
          ConfusedSLDADiscreteModel.maximizeB(state); 
          logger.debug("sampling class label vector Y iteration "+iteration);
          ConfusedSLDADiscreteModel.sampleY(state, rnd);
          logger.debug("sampling topic matrix Z iteration "+iteration);
          state.includeMetadataSupervision = false;
          ConfusedSLDADiscreteModel.sampleZ(state, rnd);
          state.includeMetadataSupervision = true;
          // periodically tune hypers and report joint
          if (iteration%ConfusedSLDADiscreteModel.HYPERPARAM_TUNING_PERIOD==0){
            logger.info("sample Y+Z+B iteration "+iteration+" with (unnormalized) log joint "+ConfusedSLDADiscreteModel.unnormalizedLogJoint(state));
            if (state.priors.getInlineHyperparamTuning()){
              ConfusedSLDADiscreteModel.updateBTheta(state);
              ConfusedSLDADiscreteModel.updateBPhi(state);
              ConfusedSLDADiscreteModel.updateBGamma(state);
            }
          }
        }
        // Y
        else if (variableName.toLowerCase().equals("y")){
          ConfusedSLDADiscreteModel.sampleY(state, rnd);
          logger.debug("sample Y+B iteration "+iteration);
          // periodically tune hypers and report joint
          if (iteration%ConfusedSLDADiscreteModel.HYPERPARAM_TUNING_PERIOD==0){
            logger.info("maximizing log-linear weights B iteration "+iteration);
            ConfusedSLDADiscreteModel.maximizeB(state); 
            logger.info("sample Y+B iteration "+iteration+" with (unnormalized) log joint "+ConfusedSLDADiscreteModel.unnormalizedLogJoint(state));
            if (state.priors.getInlineHyperparamTuning()){
              ConfusedSLDADiscreteModel.updateBGamma(state);
            }
          }
        }
        // Z
        else if (variableName.toLowerCase().equals("z")){
          state.includeMetadataSupervision = false;
          ConfusedSLDADiscreteModel.sampleZ(state, rnd);
          state.includeMetadataSupervision = true;
          logger.debug("sample Z+B iteration "+iteration);
          // periodically tune hypers and report joint
          if (iteration%ConfusedSLDADiscreteModel.HYPERPARAM_TUNING_PERIOD==0){
            logger.info("maximizing log-linear weights B iteration "+iteration);
            ConfusedSLDADiscreteModel.maximizeB(state); 
            logger.info("sample Z+B iteration "+iteration+" with (unnormalized) log joint "+ConfusedSLDADiscreteModel.unnormalizedLogJoint(state));
            if (state.priors.getInlineHyperparamTuning()){
              ConfusedSLDADiscreteModel.updateBTheta(state);
              ConfusedSLDADiscreteModel.updateBPhi(state);
            }
          }
        }
        // B
        else if (variableName.toLowerCase().equals("b")){
          throw new UnsupportedOperationException("cannot sample b");
        }
        else{
          throw new IllegalArgumentException("unknown variable name "+variableName);
        }

        // for efficiency, only calculate objective value periodically
        if (iteration%ConfusedSLDADiscreteModel.HYPERPARAM_TUNING_PERIOD==0){
          return ConfusedSLDADiscreteModel.unnormalizedLogJoint(state);
        }
        return null;
      }

      private int cumulativeNumChanges = 0;
      /** {@inheritDoc} */
      @Override
      public Double maximize(String variableName, int iteration, String[] args) {
        // reset the cumulative number of changes
        if (iteration==0){
          cumulativeNumChanges = 0;
        }

        // all
        if (variableName.toLowerCase().equals("all")){ 
          Preconditions.checkNotNull(variableName);
          
          // maximize topics (vanilla LDA)
          logger.debug("maximizing log-linear model parameters b iteration "+iteration);
          ConfusedSLDADiscreteModel.maximizeB(state); // set maxent model weights
          logger.debug("maximizing topic assignments Z iteration "+iteration);
          state.includeMetadataSupervision = false;
          cumulativeNumChanges += ConfusedSLDADiscreteModel.maximizeZ(state); // set topic assignments
          state.includeMetadataSupervision = true;
          logger.debug("maximizing inferred labels Y iteration "+iteration);
          cumulativeNumChanges += ConfusedSLDADiscreteModel.maximizeY(state); // set inferred label values
          // tune hyperparams
          ConfusedSLDADiscreteModel.updateBTheta(state);
          ConfusedSLDADiscreteModel.updateBPhi(state);
          ConfusedSLDADiscreteModel.updateBGamma(state);
        }
        // Y
        else if (variableName.toLowerCase().equals("y")){ 
          // maximize class labels independently (FAST)
          state.includeMetadataSupervision = true;
          logger.debug("maximizing inferred labels Y iteration "+iteration);
          cumulativeNumChanges += ConfusedSLDADiscreteModel.maximizeY(state); // set inferred label values
          // update hyperparam
          ConfusedSLDADiscreteModel.updateBGamma(state);
        }
        // Z
        else if (variableName.toLowerCase().equals("z")){
          // maximize topics independently (FAST)
          state.includeMetadataSupervision = false;
          logger.debug("maximizing topic assignments Z iteration "+iteration);
          cumulativeNumChanges += ConfusedSLDADiscreteModel.maximizeZ(state); // set topic assignments
          // tune hyperparams
          ConfusedSLDADiscreteModel.updateBTheta(state);
          ConfusedSLDADiscreteModel.updateBPhi(state);
        }
        // B
        else if (variableName.toLowerCase().equals("b")){
          // maximize log linear model independently 
          state.includeMetadataSupervision = true;
          logger.debug("maximizing regression vector B iteration "+iteration);
          ConfusedSLDADiscreteModel.maximizeB(state);
        }
        else{
          throw new IllegalArgumentException("unknown variable name "+variableName);
        }
        return (double)cumulativeNumChanges;
      }
      /** {@inheritDoc} */
      @Override
      public DatasetLabeler getIntermediateLabeler() {
        // TODO Auto-generated method stub
        return null;
      }
      
    }

  } // end builder

  
}
