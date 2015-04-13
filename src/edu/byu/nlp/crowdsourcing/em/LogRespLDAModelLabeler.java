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

import edu.byu.nlp.classify.data.DatasetLabeler;
import edu.byu.nlp.classify.eval.Predictions;
import edu.byu.nlp.classify.util.ModelTraining.IntermediatePredictionLogger;
import edu.byu.nlp.crowdsourcing.ModelInitialization.AssignmentInitializer;
import edu.byu.nlp.crowdsourcing.ModelInitialization.MatrixAssignmentInitializer;
import edu.byu.nlp.crowdsourcing.PriorSpecification;
import edu.byu.nlp.data.types.Dataset;

/**
 * @author pfelt
 *
 */
public class LogRespLDAModelLabeler implements DatasetLabeler{

  private RandomGenerator rnd;
  private LogRespLDAModel model;

  public LogRespLDAModelLabeler(Dataset data, int numTopics, String trainingOps, MatrixAssignmentInitializer zInitializer, AssignmentInitializer yInitializer, 
      PriorSpecification priors, IntermediatePredictionLogger predictionLogger, RandomGenerator rnd){
    this.rnd=rnd;
    this.model = new LogRespLDAModel.ModelBuilder(data)
        .setNumTopics(numTopics)
        .setPriors(priors)
        .setRandomGenerator(rnd)
        .setYInitializer(yInitializer)
        .setZInitializer(zInitializer)
        .setTrainingOps(trainingOps)
        .setIntermediatePredictionLogger(predictionLogger)
        .build();
  }
  
  /** {@inheritDoc} */
  @Override
  public Predictions label(Dataset trainingInstances, Dataset heldoutInstances) {
    return model.predict(trainingInstances, heldoutInstances, rnd);
  }

}
