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

import java.util.List;

import com.google.common.collect.Lists;

import edu.byu.nlp.classify.data.DatasetLabeler;
import edu.byu.nlp.classify.eval.BasicPrediction;
import edu.byu.nlp.classify.eval.Prediction;
import edu.byu.nlp.classify.eval.Predictions;
import edu.byu.nlp.classify.util.ModelTraining;
import edu.byu.nlp.classify.util.ModelTraining.IntermediatePredictionLogger;
import edu.byu.nlp.crowdsourcing.measurements.AbstractMeasurementModelBuilder;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.DatasetInstance;
import edu.byu.nlp.util.DoubleArrays;
import edu.byu.nlp.util.Enumeration;
import edu.byu.nlp.util.Iterables2;

/**
 * @author plf1
 *
 */
public class ClassificationMeasurementModelLabeler implements DatasetLabeler{

  private AbstractMeasurementModelBuilder builder;
  private ClassificationMeasurementModel model;
  private IntermediatePredictionLogger intermediatePredictionLogger;
  private String trainingOperations;

  public ClassificationMeasurementModelLabeler(AbstractMeasurementModelBuilder builder, String trainingOperations, IntermediatePredictionLogger intermediatePredictionLogger) {
    this.builder=builder;
    this.trainingOperations=trainingOperations;
    this.intermediatePredictionLogger=intermediatePredictionLogger;
  }
  public ClassificationMeasurementModelLabeler(ClassificationMeasurementModel model) {
    this.model=model;
  }
  
  /** {@inheritDoc} */
  @Override
  public Predictions label(Dataset trainingInstances, Dataset heldoutInstances) {
    // if necessary, build and train a model 
    if (this.model == null){
      model = builder.build();
      ModelTraining.doOperations(trainingOperations, model, intermediatePredictionLogger);
    }
    
    ClassificationMeasurementModel.State state = model.getCurrentState();
    Dataset data = state.getData();
    
    // corpus predictions 
    List<Prediction> labeledPredictions = Lists.newArrayList();
    List<Prediction> unlabeledPredictions = Lists.newArrayList();
    for (Prediction prediction : calculateCorpusPredictions(
        data, data.getInfo().getNumClasses(), data.getInfo().getNumDocuments(), state)) {
      if (prediction.getInstance().hasAnnotations()) {
        labeledPredictions.add(prediction);
      } else {
        unlabeledPredictions.add(prediction);
      }
    }

    // out-of-corpus predictions
    List<Prediction> heldoutPredictions = calculateGeneralizationPredictions(
        heldoutInstances, data.getInfo().getNumFeatures(), data.getInfo().getNumClasses(), model);
    
    // confusions
    double[][][] annotatorConfusionMatrices = null;
    double[][] machineConfusionMatrix = null;
    // accuracies
    double[] annotatorAccuracies = null;
    double machineAccuracy = -1;
    
    double logJoint = model.logJoint();
    return new Predictions(labeledPredictions,
                                                         unlabeledPredictions,
                                                         heldoutPredictions,
                                                         annotatorAccuracies,
                                                         annotatorConfusionMatrices,
                                                         machineAccuracy,
                                                         machineConfusionMatrix,
                                                         logJoint);
  }

  private List<Prediction> calculateGeneralizationPredictions(
      Dataset instances, int numFeatures,
      int numLabels, ClassificationMeasurementModel model) {
    List<Prediction> predictions = Lists.newArrayList();
    
    for (DatasetInstance instance: instances) {
      // out-of-corpus prediction (single-instance inference)
      double[] probs = model.fitOutOfCorpusInstance(instance);
      predictions.add(new BasicPrediction(DoubleArrays.argMaxList(-1, probs), instance));
    }
    
    return predictions;
  }

  private List<Prediction> calculateCorpusPredictions(
      Iterable<DatasetInstance> instances,
      int numLabels, int size, ClassificationMeasurementModel.State state) {

    List<Prediction> predictions = Lists.newArrayList();
    for (Enumeration<DatasetInstance> entry : Iterables2.enumerate(instances)) {
      DatasetInstance instance = entry.getElement();
      Integer index = entry.getIndex();
      
      double[] probs = state.getLogNuY()[index];
      predictions.add(new BasicPrediction(DoubleArrays.argMaxList(-1, probs), instance));
    }
    return predictions;
    
  }

}
