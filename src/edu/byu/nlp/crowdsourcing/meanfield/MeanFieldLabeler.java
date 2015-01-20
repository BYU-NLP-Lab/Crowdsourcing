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
package edu.byu.nlp.crowdsourcing.meanfield;

import java.util.List;

import com.google.common.collect.Lists;

import edu.byu.nlp.classify.data.DatasetLabeler;
import edu.byu.nlp.classify.eval.BasicPrediction;
import edu.byu.nlp.classify.eval.Prediction;
import edu.byu.nlp.classify.eval.Predictions;
import edu.byu.nlp.classify.util.ModelTraining;
import edu.byu.nlp.crowdsourcing.CrowdsourcingUtils;
import edu.byu.nlp.crowdsourcing.MultiAnnModel;
import edu.byu.nlp.crowdsourcing.MultiAnnModelBuilders.MultiAnnModelBuilder;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.DatasetInstance;
import edu.byu.nlp.util.DoubleArrays;
import edu.byu.nlp.util.Enumeration;
import edu.byu.nlp.util.Iterables2;
import edu.byu.nlp.util.Matrices;

/**
 * @author pfelt
 * 
 */
public class MeanFieldLabeler implements DatasetLabeler{

  private MultiAnnModelBuilder modelBuilder;
  private MultiAnnModel model;
  private String trainingOperations;

  public MeanFieldLabeler(MultiAnnModelBuilder modelBuilder, String trainingOperations) {
    this.modelBuilder=modelBuilder;
    this.trainingOperations=trainingOperations;
  }

  /** {@inheritDoc} */
  @Override
  public Predictions label(Dataset trainingInstances, Dataset heldoutInstances) {
    model = modelBuilder.build();

    // train
    ModelTraining.doOperations(trainingOperations, model);
    
    MeanFieldMultiAnnState state = (MeanFieldMultiAnnState) model.getCurrentState();
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
        heldoutInstances, data.getInfo().getNumFeatures(), data.getInfo().getNumClasses(), (MeanFieldMultiAnnModel) model);
    
    // confusions
    double[][][] annotatorConfusionMatrices = Matrices.clone(state.getNu());
    for (int j=0; j<annotatorConfusionMatrices.length; j++){
      Matrices.normalizeRowsToSelf(annotatorConfusionMatrices[j]);
    }
    double[][] machineConfusionMatrix = Matrices.clone(state.getTau());
    Matrices.normalizeRowsToSelf(machineConfusionMatrix);
    // accuracies
    double[] annotatorAccuracies = getMeanAnnotatorAccuracies(state);
    double machineAccuracy = getMeanMachineAccuracy(state);
    
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
      int numLabels, MeanFieldMultiAnnModel model) {
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
      int numLabels, int size, MeanFieldMultiAnnState state) {

    List<Prediction> predictions = Lists.newArrayList();
    for (Enumeration<DatasetInstance> entry : Iterables2.enumerate(instances)) {
      DatasetInstance instance = entry.getElement();
      Integer index = entry.getIndex();
      
      double[] probs = state.getG()[index];
      predictions.add(new BasicPrediction(DoubleArrays.argMaxList(-1, probs), instance));
    }
    return predictions;
    
  }

  public static double getMeanMachineAccuracy(MeanFieldMultiAnnState state) {
    // mean bias
    double[] meanTheta = DoubleArrays.of(1, state.getNumLabels());
    if (state.getPi()!=null){
      meanTheta = state.getPi().clone();
      DoubleArrays.normalizeToSelf(meanTheta);
    }
    // mean confusion
    double[][] meanMu = Matrices.clone(state.getTau());
    Matrices.normalizeRowsToSelf(meanMu);

    // combine bias&confusion
    return CrowdsourcingUtils.getAccuracy(meanTheta, meanMu);
  }

  // can't we do better since we've got the full distribution now?
  public static double[] getMeanAnnotatorAccuracies(MeanFieldMultiAnnState state) {
    double[] accuracies = new double[state.getNumAnnotators()];
    
    // mean bias
    double[] meanTheta = DoubleArrays.of(1.0/state.getNumLabels(), state.getNumLabels());
    if (state.getPi()!=null){
      meanTheta = state.getPi().clone();
      DoubleArrays.normalizeToSelf(meanTheta);
    }
    
    // mean confusions
    double[][][] meanGamma = Matrices.clone(state.getNu());
    for (int j=0; j<meanGamma.length; j++){
      Matrices.normalizeRowsToSelf(meanGamma[j]);
    }
    
    // combine mean bias & confusion into mean accuracy 
    for (int l = 0; l < accuracies.length; l++) {
      double alpha[][] = meanGamma[l];
      accuracies[l] = CrowdsourcingUtils.getAccuracy(meanTheta, alpha);
    }
    return accuracies;
  }


}
