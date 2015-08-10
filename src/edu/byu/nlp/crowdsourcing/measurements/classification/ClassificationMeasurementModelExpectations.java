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

import java.util.Collection;
import java.util.Map;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Lists;
import com.google.common.collect.Multimap;

import edu.byu.nlp.crowdsourcing.measurements.MeasurementExpectation;
import edu.byu.nlp.crowdsourcing.measurements.classification.ClassificationMeasurementModel.State;
import edu.byu.nlp.data.measurements.ClassificationMeasurements.ClassificationMeasurement;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.Measurement;
import edu.byu.nlp.util.Pair;
import edu.byu.nlp.util.Triple;

/**
 * @author plf1
 *
 */
public class ClassificationMeasurementModelExpectations {

  private Map<Integer, Collection<MeasurementExpectation<Integer>>> measurementsForDocIndex;
  private Map<Integer, Collection<MeasurementExpectation<Integer>>> measurementsForAnnotator;
  private Map<Pair<Integer,Integer>, Collection<MeasurementExpectation<Integer>>> measurementsForAnnotatorAndDocIndex;
  private Map<Triple<Integer, Integer,Integer>, Collection<MeasurementExpectation<Integer>>> measurementsForAnnotatorDocIndexAndLabel;

  private ClassificationMeasurementModelExpectations() {}

  public static ClassificationMeasurementModelExpectations from(State state) {
    ClassificationMeasurementModelExpectations expectations = new ClassificationMeasurementModelExpectations();
    expectations.initialize(state.getData(), state.getInstanceIndices(), state.getLogNuY());
    return expectations;
  }

  public Collection<MeasurementExpectation<Integer>> getExpectationsForAnnotator(int annotator){
    return nonNullCollection(measurementsForAnnotator.get(annotator));
  }

  public Collection<MeasurementExpectation<Integer>> getExpectationsForAnnotatorAndInstance(int annotator, int docIndex){
    return nonNullCollection(measurementsForAnnotatorAndDocIndex.get(Pair.of(annotator, docIndex)));
  }

  public Collection<MeasurementExpectation<Integer>> getExpectationsForAnnotatorInstanceAndLabel(int annotator, int docIndex, int label){
    return nonNullCollection(measurementsForAnnotatorDocIndexAndLabel.get(Triple.of(annotator, docIndex, label)));
  }

  public Collection<MeasurementExpectation<Integer>> getExpectationsForDocumentIndex(int docIndex) {
    return nonNullCollection(measurementsForDocIndex.get(docIndex));
  }
  
  private static <T> Collection<T> nonNullCollection(Collection<T> baseCollection){
    if (baseCollection==null){
      return Lists.newArrayList();
    }
    return baseCollection;
  }

  public void initialize(Dataset dataset, Map<String, Integer> instanceIndices, double[][] logNuY) {

    if (measurementsForAnnotator == null) {

      // multimaps
      Multimap<Integer, MeasurementExpectation<Integer>> perDocIndex = ArrayListMultimap.create();
      Multimap<Integer, MeasurementExpectation<Integer>> perAnnotator = ArrayListMultimap.create();
      Multimap<Pair<Integer,Integer>, MeasurementExpectation<Integer>> perAnnotatorAndDocIndex = ArrayListMultimap.create(); 
      Multimap<Triple<Integer, Integer,Integer>, MeasurementExpectation<Integer>> perAnnotatorDocIndexAndLabel = ArrayListMultimap.create();

      // initialize each measurement expectation with the data (and index it for easy lookup)
      for (Measurement measurement : dataset.getMeasurements()) {
        int label = ((ClassificationMeasurement)measurement).getLabel();
        MeasurementExpectation<Integer> expectation = ClassificationMeasurementExpectations.fromMeasurement(measurement, dataset, instanceIndices, logNuY);
        // ignore measurements that don't apply to any documents
        if (expectation.getDependentIndices().size()==0){
          continue;
        }
        perAnnotator.put(measurement.getAnnotator(), expectation);
        for (Integer docIndex: expectation.getDependentIndices()){
          perDocIndex.put(docIndex, expectation);
          perAnnotatorAndDocIndex.put(Pair.of(measurement.getAnnotator(), docIndex), expectation);
          perAnnotatorDocIndexAndLabel.put(Triple.of(measurement.getAnnotator(), docIndex, label), expectation);
        }
        
      }

      measurementsForAnnotator = perAnnotator.asMap();
      measurementsForAnnotatorAndDocIndex = perAnnotatorAndDocIndex.asMap();
      measurementsForAnnotatorDocIndexAndLabel = perAnnotatorDocIndexAndLabel.asMap();
      measurementsForDocIndex = perDocIndex.asMap();
    }

  }

  /**
   * Signal that logNuY_i has changed, and update all expectations
   * that depend on this doc
   */
  public void updateLogNuY_i(int docIndex, double[] logNuY_i) {
    for (MeasurementExpectation<Integer> expectation: getExpectationsForDocumentIndex(docIndex)){
      expectation.setLogNuY_i(docIndex, logNuY_i);
    }
  }

}
