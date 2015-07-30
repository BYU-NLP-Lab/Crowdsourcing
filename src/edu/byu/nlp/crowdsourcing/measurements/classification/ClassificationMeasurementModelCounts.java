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
import edu.byu.nlp.data.types.DatasetInstance;
import edu.byu.nlp.data.types.Measurement;
import edu.byu.nlp.util.Pair;
import edu.byu.nlp.util.Triple;

/**
 * @author plf1
 *
 */
public class ClassificationMeasurementModelCounts {

//  private Map<Integer, Collection<MeasurementExpectation<Integer>>> measurementsForDocIndex;
  private Map<Integer, Collection<MeasurementExpectation<Integer>>> measurementsForAnnotator;
  private Map<Pair<Integer,Integer>, Collection<MeasurementExpectation<Integer>>> measurementsForAnnotatorAndDocIndex;
  private Map<Triple<Integer,Integer,Integer>, Collection<MeasurementExpectation<Integer>>> measurementsForAnnotatorDocIndexAndClass;

  private ClassificationMeasurementModelCounts() {
  }

  public static ClassificationMeasurementModelCounts from(State state) {
    ClassificationMeasurementModelCounts updater = new ClassificationMeasurementModelCounts();
    updater.initialize(state.getData(), state.getInstanceIndices(), state.getLogNuY());
    return updater;
  }

//  public void setLogNuY_i(int docIndex, double[] logNuY_i) {
//    for (MeasurementExpectation<Integer> expectation : measurementsForDocIndex.get(docIndex)) {
//      expectation.setLogNuY_i(docIndex, logNuY_i);
//    }
//  }

  public Collection<MeasurementExpectation<Integer>> getExpectationsForAnnotator(int annotator){
    return nonNullCollection(measurementsForAnnotator.get(annotator));
  }

//  public Collection<MeasurementExpectation<Integer>> getExpectationsForInstance(int docIndex){
//    return nonNullCollection(measurementsForDocIndex.get(docIndex));
//  }

  public Collection<MeasurementExpectation<Integer>> getExpectationsForAnnotatorAndInstance(int annotator, int docIndex){
    return nonNullCollection(measurementsForAnnotatorAndDocIndex.get(Pair.of(annotator, docIndex)));
  }

  public Collection<MeasurementExpectation<Integer>> getExpectationsForAnnotatorInstanceAndLabel(int annotator, int docIndex, int label){
    return nonNullCollection(measurementsForAnnotatorDocIndexAndClass.get(Triple.of(annotator, docIndex, label)));
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
//      Multimap<Integer, MeasurementExpectation<Integer>> perDocIndex = ArrayListMultimap.create();
      Multimap<Integer, MeasurementExpectation<Integer>> perAnnotator = ArrayListMultimap.create();
      Multimap<Pair<Integer,Integer>, MeasurementExpectation<Integer>> perAnnotatorAndDocIndex = ArrayListMultimap.create(); 
      Multimap<Triple<Integer,Integer,Integer>, MeasurementExpectation<Integer>> perAnnotatorDocIndexAndClass = ArrayListMultimap.create(); 

      // initialize each measurement expectation with the data (and index it for easy lookup)
      for (DatasetInstance item : dataset) {
        int docIndex = instanceIndices.get(item.getInfo().getRawSource());
        for (Measurement measurement : item.getAnnotations().getMeasurements()) {
          ClassificationMeasurement cmeas = (ClassificationMeasurement)measurement;
          MeasurementExpectation<Integer> expectation = ClassificationMeasurementExpectations.fromMeasurement(measurement, dataset, instanceIndices, logNuY);
//          perDocIndex.put(docIndex, expectation);
          perAnnotator.put(measurement.getAnnotator(), expectation);
          perAnnotatorAndDocIndex.put(Pair.of(measurement.getAnnotator(), docIndex), expectation);
          perAnnotatorDocIndexAndClass.put(Triple.of(measurement.getAnnotator(), docIndex, cmeas.getLabel()), expectation);
        }
      }

//      measurementsForDocIndex = perDocIndex.asMap();
      measurementsForAnnotator = perAnnotator.asMap();
      measurementsForAnnotatorAndDocIndex = perAnnotatorAndDocIndex.asMap();
      measurementsForAnnotatorDocIndexAndClass = perAnnotatorDocIndexAndClass.asMap();
    }

  }

}
