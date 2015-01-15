/**
 * Copyright 2013 Brigham Young University
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
package edu.byu.nlp.crowdsourcing;

import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.util.DatasetMocker;
import edu.byu.nlp.dataset.BasicSparseFeatureVector;

/**
 * @author rah67
 *
 */
public class TestUtil {
  
  /**
   *  Setup a scenario with two annotators, three possible classes, five possible features, and
   *  four data instances.
   */
  public static Dataset stubDataset() {
    DatasetMocker mocker = new DatasetMocker();
    
    int i=0;

    mocker.addInstance(""+(i++), new BasicSparseFeatureVector(new int[]{0, 4}, new double[]{1, 2}), 0, 
        new int[][]{{1,0,0},{0,0,0},{0,0,0}}); // annotator 0 annotated 0
//    unlabeledData.add(new BasicInstance<Integer, SparseFeatureVector>(
//        0, true, null, ""+(i++), ,
//        newAnnotations(new long[]{0}, new int[]{0})));

    mocker.addInstance(""+(i++), new BasicSparseFeatureVector(new int[]{1, 2}, new double[]{3, 1}), 1, 
        new int[][]{{0,0,0},{0,1,0},{0,0,0}}); // annotator 1 annotated 1
//    unlabeledData.add(new BasicInstance<Integer, SparseFeatureVector>(
//        1, true, null, ""+(i++), new SparseFeatureVector(new int[]{1, 2}, new double[]{3, 1}),
//        newAnnotations(new long[]{1}, new int[]{1})));

    mocker.addInstance(""+(i++), new BasicSparseFeatureVector(new int[]{1, 3}, new double[]{1, 4}), 2, 
        new int[][]{{0,0,1},{0,1,0},{0,0,0}}); // annotator 0 said 2 and annotator 1 said 1
//    unlabeledData.add(new BasicInstance<Integer, SparseFeatureVector>(
//        2, true, null, ""+(i++), new SparseFeatureVector(new int[]{1, 3}, new double[]{1, 4}),
//        newAnnotations(new long[]{0, 1}, new int[]{2, 1})));

    mocker.addInstance(""+(i++), new BasicSparseFeatureVector(new int[]{1, 4}, new double[]{1, 1}), 0, 
        new int[][]{{2,0,0},{1,1,0},{0,0,0}}); // 0->0 twice, 1->0, 1->1
//    unlabeledData.add(new BasicInstance<Integer, SparseFeatureVector>(
//        0, true, null, ""+(i++), new SparseFeatureVector(new int[]{1, 4}, new double[]{1, 1}),
//        newAnnotations(new long[]{0, 1, 0, 1}, new int[]{0, 0, 0, 1})));

    mocker.addInstance(""+(i++), new BasicSparseFeatureVector(new int[]{2, 3}, new double[]{4, 1}), 0, 
        new int[][]{{0,0,0},{0,0,0},{0,0,0}}); // no annotations
//    unlabeledData.add(new BasicInstance<Integer, SparseFeatureVector>(
//        0, true, null, ""+(i++), new SparseFeatureVector(new int[]{2, 3}, new double[]{4, 1}),
//        newAnnotations(new long[]{}, new int[]{})));
    
//    Dataset data = new Dataset(null, unlabeledData, wordIndex, labelIndex);
    return mocker.build();
  }

//  public static Multimap<Long, TimedAnnotation<Integer>> newAnnotations(
//      long[] annotators, int[] annotations) {
//    Map<Long, Collection<TimedAnnotation<Integer>>> map = Maps.newHashMap();
//    Supplier<List<TimedAnnotation<Integer>>> supplier =
//        new Supplier<List<TimedAnnotation<Integer>>>() {
//          @Override
//          public List<TimedAnnotation<Integer>> get() {
//            return Lists.newArrayList();
//          }
//    };
//    Multimap<Long, TimedAnnotation<Integer>> anns = Multimaps.newListMultimap(map, supplier);
//    
//    if (annotators.length != annotations.length) {
//      throw new IllegalArgumentException();
//    }
//    
//    for (int i = 0; i < annotators.length; i++) {
//      anns.put(annotators[i], BasicTimedAnnotation.of(annotations[i]));
//    }
//    
//    return anns;
//  }
  
}
