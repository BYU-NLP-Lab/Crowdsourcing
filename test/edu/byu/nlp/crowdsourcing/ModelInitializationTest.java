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

import static org.fest.assertions.Assertions.assertThat;

import java.io.FileNotFoundException;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.fest.assertions.Assertions;
import org.junit.Test;
import org.mockito.Mockito;

import com.google.common.base.Joiner;
import com.google.common.collect.Lists;

import edu.byu.nlp.crowdsourcing.ModelInitialization.AccuracyInitializer;
import edu.byu.nlp.crowdsourcing.ModelInitialization.AssignmentInitializer;
import edu.byu.nlp.crowdsourcing.ModelInitialization.NoisyInitializer;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.util.JsonDatasetMocker;
import edu.byu.nlp.dataset.Datasets;
import edu.byu.nlp.util.Counter;
import edu.byu.nlp.util.DenseCounter;
import edu.byu.nlp.util.Matrices;

/**
 * @author pfelt
 * todo (pfelt): I think robbie devised a bayesian significance test for tests like these used elsewhere in the codebase. adopt it. 
 */
public class ModelInitializationTest {

  @Test
  public void testBaselineAssignmentInitializer() throws FileNotFoundException {
    
    RandomGenerator rnd = Mockito.mock(RandomGenerator.class); // always return 0
    AssignmentInitializer init = new ModelInitialization.BaselineInitializer(rnd, false);

    Dataset data = JsonDatasetMocker.buildTestDatasetFromJson(jsonInstances(1));
    Map<String,Integer> instanceIndices = Datasets.instanceIndices(data);
    int[] assignments = new int[data.getInfo().getNumDocuments()];
    
    init.setData(data, instanceIndices);
    init.initialize(assignments);

    assertThat(assignments[0]).isEqualTo(0); // majority vote
    assertThat(assignments[1]).isEqualTo(1); // majority vote
    assertThat(assignments[2]).isEqualTo(0); // majority vote
    assertThat(assignments[3]).isEqualTo(0); // naive bayes prediction (no annotations)
      
  }
  
  @Test
  public void testUniformAssignmentInitializer() {
    int numLabels = 5;
    int numSamples = 10000;
    
    RandomGenerator rnd = new MersenneTwister(1);
    AssignmentInitializer init = ModelInitialization.newUniformAssignmentInitializer(numLabels, rnd);

    int[] assignments = new int[numSamples];
    init.initialize(assignments);
    Counter<Integer> cnt = new DenseCounter(numLabels);
    for (int i=0; i<assignments.length; i++){
      cnt.incrementCount(assignments[i], 1);
    }
    
    double uniform = 1.0/numLabels;
    for (Entry<Integer, Integer> entry: cnt.entrySet()){
      double distFromUniform = Math.abs(((double)entry.getValue()/(double)numSamples) - uniform);
      assertThat(distFromUniform).isLessThan(1e-2); 
    }
      
  }

  @Test
  public void testUniformAccuracyInitializer(){
    int numLabels = 5;
    
    RandomGenerator rnd = new MersenneTwister(1);
    AccuracyInitializer init = ModelInitialization.newUniformAccuracyInitializer(rnd);

    double[][] logAccuracies = new double[numLabels][numLabels];
    init.initializeLogAccuracies(logAccuracies);
    double[][] accuracies = Matrices.exp(logAccuracies);
    double uniform = 1.0/numLabels;
    for (int i=0; i<accuracies.length; i++){
      for (int j=0; j<accuracies[i].length; j++){
         double distFromUniform = Math.abs((accuracies[i][j]/(double)numLabels) - uniform);
         assertThat(distFromUniform).isLessThan(.5); 
      }
    }
    
  }

  @Test
  public void testPriorAccuracyInitializer(){
    int numLabels = 5;
    
    RandomGenerator rnd = new MersenneTwister(1);
    double concentration = 1000;

    for (int test=0; test<10; test++){
      // choose a random accuracy parameter to test
      double accuracy = rnd.nextDouble();
      AccuracyInitializer init = ModelInitialization.newPriorAccuracyInitializer(accuracy, concentration, rnd);
  
      double[][] logAccuracies = new double[numLabels][numLabels];
      init.initializeLogAccuracies(logAccuracies);
      double[][] accuracies = Matrices.exp(logAccuracies);
      for (int i=0; i<accuracies.length; i++){
        for (int j=0; j<accuracies[i].length; j++){
          // diagonal entry expected value = accuracy  
          if (i==j){
            double expected = accuracy;
            double distFromExpected = Math.abs(expected - accuracies[i][j]);
            assertThat(distFromExpected).isLessThan(0.1); 
          }
          // off-diagonal entry expected value = ((1-accuracy)/numLabels) 
          else{
            double expected = (1-accuracy)/numLabels;
            double distFromExpected = Math.abs(expected - accuracies[i][j]);
            assertThat(distFromExpected).isLessThan(0.1); 
          }
        }
      }
    
    } // end tests
  }
  
  @Test 
  public void testNoisyInitializer(){
    
    // a dummy rand number generator
    RandomGenerator rnd = new RandomGenerator() {
      private int i=0;
      private int j=0;
      @Override
      public int nextInt(int arg0) {
        return new int[]{0,1,2,3,4,5,6,7,8,9}[i++];
      }
      @Override
      public double nextDouble() {
        return new double[]{.1,.9,.1,.9,.1,.9,.1,.9,.1,.9}[j++];
      }
      @Override
      public void setSeed(long arg0) {}
      @Override
      public void setSeed(int[] arg0) {}
      @Override
      public void setSeed(int arg0) {}
      @Override
      public long nextLong() {
        throw new UnsupportedOperationException();
      }
      @Override
      public int nextInt() {
        throw new UnsupportedOperationException();
      }
      @Override
      public double nextGaussian() {
        throw new UnsupportedOperationException();
      }
      @Override
      public float nextFloat() {
        throw new UnsupportedOperationException();
      }
      @Override
      public void nextBytes(byte[] arg0) {
        throw new UnsupportedOperationException();
      }
      @Override
      public boolean nextBoolean() {
        throw new UnsupportedOperationException();
      }
    };
    
    // a do-nothing delegate
    AssignmentInitializer delegate = new AssignmentInitializer() {
      @Override
      public void setData(Dataset data, Map<String, Integer> instanceIndices) { }
      @Override
      public void initialize(int[] assignments) {}
    };

    int[] arr = new int[]{9,9,9,9,9,9,9,9,9,9};
    NoisyInitializer noisyInit = new ModelInitialization.NoisyInitializer(delegate, 0.8, 4, rnd);
    noisyInit.initialize(arr);
    Assertions.assertThat(arr).isEqualTo(new int[]{0,9,1,9,2,9,3,9,4,9});
  }

  public static String jsonInstances(long seed){ 
    List<String> jsonInstances = Lists.newArrayList(
        "{\"batch\": 0, \"data\":\"ABCD\", \"endTime\":5, \"label\":\"0\",                           \"source\":1,     \"startTime\":0 }", // label
        "{\"batch\": 1, \"data\":\"ABCD\", \"endTime\":6, \"annotation\":\"0\", \"annotator\":\"A\", \"source\":1,     \"startTime\":0 }", // annotation to the same doc
        "{\"batch\": 2, \"data\":\"ABCD\", \"endTime\":7, \"annotation\":\"1\", \"annotator\":\"B\", \"source\":1,     \"startTime\":0 }", // annotation to the same doc
        "{\"batch\": 2, \"data\":\"ABCD\", \"endTime\":8, \"annotation\":\"0\", \"annotator\":\"B\", \"source\":1,     \"startTime\":0 }", // annotation to the same doc
        "{\"batch\": 0, \"data\":\"ABCD\", \"endTime\":1, \"label\":\"1\",                           \"source\":2,     \"startTime\":0 }", // label 
        "{\"batch\": 1, \"data\":\"ABCD\", \"endTime\":2, \"annotation\":\"0\", \"annotator\":\"A\", \"source\":2,     \"startTime\":0 }", // annotation to the same doc
        "{\"batch\": 2, \"data\":\"ABCD\", \"endTime\":3, \"annotation\":\"1\", \"annotator\":\"B\", \"source\":2,     \"startTime\":0 }", // annotation to the same doc
        "{\"batch\": 2, \"data\":\"ABCD\", \"endTime\":4, \"annotation\":\"1\", \"annotator\":\"B\", \"source\":2,     \"startTime\":0 }", // annotation to the same doc
        "{\"batch\": 0, \"data\":\"ABCD\", \"endTime\":9, \"label\":\"0\",                           \"source\":3,     \"startTime\":0 }", // label
        "{\"batch\": 1, \"data\":\"ABCD\", \"endTime\":10, \"annotation\":\"1\", \"annotator\":\"A\", \"source\":3,     \"startTime\":0 }", // annotation to the same doc
        "{\"batch\": 2, \"data\":\"ABCD\", \"endTime\":11, \"annotation\":\"0\", \"annotator\":\"B\", \"source\":3,     \"startTime\":0 }", // annotation to the same doc
        "{\"batch\": 2, \"data\":\"ABCD\", \"endTime\":12, \"annotation\":\"0\", \"annotator\":\"B\", \"source\":3,     \"startTime\":0 }", // annotation to the same doc
        "{\"batch\": 0, \"data\":\"ABCD\", \"endTime\":13, \"label\":\"0\",                           \"source\":4,     \"startTime\":0 }" // label
    );
    Random rand = new Random(seed);
    Collections.shuffle(jsonInstances, rand); // try different orderings
    return "[ \n"+ Joiner.on(", \n").join(jsonInstances) +"]";
    
  }
  
}
