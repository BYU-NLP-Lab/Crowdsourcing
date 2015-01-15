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

import java.util.Map;

import org.apache.commons.math3.random.RandomGenerator;

import com.google.common.collect.Iterables;

import edu.byu.nlp.classify.NaiveBayesLearner;
import edu.byu.nlp.classify.data.DatasetBuilder;
import edu.byu.nlp.classify.data.SingleLabelLabeler;
import edu.byu.nlp.classify.eval.Prediction;
import edu.byu.nlp.classify.eval.Predictions;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.DatasetInstance;
import edu.byu.nlp.dataset.Datasets;
import edu.byu.nlp.stats.DirichletDistribution;
import edu.byu.nlp.util.IntArrayCounter;

/**
 * @author rah67
 *
 */
public class ModelInitialization {
  
  public interface AccuracyInitializer {
    void initializeLogAccuracies(double[][] logAccuracies);
  }
  
  public interface AssignmentInitializer {
    void setData(Dataset data, Map<String,Integer> instanceIndices);
    void initialize(int[] assignments);
  }
  
  public static class UniformAssignmentInitializer implements AssignmentInitializer {
    private final int numLabels;
    private final RandomGenerator rnd;
    
    public UniformAssignmentInitializer(int numLabels, RandomGenerator rnd) {
      this.numLabels = numLabels;
      this.rnd = rnd;
    }
    
    @Override
    public void initialize(int[] assignments) {
      for (int i = 0; i < assignments.length; i++) {
        assignments[i] = rnd.nextInt(numLabels);
      }
    }

    @Override
    public void setData(Dataset data, Map<String,Integer> instanceIndices) {}
  }
  
  public static AssignmentInitializer newUniformAssignmentInitializer(int numLabels,
                                                                      RandomGenerator rnd) {
    return new UniformAssignmentInitializer(numLabels, rnd);
  }
  
  private static class UniformAccuracyInitializer implements AccuracyInitializer {
    private final RandomGenerator rnd;

    public UniformAccuracyInitializer(RandomGenerator rnd) {
      this.rnd = rnd;
    }
    
    /** {@inheritDoc} */
    @Override
    public void initializeLogAccuracies(double[][] logAccuracies) {
      for (int k = 0; k < logAccuracies.length; k++) {
        logAccuracies[k] = DirichletDistribution.logSample(1.0, logAccuracies.length, rnd);
      }
    }
  }
  
  public static AccuracyInitializer newUniformAccuracyInitializer(RandomGenerator rnd) {
    return new UniformAccuracyInitializer(rnd);
  }
  
  
  private static class PriorAccuracyInitializer implements AccuracyInitializer {
    private final double accuracy;
    private final double concentration;
    private final RandomGenerator rnd;

    public PriorAccuracyInitializer(double accuracy, double concentration, RandomGenerator rnd) {
      this.accuracy = accuracy;
      this.concentration = concentration;
      this.rnd = rnd;
    }

    /** {@inheritDoc} */
    @Override
    public void initializeLogAccuracies(double[][] logAccuracies) {
      CrowdsourcingUtils.initializeConfusionMatrixWithPrior(logAccuracies, accuracy, concentration);
      DirichletDistribution.logSampleToSelf(logAccuracies, rnd);
    }
  }
  
  /** Sample accuracies using a dirichlet with the given accuracy and concentration parameters. Off
   * diagonal entries are assumed to be uniform. **/
  public static AccuracyInitializer newPriorAccuracyInitializer(
      double accuracy, double concentration, RandomGenerator rnd) {
    return new PriorAccuracyInitializer(accuracy, concentration, rnd);
  }
  
  /**
   * Initialize y based on the baseline approach (majority vote where 
   * possible, naive bayes otherwise) instanceIndi
   */
  public static class BaselineInitializer implements AssignmentInitializer {
    private RandomGenerator rnd;
    private Dataset data;
    private boolean labelAllWithClassifier;
    private Map<String,Integer> instanceIndices;
    public BaselineInitializer(RandomGenerator rnd) {
      this(rnd, false);
    }
    public BaselineInitializer(RandomGenerator rnd, boolean labelAllWithClassifier) {
      this.rnd = rnd;
      this.labelAllWithClassifier=labelAllWithClassifier;
    }
    @Override
    public void setData(Dataset data, Map<String,Integer> instanceIndices) {
      this.data=data;
      this.instanceIndices=instanceIndices;
    }
    /** {@inheritDoc} */
    @Override
    public void initialize(int[] assignments) {
      MajorityVote chooser = new MajorityVote(rnd);
      DatasetBuilder datasetBuilder = new DatasetBuilder(chooser);
      // this labeler reports labeled data without alteration, and trains a
      // naive bayes classifier
      // on it to label the unlabeled and heldout portions
      SingleLabelLabeler labeler = new SingleLabelLabeler(
          new NaiveBayesLearner(), datasetBuilder, 0, null, labelAllWithClassifier);
      Dataset emptyData = Datasets.emptyDataset(data.getInfo());
      Predictions predictions = labeler.label(data, emptyData);
      for (Prediction pred : Iterables.concat(
          predictions.labeledPredictions(), predictions.unlabeledPredictions())) {
        if (instanceIndices.containsKey(pred.getInstance().getInfo().getSource())){
          int modelIndex = instanceIndices.get(pred.getInstance().getInfo().getSource());
          assignments[modelIndex] = pred.getPredictedLabel();
        }
        else{
          // this data instance is not known by the model (doesn't correspond to any parameter) 
          // ignore
        }
      }
    }
  } // end class BaselineInitializer


  /**
   * Takes the max vote for each element from parallel int arrays
   */
  public static class MaxMarginalInitializer implements AssignmentInitializer {
    private int[][] var;
    private int numLabels;

    public MaxMarginalInitializer(int[] var, int numLabels) {
      this(new int[][] { var }, numLabels);
    }

    public MaxMarginalInitializer(int[][] var, int numLabels) {
      this.var = var;
      this.numLabels = numLabels;
    }

    /**
     * We have multiple prior samples. Assign the max marginal assignment to
     * each variable
     */
    @Override
    public void initialize(int[] assignments) {
      // calculate marginals
      IntArrayCounter counter = new IntArrayCounter(assignments.length, numLabels);
      for (int sample = 0; sample < var.length; sample++) {
        for (int v = 0; v < assignments.length; v++) {
          counter.increment(v, var[sample][v]);
        }
      }
      // assign maxes
      for (int v = 0; v < assignments.length; v++) {
        assignments[v] = counter.argmax(v);
      }
    }
    @Override
    public void setData(Dataset data, Map<String,Integer> instanceIndices) {}
  } // end class MaxMarginalInitializer

  public static class LabeledDataInitializer implements AssignmentInitializer {
    private Dataset data;
    private Map<String, Integer> instanceIndices;
    /** {@inheritDoc} */
    @Override
    public void setData(Dataset data, Map<String, Integer> instanceIndices) {
      this.data=data;
      this.instanceIndices=instanceIndices;
    }
    /** {@inheritDoc} */
    @Override
    public void initialize(int[] assignments) {
      for (DatasetInstance instance: Datasets.divideInstancesWithObservedLabels(data).getFirst()){
        int index = instanceIndices.get(instance.getInfo().getSource());
        assignments[index] = instance.getObservedLabel();
      }
    }
  }
  
  public static class NoisyInitializer implements AssignmentInitializer{
    private AssignmentInitializer delegate;
    private double noiseLevel;
    private RandomGenerator rnd;
    private int numClasses;
    public NoisyInitializer(AssignmentInitializer delegate, double noiseLevel, int numClasses, RandomGenerator rnd){
      this.delegate=delegate;
      this.noiseLevel=noiseLevel;
      this.numClasses=numClasses;
      this.rnd=rnd;
    }
    @Override
    public void setData(Dataset data, Map<String, Integer> instanceIndices) {
      delegate.setData(data, instanceIndices);
    }
    @Override
    public void initialize(int[] assignments) {
      delegate.initialize(assignments);
      for (int i=0; i<assignments.length; i++){
        if (rnd.nextDouble()<noiseLevel){
          assignments[i] = rnd.nextInt(numClasses);
        }
      }
    }
  }
  
}
