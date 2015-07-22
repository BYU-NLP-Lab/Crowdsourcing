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

import java.util.List;
import java.util.Map;

import org.apache.commons.math3.random.RandomGenerator;

import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;

import edu.byu.nlp.classify.NaiveBayesLearner;
import edu.byu.nlp.classify.data.DatasetBuilder;
import edu.byu.nlp.classify.data.SingleLabelLabeler;
import edu.byu.nlp.classify.eval.Prediction;
import edu.byu.nlp.classify.eval.Predictions;
import edu.byu.nlp.crowdsourcing.SerializableCrowdsourcingState.SerializableCrowdsourcingDocumentState;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.DatasetInstance;
import edu.byu.nlp.dataset.Datasets;
import edu.byu.nlp.stats.DirichletDistribution;

/**
 * @author rah67
 * @author plf1
 *
 */
public class ModelInitialization {
  
  public interface AccuracyInitializer {
    void initializeLogAccuracies(double[][] logAccuracies);
  }
  
  public interface MatrixAssignmentInitializer{
    void setData(Dataset data, Map<String,Integer> instanceIndices);
    AssignmentInitializer getInitializerFor(int row);
  }

  public static MatrixAssignmentInitializer uniformRowMatrixInitializer(final AssignmentInitializer delegate){
    return new MatrixAssignmentInitializer() {
      @Override
      public void setData(Dataset data, Map<String, Integer> instanceIndices) {
        delegate.setData(data, instanceIndices);
      }
      @Override
      public AssignmentInitializer getInitializerFor(int row) {
        return delegate;
      }
    };
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
        if (instanceIndices.containsKey(pred.getInstance().getInfo().getRawSource())){
          int modelIndex = instanceIndices.get(pred.getInstance().getInfo().getRawSource());
          assignments[modelIndex] = pred.getPredictedLabel();
        }
        else{
          // this data instance is not known by the model (doesn't correspond to any parameter) 
          // ignore
        }
      }
    }
  } // end class BaselineInitializer


  public static class SerializedZInitializer implements MatrixAssignmentInitializer {
	private List<DatasetInstance> instances;
	private SerializableCrowdsourcingState serializedState;
	public SerializedZInitializer(SerializableCrowdsourcingState serializedState){
		this.serializedState=serializedState;
	}
	@Override
	public void setData(Dataset data, Map<String, Integer> instanceIndices) {
		this.instances = Lists.newArrayList(data);
	}
	@Override
	public AssignmentInitializer getInitializerFor(final int docIndex) {
		final String docSrc = instances.get(docIndex).getInfo().getRawSource();
		return new AssignmentInitializer() {
			@Override
			public void setData(Dataset data, Map<String, Integer> instanceIndices) {}
			@Override
			public void initialize(int[] assignments) {
				int[] serializedAssignments = serializedState.getDocument(docSrc).getZ();
				for (int z=0; z<assignments.length; z++){
					assignments[z] = serializedAssignments[z];
				}
			}
		};
	}
	  
  }
  
  public static class SerializedYInitializer implements AssignmentInitializer {
	private Map<String, Integer> instanceIndices;
	private Dataset data;
	private SerializableCrowdsourcingState serializedState;
	public SerializedYInitializer(SerializableCrowdsourcingState serializedState){
		this.serializedState=serializedState;
	}
	@Override
	public void setData(Dataset data, Map<String, Integer> instanceIndices) {
		this.data=data;
		this.instanceIndices=instanceIndices;
	}
	@Override
	public void initialize(int[] assignments) {
		for (DatasetInstance inst: data){
			String src = inst.getInfo().getRawSource();
			SerializableCrowdsourcingDocumentState docState = serializedState.getDocument(src);
			Integer index = instanceIndices.get(src);
			if (docState!=null && index!=null){
  			assignments[index] = docState.getY();
			}
		}
	}
  }

  
  public static class SerializedMInitializer implements AssignmentInitializer {
	private Map<String, Integer> instanceIndices;
	private Dataset data;
	private SerializableCrowdsourcingState serializedState;
	public SerializedMInitializer(SerializableCrowdsourcingState serializedState){
		this.serializedState=serializedState;
	}
	@Override
	public void setData(Dataset data, Map<String, Integer> instanceIndices) {
		this.data=data;
		this.instanceIndices=instanceIndices;
	}
	@Override
	public void initialize(int[] assignments) {
		for (DatasetInstance inst: data){
			String src = inst.getInfo().getRawSource();
			SerializableCrowdsourcingDocumentState docState = serializedState.getDocument(src);
			int index = instanceIndices.get(src);
			assignments[index] = docState.getM(); 
		}
	}
  }
  

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
        int index = instanceIndices.get(instance.getInfo().getRawSource());
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
  

  /**
   * Either create a state initializer, or if the state has no values for Y, return the provided backoff initializer
   */
  public static AssignmentInitializer backoffStateInitializerForY(SerializableCrowdsourcingState state, AssignmentInitializer backoffInitializer){
    AssignmentInitializer yInitializer;
    if (state!=null && state.hasY()){
    	yInitializer = new ModelInitialization.SerializedYInitializer(state);
    }
    else{
    	yInitializer = backoffInitializer;
    }
    return yInitializer;
  }

  /**
   * Either create a state initializer, or if the state has no values for M, return the provided backoff initializer
   */
  public static AssignmentInitializer backoffStateInitializerForM(SerializableCrowdsourcingState state, AssignmentInitializer backoffInitializer){
    AssignmentInitializer mInitializer;
    if (state!=null && state.hasM()){
    	mInitializer = new ModelInitialization.SerializedMInitializer(state);
    }
    else{
    	mInitializer = backoffInitializer;
    }
    return mInitializer;
  }

  /**
   * Either create a state initializer, or if the state has no values for Z, return the provided backoff initializer
   */
  public static MatrixAssignmentInitializer backoffStateInitializerForZ(SerializableCrowdsourcingState state, MatrixAssignmentInitializer backoffInitializer){
	  MatrixAssignmentInitializer zInitializer;
    if (state!=null && state.hasZ()){
    	zInitializer = new ModelInitialization.SerializedZInitializer(state);
    }
    else{
    	zInitializer = backoffInitializer;
    }
    return zInitializer;
  }

}
