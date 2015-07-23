package edu.byu.nlp.crowdsourcing.measurements;

import java.util.Map;

import org.apache.commons.math3.random.RandomGenerator;

import edu.byu.nlp.crowdsourcing.ModelInitialization.AssignmentInitializer;
import edu.byu.nlp.crowdsourcing.measurements.classification.ClassificationMeasurementModel;
import edu.byu.nlp.crowdsourcing.PriorSpecification;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.dataset.Datasets;

public abstract class AbstractMeasurementModelBuilder {

  private PriorSpecification priors;
  private RandomGenerator rnd;
  private AssignmentInitializer yInitializer;
  private Dataset data;

  public AbstractMeasurementModelBuilder(){  }
  
  public AbstractMeasurementModelBuilder setPriors(PriorSpecification priors){
    this.priors=priors;
    return this;
  }

  public AbstractMeasurementModelBuilder setRnd(RandomGenerator rnd){
    this.rnd=rnd;
    return this;
  }

  public AbstractMeasurementModelBuilder setData(Dataset data) {
    this.data = data;
    return this;
  }

  public AbstractMeasurementModelBuilder setYInitializer(AssignmentInitializer yInitializer){
    this.yInitializer=yInitializer;
    return this;
  }
  
  public ClassificationMeasurementModel build(){
    // TODO: pre-compute data stats used by measurement models
    StaticMeasurementModelCounts staticCounts = new StaticMeasurementModelCounts(data);

    // precalculate which instance corresponds to which index. This will 
    // be used by external code when mapping y to specific predictions
    Map<String,Integer> instanceIndices = Datasets.instanceIndices(data);
    
    int[] y = null;
    return initializeModel(priors, data, y, staticCounts, instanceIndices, rnd);
  }
  
  protected abstract ClassificationMeasurementModel initializeModel(PriorSpecification priors, Dataset data, int[] y, StaticMeasurementModelCounts staticCounts, Map<String,Integer> instanceIndices, RandomGenerator rnd);
  
  
  public static class StaticMeasurementModelCounts{
    private final int[] perAnnotatorMeasurements;
    public StaticMeasurementModelCounts (Dataset data){
      this.perAnnotatorMeasurements = new int[data.getInfo().getNumAnnotators()];
    }
    public int[] getPerAnnotatorMeasurements(){
      return perAnnotatorMeasurements;
    }
  }
  
}
