package edu.byu.nlp.crowdsourcing.measurements;

import java.util.Map;

import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.special.Gamma;

import edu.byu.nlp.crowdsourcing.ModelInitialization.AssignmentInitializer;
import edu.byu.nlp.crowdsourcing.PriorSpecification;
import edu.byu.nlp.crowdsourcing.measurements.classification.ClassificationMeasurementModel;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.DatasetInstance;
import edu.byu.nlp.data.types.Measurement;
import edu.byu.nlp.dataset.Datasets;
import edu.byu.nlp.math.GammaFunctions;
import edu.byu.nlp.util.Counter;
import edu.byu.nlp.util.DenseCounter;

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
    // pre-compute data stats used by measurement models
    StaticMeasurementModelCounts staticCounts = new StaticMeasurementModelCounts(data, priors);
    
    // precalculate which instance corresponds to which index. This will 
    // be used by external code when mapping y to specific predictions
    Map<String,Integer> instanceIndices = Datasets.instanceIndices(data);

    // initialize y values (probably via majority vote)
    yInitializer.setData(data, instanceIndices);
    int[] y = new int[data.getInfo().getNumDocuments()];
    yInitializer.initialize(y);
    
    return buildModel(priors, data, y, staticCounts, instanceIndices, rnd);
    
  }
  
  protected abstract ClassificationMeasurementModel buildModel(PriorSpecification priors, Dataset data, int[] y, StaticMeasurementModelCounts staticCounts, Map<String,Integer> instanceIndices, RandomGenerator rnd);
  
  
  public static class StaticMeasurementModelCounts{
    private Counter<Integer> perAnnotatorMeasurements;
    private double logLowerBoundConstant;
    public StaticMeasurementModelCounts (Dataset data, PriorSpecification priors){
      initialize(data, priors);
    }
    /**
     * A constant used by q(y) update
     */
    public Counter<Integer> getPerAnnotatorMeasurements(){
      return perAnnotatorMeasurements;
    }
    /**
     * A constant used in the lower bound calculation
     */
    public double getLogLowerBoundConstant(){
      return logLowerBoundConstant;
    }
    
    private void initialize(Dataset data, PriorSpecification priors){
      perAnnotatorMeasurements = new DenseCounter(data.getInfo().getNumAnnotators());
      for (DatasetInstance instance: data){
        for (Measurement measurement: instance.getAnnotations().getMeasurements()){
          perAnnotatorMeasurements.incrementCount(measurement.getAnnotator(), 1);
        }
      }

      // lower bound constant
      double alpha = priors.getBGamma(), beta = priors.getCGamma(); // shoe-horned inverse gamma prior values
      double lbterm1 = GammaFunctions.logBetaSymmetric(priors.getBTheta(),data.getInfo().getNumClasses());
      double lbterm2 = (alpha * Math.log(beta)) - Gamma.logGamma(alpha);
      double lbterm3 = 0;
      for (int j=0; j<data.getInfo().getNumAnnotators(); j++){
        lbterm3 -= ((perAnnotatorMeasurements.getCount(j)/2.0) * Math.log(2.0*Math.PI));
      }
      logLowerBoundConstant = lbterm1 + lbterm2 + lbterm3;
    }
  }
  
}