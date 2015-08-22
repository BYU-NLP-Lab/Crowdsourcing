package edu.byu.nlp.crowdsourcing.measurements;

import java.util.Map;

import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.special.Gamma;

import edu.byu.nlp.crowdsourcing.ModelInitialization.AssignmentInitializer;
import edu.byu.nlp.crowdsourcing.PriorSpecification;
import edu.byu.nlp.crowdsourcing.measurements.classification.PANClassificationMeasurementModel;
import edu.byu.nlp.crowdsourcing.measurements.classification.ClassificationMeasurementModel;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.Measurement;
import edu.byu.nlp.dataset.Datasets;
import edu.byu.nlp.math.GammaFunctions;
import edu.byu.nlp.util.Counter;
import edu.byu.nlp.util.DenseCounter;

public abstract class MeasurementModelBuilder {

  private PriorSpecification priors;
  private RandomGenerator rnd;
  private AssignmentInitializer yInitializer;
  private Dataset data;
  private boolean measurementsPreScaled;
  private String trustedMeasurementAnnotator;

  public MeasurementModelBuilder(){  }
  
  public MeasurementModelBuilder setPriors(PriorSpecification priors){
    this.priors=priors;
    return this;
  }

  public MeasurementModelBuilder setRnd(RandomGenerator rnd){
    this.rnd=rnd;
    return this;
  }

  public MeasurementModelBuilder setData(Dataset data) {
    this.data = data;
    return this;
  }

  public MeasurementModelBuilder setYInitializer(AssignmentInitializer yInitializer){
    this.yInitializer=yInitializer;
    return this;
  }

  public MeasurementModelBuilder setMeasurementsArePreScaled(boolean measurementsPreScaled){
    this.measurementsPreScaled=measurementsPreScaled;
    return this;
  }

  public MeasurementModelBuilder setTrustedAnnotator(String trustedMeasurementAnnotator) {
    this.trustedMeasurementAnnotator=trustedMeasurementAnnotator;
    return this;
  }
  
  public ClassificationMeasurementModel build(){
    // pre-compute data stats used by measurement models
    int trustedAnnotator = data.getInfo().getAnnotatorIdIndexer().indexOf(trustedMeasurementAnnotator);
    StaticMeasurementModelCounts staticCounts = new StaticMeasurementModelCounts(data, priors, trustedAnnotator);
    
    // precalculate which instance corresponds to which index. This will 
    // be used by external code when mapping y to specific predictions
    Map<String,Integer> instanceIndices = Datasets.instanceIndices(data);

    // initialize y values (probably via majority vote)
    yInitializer.setData(data, instanceIndices);
    int[] y = new int[data.getInfo().getNumDocuments()];
    yInitializer.initialize(y);
    
    return buildModel(priors, data, y, staticCounts, instanceIndices, measurementsPreScaled, trustedAnnotator, rnd);
    
  }
  
  protected abstract ClassificationMeasurementModel buildModel(
      PriorSpecification priors, Dataset data, int[] y, StaticMeasurementModelCounts staticCounts, 
      Map<String,Integer> instanceIndices, boolean measurementsPreScaled, int trustedAnnotator, RandomGenerator rnd);
  
  
  public static class StaticMeasurementModelCounts{
    private Counter<Integer> annotatorMeasurementCounter;
    private double logLowerBoundConstant, logQThetaNormalizer, logQTauWNormalizer;
    public StaticMeasurementModelCounts (Dataset data, PriorSpecification priors, int trustedAnnotator){
      initialize(data, priors, trustedAnnotator);
    }
    /**
     * A constant used by q(y) update
     */
    public Counter<Integer> getPerAnnotatorMeasurementCounts(){
      return annotatorMeasurementCounter;
    }
    /**
     * A constant used in the lower bound calculation
     */
    public double getLogLowerBoundConstant(){
      return logLowerBoundConstant;
    }
    public double getQThetaNormalizer(){
      return logQThetaNormalizer;
    }
    public double getQTauWNormalizer(){
      return logQTauWNormalizer;
    }      

    public double getPriorAlpha(PriorSpecification priors, int annotator, int trustedAnnotator){
      return annotator==trustedAnnotator? PANClassificationMeasurementModel.TRUSTED_ALPHA: priors.getBGamma();
    }

    public double getPriorBeta(PriorSpecification priors, int annotator, int trustedAnnotator){
      return annotator==trustedAnnotator? PANClassificationMeasurementModel.TRUSTED_BETA: priors.getCGamma();
    }

    
    private void initialize(Dataset data, PriorSpecification priors, int trustedAnnotator){
      annotatorMeasurementCounter = new DenseCounter(data.getInfo().getNumAnnotators());
      double priorDelta = priors.getBTheta();
      
      for (Measurement measurement: data.getMeasurements()){
        annotatorMeasurementCounter.incrementCount(measurement.getAnnotator(), 1);
      }
      
      // lower bound constant
      double term1 = -1 * GammaFunctions.logBetaSymmetric(priorDelta,data.getInfo().getNumClasses());
      double term2 = 0;
      for (int j=0; j<data.getInfo().getNumAnnotators(); j++){
        double priorAlpha = getPriorAlpha(priors,j,trustedAnnotator), priorBeta = getPriorBeta(priors,j,trustedAnnotator);
        term2 += (priorAlpha * Math.log(priorBeta) - Gamma.logGamma(priorAlpha));
      }
      double term3 = 0;
      for (int j=0; j<data.getInfo().getNumAnnotators(); j++){
        term3 -= ((annotatorMeasurementCounter.getCount(j)/2.0) * Math.log(2.0*Math.PI));
      }
      logQThetaNormalizer = term1;
      logQTauWNormalizer = term2 + term3;
      logLowerBoundConstant = logQThetaNormalizer + logQTauWNormalizer;
    }
  }


  
}
