package measurements;

import java.util.Arrays;

import measurements.MeasurementModel.Priors;

import org.apache.commons.math3.random.RandomGenerator;

import edu.byu.nlp.crowdsourcing.ModelInitialization.AssignmentInitializer;
import edu.byu.nlp.util.DoubleArrays;

public abstract class MeasurementModelBuilder {

  private Priors priors;
  private RandomGenerator rnd;
  private AssignmentInitializer yInitializer;

  public MeasurementModelBuilder(){  }
  
  public MeasurementModelBuilder setPriors(MeasurementModel.Priors priors){
    this.priors=priors;
    return this;
  }

  public MeasurementModelBuilder setRnd(RandomGenerator rnd){
    this.rnd=rnd;
    return this;
  }

  public MeasurementModelBuilder setYInitializer(AssignmentInitializer yInitializer){
    this.yInitializer=yInitializer;
    return this;
  }
  
  public MeasurementModel build(){
    int[] y = null;
    return initializeModel(priors, y, rnd);
  }
  
  protected abstract MeasurementModel initializeModel(MeasurementModel.Priors priors, int[] y, RandomGenerator rnd);
  
  
  
}
