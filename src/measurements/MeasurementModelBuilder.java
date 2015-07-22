package measurements;

import org.apache.commons.math3.random.RandomGenerator;

import edu.byu.nlp.crowdsourcing.ModelInitialization.AssignmentInitializer;
import edu.byu.nlp.crowdsourcing.PriorSpecification;

public abstract class MeasurementModelBuilder {

  private PriorSpecification priors;
  private RandomGenerator rnd;
  private AssignmentInitializer yInitializer;

  public MeasurementModelBuilder(){  }
  
  public MeasurementModelBuilder setPriors(PriorSpecification priors){
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
    // TODO: pre-compute data stats used by measurement models
    int[] y = null;
    return initializeModel(priors, y, rnd);
  }
  
  protected abstract MeasurementModel initializeModel(PriorSpecification priors, int[] y, RandomGenerator rnd);
  
  
  public static MeasurementModelBuilder initializeBuilder(MeasurementModelBuilder builder, PriorSpecification priors, AssignmentInitializer yInitializer, RandomGenerator rnd){
    return builder
        .setPriors(priors)
        .setRnd(rnd)
        .setYInitializer(yInitializer);
  }
  
}
