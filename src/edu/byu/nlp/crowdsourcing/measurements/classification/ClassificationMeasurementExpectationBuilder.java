package edu.byu.nlp.crowdsourcing.measurements.classification;

import edu.byu.nlp.crowdsourcing.measurements.classification.ClassificationMeasurementExpectations.MeasurementExpectation;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.Measurement;

public class ClassificationMeasurementExpectationBuilder {

  private Dataset data;
  private double[][] logNuY;
  private Measurement<Integer> measurement;

  public ClassificationMeasurementExpectationBuilder setData(Dataset data){
    this.data=data;
    return this;
  }

  public ClassificationMeasurementExpectationBuilder setLogNuY(double[][] logNuY){
    this.logNuY=logNuY;
    return this;
  }

  public ClassificationMeasurementExpectationBuilder setMeasurement(ClassificationMeasurementExpectations.MeasurementExpectation expectation){
    this.measurement=measurement;
    return this;
  }
  
  public MeasurementExpectation build(){
    if (measurement instanceof ClassificationMeasurements.Annotation){
      
    }
    else{
      throw new IllegalArgumentException("unknown measurement type: "+measurement.getClass());
    }
  }
  
}
