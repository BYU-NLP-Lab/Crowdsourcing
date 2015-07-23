package edu.byu.nlp.crowdsourcing.measurements.classification;

import edu.byu.nlp.crowdsourcing.measurements.AbstractMeasurement;
import edu.byu.nlp.data.types.Dataset;

public class ClassificationMeasurements {

  /**
   * A measurement that encodes a basic annotation (somebody evaluates 
   * the degree to which a hypothesis label y is correct). This can be any real 
   * number, but it makes sense to keep it between 1 (the hypothesis label y is perfectly
   * correct) and -1 (the hypothesis label y is entirely incorrect)
   * 
   * @author plf1
   * 
   */
  public static class Annotation extends AbstractMeasurement<Integer>{

    private int i;
    private int y;
    private double value;

    public Annotation(Dataset dataset, int annotator, int i, int y, double value){
      super(dataset, annotator);
      this.i=i;
      this.y=y;
      this.value=value;
    }
    
    @Override
    public double featureValue(int docIndex, Integer label) {
      return (docIndex==i && label==y)? value: 0;
    }

  }

}
