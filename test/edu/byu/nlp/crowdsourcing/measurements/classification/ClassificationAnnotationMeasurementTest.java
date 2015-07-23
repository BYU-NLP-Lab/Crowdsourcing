package edu.byu.nlp.crowdsourcing.measurements.classification;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import edu.byu.nlp.data.measurements.ClassificationAnnotationMeasurement;

public class ClassificationAnnotationMeasurementTest {

  private static final double THRESHOLD = 1e-20;
  
  @Test
  public void test() {
    int annotator = 0;
    int index = 9;
    int label = 1;
    double annotation = 0.9;
    double confidence = 0.8;
    ClassificationAnnotationMeasurement m = new ClassificationAnnotationMeasurement(annotator, annotation, confidence, index, label);

    for (int i=0; i<20; i++){
      for (int j=0; j<20; j++){
        if (i==index && j==label){
          assertEquals(annotation, m.featureValue(i, j), THRESHOLD);
        }
        else{
          assertEquals(0, m.featureValue(i, j), THRESHOLD);
        }
      }
    }

  }

}
