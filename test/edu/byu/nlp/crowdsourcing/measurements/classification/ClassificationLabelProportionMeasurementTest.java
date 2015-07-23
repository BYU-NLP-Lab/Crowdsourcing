package edu.byu.nlp.crowdsourcing.measurements.classification;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import edu.byu.nlp.data.measurements.ClassificationLabelProportionMeasurement;

public class ClassificationLabelProportionMeasurementTest {

  private static final double THRESHOLD = 1e-20;
  
  @Test
  public void test() {
    int annotator = 0;
    int label = 1;
    double proportion = 0.7;
    double confidence = 0.2;
    ClassificationLabelProportionMeasurement m = new ClassificationLabelProportionMeasurement(annotator, proportion, confidence, label);

    assertEquals(1, m.featureValue(5, 1), THRESHOLD);
    assertEquals(1, m.featureValue(0, 1), THRESHOLD);
    assertEquals(1, m.featureValue(-1, 1), THRESHOLD);
    assertEquals(0, m.featureValue(5, 4), THRESHOLD);
    assertEquals(0, m.featureValue(-7, -2), THRESHOLD);
    assertEquals(0, m.featureValue(2, 3), THRESHOLD);
  }

}
