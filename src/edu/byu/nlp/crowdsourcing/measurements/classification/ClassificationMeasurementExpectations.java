package edu.byu.nlp.crowdsourcing.measurements.classification;

import java.util.Set;

import com.google.common.collect.Sets;

import edu.byu.nlp.crowdsourcing.measurements.MeasurementExpectation;
import edu.byu.nlp.data.measurements.ClassificationMeasurements.BasicClassificationLabelProportionMeasurement;
import edu.byu.nlp.data.measurements.ClassificationMeasurements.ClassificationAnnotationMeasurement;
import edu.byu.nlp.data.measurements.ClassificationMeasurements.ClassificationProportionMeasurement;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.Measurement;
import edu.byu.nlp.stats.MutableSum;
import edu.byu.nlp.util.IntArrays;

/**
 * In the variation equations for the measurment model, 
 * there are seveal place where it is necessary to compute 
 * the expected value of a global measurement. In 
 * general, computing this requires summing over all 
 * instances. Doing that naively each time it is 
 * required (once per measurement) would be prohibitively 
 * expensive. So instead we maintain each expectation as 
 * a sum and update it whenever a log q(y_i|logNuY_i) 
 * (that it depends on) changes. As a convenience, 
 * each expectation also pre-calculates and returns the set of 
 * indices that it depends on, so that it can be 
 * ignored during irrelevant updates.
 *  
 * @author plf1
 *
 */
public class ClassificationMeasurementExpectations {

  
  public static abstract class AbstractExpectation implements MeasurementExpectation<Integer>{
    
    private MutableSum expectation = new MutableSum();
    private Measurement measurement;
    private Dataset dataset;

    public AbstractExpectation(Measurement measurement, Dataset dataset, double[][] logNuY){
      this.measurement=measurement;
      this.dataset=dataset;
      for (int i=0; i<logNuY.length; i++){
        setLogNuY_i(i, logNuY[i]);
      }
    }
    @Override
    public Dataset getDataset() {
      return dataset;
    }
    @Override
    public int getAnnotator() {
      return measurement.getAnnotator();
    }
    @Override
    public void setLogNuY_i(int docIndex, double[] logNuY_i) {
      expectation.setSummand(docIndex, expectedValue_i(docIndex, logNuY_i));
      
    }
    @Override
    public double expectedValue() {
      return expectation.getSum();
    }
    protected abstract double expectedValue_i(int docIndex, double[] logNuY_i);
  }
  
  
  
  public static class LabelProportion extends AbstractExpectation{

    private ClassificationProportionMeasurement measurement;

    public LabelProportion(ClassificationProportionMeasurement measurement, Dataset dataset, double[][] logNuY) {
      super((Measurement) measurement, dataset, logNuY);
      this.measurement=measurement;
    }
    @Override
    public double featureValue(int docIndex, Integer label) {
      return (label==measurement.getLabel())? 1: 0;
    }
    @Override
    protected double expectedValue_i(int docIndex, double[] logNuY_i) {
      return Math.exp(logNuY_i[measurement.getLabel()]);
    }
    @Override
    public Set<Integer> getDependentIndices() {
      // all of them
      return Sets.newHashSet(
          IntArrays.asList(
              IntArrays.sequence(0, getDataset().getInfo().getNumDocuments())));
    }
    @Override
    public Measurement getMeasurement() {
      return (Measurement) measurement;
    }
    
  }
  
  
  
  public static class Annotation extends AbstractExpectation{

    private ClassificationAnnotationMeasurement measurement;
    public Annotation(ClassificationAnnotationMeasurement measurement, Dataset dataset, double[][] logNuY) {
      super((Measurement) measurement, dataset, logNuY);
      this.measurement=measurement;
    }
    @Override
    public double featureValue(int docIndex, Integer label) {
      return (docIndex==this.measurement.getDocumentIndex() && label==this.measurement.getLabel())? measurement.getValue(): 0;
    }
    @Override
    protected double expectedValue_i(int docIndex, double[] logNuY_i) {
      // q(y) * annotation_value
      return Math.exp(logNuY_i[measurement.getLabel()]) * measurement.getValue();
    }
    @Override
    public Set<Integer> getDependentIndices() {
      return Sets.newHashSet(measurement.getDocumentIndex());
    }
    @Override
    public Measurement getMeasurement() {
      return (Measurement) measurement;
    }

  }
 
  

  public static MeasurementExpectation<Integer> fromMeasurement(Measurement measurement, Dataset dataset, double[][] logNuY){
    if (measurement instanceof ClassificationAnnotationMeasurement){
      return new ClassificationMeasurementExpectations.Annotation(
          (ClassificationAnnotationMeasurement) measurement, dataset, logNuY);
    }
    else if (measurement instanceof BasicClassificationLabelProportionMeasurement){
      return new ClassificationMeasurementExpectations.LabelProportion(
          (ClassificationProportionMeasurement) measurement, dataset, logNuY);
    }
    else{
      throw new IllegalArgumentException("unknown measurement type: "+measurement.getClass().getName());
    }
    
  }
  
}
