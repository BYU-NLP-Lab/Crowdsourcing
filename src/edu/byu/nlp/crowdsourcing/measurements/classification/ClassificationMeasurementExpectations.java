package edu.byu.nlp.crowdsourcing.measurements.classification;

import java.util.Map;
import java.util.Set;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.MoreObjects;
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

  private static final Logger logger = LoggerFactory.getLogger(ClassificationMeasurementExpectations.class);
  
  public static abstract class AbstractExpectation implements MeasurementExpectation<Integer>{
    
    private MutableSum expectation = new MutableSum();
    private Measurement measurement;
    private Dataset dataset;

    public AbstractExpectation(Measurement measurement, Dataset dataset){
      this.measurement=measurement;
      this.dataset=dataset;
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
    @Override
    public Measurement getMeasurement() {
      return measurement;
    }
    @Override
    public void setSummandVisible(int i, boolean visible) {
      expectation.setSummandActive(i,visible);
    }
    @Override
    public String toString() {
      return MoreObjects.toStringHelper(MeasurementExpectation.class)
          .add("meas",getMeasurement())
          .toString();
    }
    protected abstract double expectedValue_i(int docIndex, double[] logNuY_i);
  }
  
  
  
  public static class LabelProportion extends AbstractExpectation{

    public LabelProportion(ClassificationProportionMeasurement measurement, Dataset dataset) {
      super((Measurement) measurement, dataset);
    }
    @Override
    public double featureValue(int docIndex, Integer label) {
      return (label==getClassificationMeasurement().getLabel())? 1: 0;
    }
    @Override
    protected double expectedValue_i(int docIndex, double[] logNuY_i) {
      return Math.exp(logNuY_i[getClassificationMeasurement().getLabel()]);
    }
    @Override
    public Set<Integer> getDependentIndices() {
      // all of them
      return Sets.newHashSet(
          IntArrays.asList(
              IntArrays.sequence(0, getDataset().getInfo().getNumDocuments())));
    }
    public ClassificationProportionMeasurement getClassificationMeasurement(){
      return (ClassificationProportionMeasurement) getMeasurement();
    }
  }
  
  
  
  public static class Annotation extends AbstractExpectation{

    private int index;
    public Annotation(ClassificationAnnotationMeasurement measurement, Dataset dataset, Map<String,Integer> documentIndices) {
      super((Measurement) measurement, dataset);
      this.index = documentIndices.get(measurement.getDocumentSource());
    }
    @Override
    public double featureValue(int docIndex, Integer label) {
      ClassificationAnnotationMeasurement meas = getClassificationMeasurement();
      return (docIndex==index && label==meas.getLabel())? meas.getValue(): 0;
    }
    @Override
    protected double expectedValue_i(int docIndex, double[] logNuY_i) {
      ClassificationAnnotationMeasurement meas = getClassificationMeasurement();
      if (docIndex!=index){
        logger.warn("DANGER! An annotation measurement is being asked for indices it doesn't depend on! This is VERY inefficient and probably indicates a bug!");
        return 0;
      }
      // q(y) * annotation_value
      return Math.exp(logNuY_i[meas.getLabel()]) * meas.getValue();
    }
    @Override
    public Set<Integer> getDependentIndices() {
      return Sets.newHashSet(index);
    }
    public ClassificationAnnotationMeasurement getClassificationMeasurement(){
      return (ClassificationAnnotationMeasurement) getMeasurement();
    }
  }
 
  
  public static MeasurementExpectation<Integer> fromMeasurement(Measurement measurement, Dataset dataset, Map<String,Integer> documentIndices, double[][] logNuY){
    MeasurementExpectation<Integer> expectation;
    
    // create the expectation
    if (measurement instanceof ClassificationAnnotationMeasurement){
      expectation = new ClassificationMeasurementExpectations.Annotation(
          (ClassificationAnnotationMeasurement) measurement, dataset, documentIndices);
    }
    else if (measurement instanceof BasicClassificationLabelProportionMeasurement){
      expectation = new ClassificationMeasurementExpectations.LabelProportion(
          (ClassificationProportionMeasurement) measurement, dataset);
    }
    else{
      throw new IllegalArgumentException("unknown measurement type: "+measurement.getClass().getName());
    }
    
    // initialize the expectation
    for (Integer docIndex: expectation.getDependentIndices()){
      expectation.setLogNuY_i(docIndex, logNuY[docIndex]);
    }
    
    return expectation;
    
  }
  
}
