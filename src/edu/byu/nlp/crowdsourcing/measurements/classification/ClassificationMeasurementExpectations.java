package edu.byu.nlp.crowdsourcing.measurements.classification;

import java.util.Map;
import java.util.Set;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.collect.Range;
import com.google.common.collect.Sets;

import edu.byu.nlp.crowdsourcing.measurements.MeasurementExpectation;
import edu.byu.nlp.data.measurements.ClassificationMeasurements.BasicClassificationLabelProportionMeasurement;
import edu.byu.nlp.data.measurements.ClassificationMeasurements.BasicClassificationLabeledPredicateMeasurement;
import edu.byu.nlp.data.measurements.ClassificationMeasurements.ClassificationAnnotationMeasurement;
import edu.byu.nlp.data.measurements.ClassificationMeasurements.ClassificationLabeledPredicateMeasurement;
import edu.byu.nlp.data.measurements.ClassificationMeasurements.ClassificationProportionMeasurement;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.DatasetInstance;
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

    private MutableSum piecewiseSquaredExpectation = new MutableSum();
    private MutableSum expectationOfSquaredSigmas = new MutableSum();
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
      double expectedValue = expectedValue_i(docIndex, logNuY_i);
      expectation.setSummand(docIndex, expectedValue);
      piecewiseSquaredExpectation.setSummand(docIndex, Math.pow(expectedValue, 2));
      expectationOfSquaredSigmas.setSummand(docIndex, expectedValueOfSquaredSigma_i(docIndex, logNuY_i));
    }
    @Override
    public double sumOfExpectedValuesOfSigma() {
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
    @Override
    public double sumOfExpectedValuesOfSquaredSigma() {
      return expectationOfSquaredSigmas.getSum();
    }
    @Override
    public double piecewiseSquaredSumOfExpectedValuesOfSigma() {
      return piecewiseSquaredExpectation.getSum();
    }
    protected double expectedValueOfSquaredSigma_i(int docIndex, double[] logNuY_i) {
      return expectedValue_i(docIndex, logNuY_i);
    }
    @Override
    public double getExpectedValue(int docIndex) {
      return expectation.getSummand(docIndex);
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
    @Override
    public Range<Double> getRange() {
      return Range.closed(0.0, (double)getDataset().getInfo().getNumDocuments());
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
      if (docIndex!=index){
        logger.warn("DANGER! An annotation measurement is being asked for indices it doesn't depend on! This is VERY inefficient and probably indicates a bug!");
        return 0;
      }
      return (docIndex==index && label==getClassificationMeasurement().getLabel())? 1: 0;
    }
    @Override
    protected double expectedValue_i(int docIndex, double[] logNuY_i) {
      if (docIndex!=index){
        logger.warn("DANGER! An annotation measurement is being asked for indices it doesn't depend on! This is VERY inefficient and probably indicates a bug!");
        return 0;
      }
      return Math.exp(logNuY_i[getClassificationMeasurement().getLabel()]);
    }
    @Override
    public Set<Integer> getDependentIndices() {
      return Sets.newHashSet(index);
    }
    public ClassificationAnnotationMeasurement getClassificationMeasurement(){
      return (ClassificationAnnotationMeasurement) getMeasurement();
    }
    @Override
    public Range<Double> getRange() {
      return Range.closed(0.0, 1.0);
    }
  }
 

  public static class LabeledPredicate extends AbstractExpectation{
    private Set<Integer> dependentIndices;
    private int label;
    private int wordCount;
    private Map<String, Integer> documentIndices;
    private String predicate;
    public LabeledPredicate(ClassificationLabeledPredicateMeasurement measurement, Dataset dataset, Map<String, Integer> documentIndices) {
      super((Measurement) measurement, dataset);
      this.label = measurement.getLabel();
      this.documentIndices=documentIndices;
      this.predicate=measurement.getPredicate();
    }
    @Override
    public double featureValue(int docIndex, Integer label) {
      if (!getDependentIndices().contains(docIndex)){
        logger.error("DANGER! LabeledPredicate is being asked for indices that it doesn't depend on! This is VERY slow and probably a bug!!!");
        return 0;
      }
      return label==this.label? 1: 0;
    }
    @Override
    public Range<Double> getRange() {
      calculateDependentIndices();
      return Range.closed(0.0, (double)wordCount);
    }
    @Override
    public Set<Integer> getDependentIndices() {
      calculateDependentIndices();
      return dependentIndices;
    }
    @Override
    protected double expectedValue_i(int docIndex, double[] logNuY_i) {
      return Math.exp(logNuY_i[label]);
    }
    private void calculateDependentIndices(){
      if (dependentIndices==null){
        this.dependentIndices = Sets.newHashSet();
        Integer wordIndex = getDataset().getInfo().getFeatureIndexer().indexOf(predicate);
        // unknown word
        if (wordIndex==null){
          throw new IllegalStateException("Tried to add a predicate (word) that was not found in the corpus! This should never happen!");
        }
        
        this.wordCount = 0;
        for (DatasetInstance instance: getDataset()){
          Double localWordCount = instance.asFeatureVector().getValue(wordIndex);
          if (localWordCount!=null){
            dependentIndices.add(documentIndices.get(instance.getInfo().getRawSource()));
            this.wordCount += localWordCount;
          }
        }
      }
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
    else if (measurement instanceof BasicClassificationLabeledPredicateMeasurement){
      expectation = new ClassificationMeasurementExpectations.LabeledPredicate(
          (ClassificationLabeledPredicateMeasurement) measurement, dataset, documentIndices);
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
  
  public static class ScaledMeasurementExpectation<L> implements MeasurementExpectation<L>{
    private MeasurementExpectation<L> delegate;
    private double rangeMagnitude, rangeMagnitudeSquared;
    public static <L> ScaledMeasurementExpectation<L> from(MeasurementExpectation<L> delegate){
      return new ScaledMeasurementExpectation<L>(delegate);
    }
    public ScaledMeasurementExpectation(MeasurementExpectation<L> delegate){
      this.delegate=delegate;
      this.rangeMagnitude=delegate.getRange().upperEndpoint() - delegate.getRange().lowerEndpoint();
      this.rangeMagnitudeSquared = Math.pow(rangeMagnitude, 2);
      Preconditions.checkArgument(delegate.getRange().lowerEndpoint()==0.0,"Ranges that must be shifted (non-zero lower endpoint) are not currently supported."); 
    }
    @Override
    public Dataset getDataset() {
      return delegate.getDataset();
    }
    @Override
    public Measurement getMeasurement() {
      return delegate.getMeasurement();
    }
    @Override
    public int getAnnotator() {
      return delegate.getAnnotator();
    }
    @Override
    public double featureValue(int docIndex, L label) {
      return delegate.featureValue(docIndex, label) / rangeMagnitude;
    }
    @Override
    public Range<Double> getRange() {
      return delegate.getRange();
    }
    @Override
    public Set<Integer> getDependentIndices() {
      return delegate.getDependentIndices();
    }
    @Override
    public void setLogNuY_i(int docIndex, double[] logNuY_i) {
      delegate.setLogNuY_i(docIndex, logNuY_i);
    }
    @Override
    public double sumOfExpectedValuesOfSigma() {
      return delegate.sumOfExpectedValuesOfSigma() / rangeMagnitude;
    }
    @Override
    public void setSummandVisible(int i, boolean visible) {
      delegate.setSummandVisible(i, visible);
    }
    @Override
    public double sumOfExpectedValuesOfSquaredSigma() {
      return delegate.sumOfExpectedValuesOfSquaredSigma() / rangeMagnitudeSquared;
    }
    @Override
    public double piecewiseSquaredSumOfExpectedValuesOfSigma() {
      return delegate.piecewiseSquaredSumOfExpectedValuesOfSigma() / rangeMagnitudeSquared;
    }
    @Override
    public double getExpectedValue(int docIndex) {
      return delegate.getExpectedValue(docIndex);
    }
    @Override
    public String toString() {
      return "Scaled+"+delegate.toString();
    }
  }
  
}
