package edu.byu.nlp.crowdsourcing.measurements.classification;

import java.util.Map;
import java.util.Set;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.collect.Range;
import com.google.common.collect.Sets;

import edu.byu.nlp.crowdsourcing.measurements.MeasurementExpectation;
import edu.byu.nlp.data.measurements.ClassificationMeasurements.ClassificationAnnotationMeasurement;
import edu.byu.nlp.data.measurements.ClassificationMeasurements.ClassificationLabeledLocationMeasurement;
import edu.byu.nlp.data.measurements.ClassificationMeasurements.ClassificationLabeledPredicateMeasurement;
import edu.byu.nlp.data.measurements.ClassificationMeasurements.ClassificationLabelProportionMeasurement;
import edu.byu.nlp.data.streams.PorterStemmer;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.DatasetInstance;
import edu.byu.nlp.data.types.Measurement;
import edu.byu.nlp.dataset.Datasets;
import edu.byu.nlp.stats.MutableSum;
import edu.byu.nlp.util.DoubleArrays;
import edu.byu.nlp.util.IntArrays;
import edu.byu.nlp.util.Integers;
import edu.byu.nlp.util.Pair;

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

    public LabelProportion(ClassificationLabelProportionMeasurement measurement, Dataset dataset) {
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
    public ClassificationLabelProportionMeasurement getClassificationMeasurement(){
      return (ClassificationLabelProportionMeasurement) getMeasurement();
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
    private double wordCount;
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
      Preconditions.checkState(wordCount>0, "illegal wordCount value: "+wordCount);
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
        String stemmedPredicate = predicate;
        // if we don't know the predicate word, try stemming it
        if (!getDataset().getInfo().getFeatureIndexer().contains(predicate)){
          stemmedPredicate = new PorterStemmer().apply(predicate);
        }
        Integer wordIndex = getDataset().getInfo().getFeatureIndexer().indexOf(stemmedPredicate);
        // unknown word
//        if (wordIndex==getDataset().getInfo().getFeatureIndexer().indexOf(null)){
//          logger.debug("Tried to add a predicate ("+predicate+") that was not found in the corpus! This shouldn't happen too much.");
//        }
        
        this.wordCount = 0;
        for (DatasetInstance instance: getDataset()){
          Double rawLocalWordCount = instance.asFeatureVector().getValue(wordIndex);
          if (rawLocalWordCount!=null){
            int localWordCount = Integers.fromDouble(instance.asFeatureVector().getValue(wordIndex), 1e-20);
            dependentIndices.add(documentIndices.get(instance.getInfo().getRawSource()));
            Preconditions.checkState(localWordCount>0, "A sparse document feature vector contains a 0 entry. This should never happen.");
            this.wordCount += localWordCount;
          }
        }
      }
    }
  }

  /**
   * WARNING: NOT thread-safe. A shared cache for pairwise cosine distances between documents
   */
  private static class LocationAdjacencyMatrix{
    private LocationAdjacencyMatrix(){}
    private static double[][] matrix = null;
    protected static double[][] getMatrix(Dataset dataset, Map<String, Integer> documentIndices){
      if (matrix==null){
        logger.info("pre-calculating cosine adjacency matrix...");
        Map<Pair<String, String>, Double> srcMatrix = Datasets.calculateCosineAdjacencyMatrix(dataset);
        logger.info("done!");
        matrix = new double[documentIndices.size()][documentIndices.size()];
        for (Pair<String, String> entry: srcMatrix.keySet()){
          int i1 = documentIndices.get(entry.getFirst());
          int i2 = documentIndices.get(entry.getSecond());
          Preconditions.checkState(matrix[i1][i2]==0);
          matrix[i1][i2] = srcMatrix.get(entry);
        }
      }
      
      return matrix;
    }
  }
  
  public static class LabeledLocation extends AbstractExpectation{
    private int label;
    private double totalSimilarity = -1;
    private Map<String, Integer> documentIndices;
    private double[] location;
    private double[] cosines, cosinesSquared; // cosine distance of this location with all documents
    private String source;
    private int neighbors;
    private Set<Integer> dependentIndices;
//    public LabeledLocation(ClassificationLabeledLocationMeasurement measurement, Dataset dataset, Map<String, Integer> documentIndices) {
//      this(measurement, dataset, documentIndices, Integer.MAX_VALUE);
//    }
    public LabeledLocation(ClassificationLabeledLocationMeasurement measurement, Dataset dataset, Map<String, Integer> documentIndices, int neighbors) {
      super((Measurement) measurement, dataset);
      this.label = measurement.getLabel();
      this.documentIndices=documentIndices;
      this.location=measurement.getLocation();
      this.source=measurement.getSource();
      this.neighbors = neighbors<=0? Integer.MAX_VALUE: neighbors; // value <=0 means unlimited
      Preconditions.checkArgument(label>=0);
      Preconditions.checkArgument(location!=null || source!=null);
      Preconditions.checkNotNull(measurement);
      Preconditions.checkNotNull(dataset);
      Preconditions.checkNotNull(documentIndices);
    }
    @Override
    public double featureValue(int docIndex, Integer label) {
      calculateCosines(neighbors);
      if (!getDependentIndices().contains(docIndex)){
        logger.error("DANGER! LabeledLocation is being asked for indices that it doesn't depend on! This is VERY slow and probably a bug!!!");
        return 0;
      }
      return label==this.label? cosines[docIndex]: 0;
    }
    @Override
    public Range<Double> getRange() {
      calculateCosines(neighbors);
      Preconditions.checkState(totalSimilarity>0, "illegal totalSimilarity value: "+totalSimilarity);
      return Range.closed(0.0, totalSimilarity);
    }
    
    @Override
    public Set<Integer> getDependentIndices() {
      calculateCosines(neighbors);
      return dependentIndices;
    }
    @Override
    protected double expectedValue_i(int docIndex, double[] logNuY_i) {
      calculateCosines(neighbors);
      return cosines[docIndex] * Math.exp(logNuY_i[label]);
    }
    @Override
    protected double expectedValueOfSquaredSigma_i(int docIndex, double[] logNuY_i) {
      return cosinesSquared[docIndex] * Math.exp(logNuY_i[label]);
    }
    private void calculateCosines(int kNeighbors){
      if (cosines==null){
        if (source!=null){
          // this location refers to an instance in the dataset. Use pre-computed, shared vectors
          cosines = LocationAdjacencyMatrix.getMatrix(getDataset(), documentIndices)[documentIndices.get(source)];
        }
        else{
          cosines = new double[getDataset().getInfo().getNumDocuments()];
          // this location is a custom location. Compute manually
          for (DatasetInstance inst: getDataset()){
            RealVector v1 = new ArrayRealVector(location);
            RealVector v2 = inst.asFeatureVector().asApacheSparseRealVector();
            cosines[documentIndices.get(inst.getInfo().getRawSource())] = v1.cosine(v2);
          }
        }
        // now zero out all but the top k entries per row. This will have the effect (in the feature, expectedValue, 
        // and range functions) of adding an additional indicator function to the measurement function definition 
        // that returns 1 only for the top k closest locations. also calculate dependent indices (all the  
        // non-zero cosines)
        dependentIndices = Sets.newHashSet(DoubleArrays.argMaxList(kNeighbors, cosines));
        for (int i=0; i<getDataset().getInfo().getNumDocuments(); i++){
          if (!dependentIndices.contains(i)){
            cosines[i] = 0;
          }
        }
      }
      cosinesSquared  = DoubleArrays.pow(cosines,2);
      totalSimilarity = DoubleArrays.sum(cosines);
    }
  }
  
  public static MeasurementExpectation<Integer> fromMeasurement(Measurement measurement, Dataset dataset, Map<String,Integer> documentIndices, double[][] logNuY){
    MeasurementExpectation<Integer> expectation;
    
    // create the expectation
    if (measurement instanceof ClassificationAnnotationMeasurement){
      expectation = new ClassificationMeasurementExpectations.Annotation(
          (ClassificationAnnotationMeasurement) measurement, dataset, documentIndices);
    }
    else if (measurement instanceof ClassificationLabelProportionMeasurement){
      expectation = new ClassificationMeasurementExpectations.LabelProportion(
          (ClassificationLabelProportionMeasurement) measurement, dataset);
    }
    else if (measurement instanceof ClassificationLabeledPredicateMeasurement){
      expectation = new ClassificationMeasurementExpectations.LabeledPredicate(
          (ClassificationLabeledPredicateMeasurement) measurement, dataset, documentIndices);
    }
    else if (measurement instanceof ClassificationLabeledLocationMeasurement){
      ClassificationLabeledLocationMeasurement locMeas = (ClassificationLabeledLocationMeasurement) measurement;
      expectation = new ClassificationMeasurementExpectations.LabeledLocation(
          locMeas, dataset, documentIndices, locMeas.getNeighbors());
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
