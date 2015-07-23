package edu.byu.nlp.crowdsourcing.measurements.classification;

import java.util.Set;

import com.google.common.collect.Sets;

import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.DatasetInstance;
import edu.byu.nlp.data.types.Measurement;

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

  
  public interface MeasurementExpectation{
    Set<Integer> getDependentIndices(Dataset data);
    void setLogNuY_i(int i, double[] logNuY);
    double getValue();
  }
  
  public static abstract class AbstractExpectation implements MeasurementExpectation{
    
    private Measurement<Integer> measurement;
  
    public MeasurementExpectation(Measurement<Integer> measurement, double[][] logNuY){
      this.measurement=measurement;
      for (int i=0; i<logNuY.length; i++){
        setLogNuY_i(i, logNuY[i]);
      }
    }
    
    @Override
    public Set<Integer> getDependentIndices(Dataset data){
      Set<Integer> indices = Sets.newHashSet();
      for (DatasetInstance item: data){
        int index = item.getInfo().getSource();
        for (int c=0; c<data.getInfo().getNumClasses(); c++){
          // add all measurements that fire (pos or neg) for at least one class label
          if (measurement.featureValue(index, c)!=0){
            indices.add(index);
            break;
          }
        }
      }
      return indices;
    }
    
    @Override
    public double getValue(){
    }
    
  }
  
  
  public static class LabelProportion extends MeasurementExpectation{

    @Override
    public double featureValue(int docIndex, Integer label) {
      return (label==this.getLabel())? 1: 0;
    }
  }
  
  public static class Annotation extends MeasurementExpectation{

    public Annotation(Measurement<Integer> measurement, double[][] logNuY) {
      super(measurement, logNuY);
      // TODO Auto-generated constructor stub
    }

    @Override
    public void setLogNuY_i(int i, double[] logNuY) {
      // TODO Auto-generated method stub
      
    }
    @Override
    public double featureValue(int docIndex, Integer label) {
      return (docIndex==this.getIndex() && label==this.getLabel())? getAnnotation(): 0;
    }
    
  }

}
