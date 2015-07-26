package edu.byu.nlp.crowdsourcing.measurements;

import java.util.Set;

import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.Measurement;

/**
 * A Measurement generalizes the concept of a label or annotation.
 * It is based on Percy Liang's paper:
 * - "Learning from Measurements in Exponential Families." 
 * - http://www.cs.berkeley.edu/~jordan/papers/liang-jordan-klein-icml09.pdf
 * 
 * This class encodes the expected value of a measurement 
 * wrt an approximate variational log posterior 
 * distribution parameterized by logNuY.
 */
public interface MeasurementExpectation<L> {

  /**
   * Measurements are defined with respect to a given dataset.
   * (e.g., they might reference a specific instance inside of a dataset). 
   */
  Dataset getDataset();
  
  /**
   * Get the actual measurement data the this function was constructed from.
   */
  Measurement getMeasurement();

  /**
   * Who was responsible for generating this measurement? 
   */
  int getAnnotator();
    
  /**
   * Returns the value of the Measurement on a 
   * single item. Mathematically, Measurments are 
   * defined in terms of parameters (x,y) representing 
   * (possibly structured) data instance X and 
   * (possibly structured) hypothesis label Y. 
   * 
   * Measurements encode all sorts of specific, fine-grained human 
   * judgments. For example, one kind of measurement 
   * encodes a specific label for a specific data item i by returning 
   * 1(x=x_i, y=y_i) based on prior knowledge of x_i and y_i. 
   * 
   * Measurements represent specific human judgments and should 
   * be similarly immutable (changed only if a human changes 
   * their judgment). They are not mutated as a part of 
   * algorithmic computation. 
   */
  double featureValue(int docIndex, L label);
  
  /**
   * Many measurements depend on only a single data instance 
   * (e.g., annotations of single instances). This method 
   * yields the set of document indices that this measurement 
   * depends on (so that it can be updated via 
   * setLogNuY_i() only when absolutely necessary).
   */
  Set<Integer> getDependentIndices();
  
  /**
   * For efficiency reasons, we must be able to alter a single element of 
   * the expectation at a time (as they are updated) rather than recomputing 
   * the entire expectation each time a single element changes. 
   */
  void setLogNuY_i(int docIndex, double[] logNuY_i);
  
  /**
   * The value of E_y[sum_i feature(x_i,y)] wrt the approximate distribution q(y)
   */
  double expectedValue();
  
  
}