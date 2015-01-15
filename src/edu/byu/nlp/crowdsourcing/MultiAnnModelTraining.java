/**
 * Copyright 2014 Brigham Young University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package edu.byu.nlp.crowdsourcing;

import java.util.logging.Logger;

/**
 * @author pfelt
 */
public class MultiAnnModelTraining {

  private static final Logger logger = Logger.getLogger(MultiAnnModelTraining.class.getName());
  public static final double MAXIMIZE_IMPROVEMENT_THRESHOLD = 1e-6;
  public static final int MAXIMIZE_MAX_ITERATIONS = 20;
  public static final int MAXIMIZE_BATCH_SIZE = 3;
  
  public interface Operation{
    void doOperation(MultiAnnModel model);
  }
  
  public static enum OperationType{
    NONE,
    SAMPLE,
    SAMPLEY,
    SAMPLEM,
    MAXIMIZE,
    MAXIMIZEY,
    MAXIMIZEM
  }
  
  public static class OperationParser{
    public static Operation parse(String rawOp){
      String[] fields = rawOp.split("-");
      OperationType type = OperationType.valueOf(fields[0].toUpperCase());
      int samples = fields.length>=2? Integer.parseInt(fields[1]): 1;
      double temp = fields.length>=3? Double.parseDouble(fields[2]): 1;
      switch(type){
      case NONE:
        return new Operation() {
          @Override
          public void doOperation(MultiAnnModel model) {
            // do nothing
          }
        };
      case MAXIMIZE:
        return new MaximizationOperation(true){
          @Override
          protected void maximize(MultiAnnModel model) {
            model.maximize();
          }
        };
      case MAXIMIZEM:
        return new MaximizationOperation(false){
          @Override
          protected void maximize(MultiAnnModel model) {
            model.maximizeM();
          }
        };
      case MAXIMIZEY:
        return new MaximizationOperation(false){
          @Override
          protected void maximize(MultiAnnModel model) {
            model.maximizeY();
          }
        };
      case SAMPLE:
        return new SamplingOperation(samples,temp) {
          @Override
          protected void sample(MultiAnnModel model) {
            model.sample();
          }
        };
      case SAMPLEM:
        return new SamplingOperation(samples,temp) {
          @Override
          protected void sample(MultiAnnModel model) {
            model.sampleM();
          }
        };
      case SAMPLEY:
        return new SamplingOperation(samples,temp) {
          @Override
          protected void sample(MultiAnnModel model) {
            model.sampleY();
          }
        };
      default:
        throw new UnsupportedOperationException("Unknown operation type "+type);
      }
    }
  }
  
  /**
   * Iterate until convergence (according to log joint)
   */
  public static abstract class MaximizationOperation implements Operation{
    private boolean alwaysImprovesJoint;
    public MaximizationOperation(boolean strictlyImprovesJoint){
      this.alwaysImprovesJoint=strictlyImprovesJoint;
    }
    public void doOperation(MultiAnnModel model){
      int iterations = 0;
      double prevJoint = model.logJoint();
      logger.info("Initialized Log Joint="+prevJoint);
      double improvement = Double.MAX_VALUE;
      do {
        // optimize
        maximize(model); 
        // check for convergence every few cycles 
        if (iterations % MAXIMIZE_BATCH_SIZE==0){
          double newJoint = model.logJoint();
          logger.info("iteration "+iterations+" Log Joint="+newJoint);
          improvement = newJoint - prevJoint;
          // pfelt: this sanity check breaks when you start doing document weighting
          // (since doc weighting isn't implemented in the logProb method)
          // it also can throw spurious warnings when a method is very close to 
          // convergence 
          if (alwaysImprovesJoint && improvement<0){ // sanity check
            logger.warning(getClass().getName()+": Model got worse ("+improvement+") after maximizing. This should never happen.");
          }
          prevJoint = newJoint;
        }
        ++iterations;
//      } while(iterations < 10);
    } while(improvement > MAXIMIZE_IMPROVEMENT_THRESHOLD && iterations < MAXIMIZE_MAX_ITERATIONS);
      logger.info("Maximization converged after "+iterations+" iterations at "+prevJoint);
    }
    protected abstract void maximize(MultiAnnModel model);
  }

  /**
   * Take given number of samples at given temperature
   */
  public static abstract class SamplingOperation implements Operation{
    private int samples;
    private double temp;
    public SamplingOperation(int numSamples, double annealingTemp){
      this.samples=numSamples;
      this.temp=annealingTemp;
    }
    public void doOperation(MultiAnnModel model){
      model.setTemp(temp);
      for (int s = 0; s < samples; s++) {
        sample(model);
      }
      logger.info("Joint: " + model.logJoint());
    }
    @Override
    public String toString() {
      return samples+" samples at annealing temp: " + temp;
    }
    protected abstract void sample(MultiAnnModel model);
  }
  
  /**
   * Parse and then perform a sequence of colon-delimited training operations with valid values 
   * sample,samplem,sampley,maximize,maximizem,maximizey,none where 
   * sample operations take hyphen-delimited arguments samples:annealingTemp. 
   * For example, samplem-1-1:maximize:maximizey will 
   * take one sample of all the m variables at temp=1, then will do 
   * joint maximization followed by marginal maximization of y.
   */
  public static void doOperations(String ops, MultiAnnModel model){
    logger.info("Training operations "+ops);
    for (String op: ops.split(":")){
      logger.info("Doing training operation "+op);
      OperationParser.parse(op).doOperation(model);
    }
  }

  
}
