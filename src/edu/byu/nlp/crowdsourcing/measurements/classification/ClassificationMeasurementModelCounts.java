/**
 * Copyright 2015 Brigham Young University
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
package edu.byu.nlp.crowdsourcing.measurements.classification;

import java.util.List;

import com.google.common.collect.Lists;

import edu.byu.nlp.crowdsourcing.measurements.classification.ClassificationMeasurementModel.State;
import edu.byu.nlp.stats.MutableSum;

/**
 * @author plf1
 *
 */
public class ClassificationMeasurementModelCounts {

  private List<MutableSum> logNuYSums;
  
  private ClassificationMeasurementModelCounts(){}
  
  public static ClassificationMeasurementModelCounts from(State state){
    ClassificationMeasurementModelCounts counts = new ClassificationMeasurementModelCounts();
    
    // initialize sums
    counts.logNuYSums = Lists.newArrayList();
    for (int c=0; c<state.getNumClasses(); c++){
      counts.logNuYSums.add(new MutableSum());
    }
    
    // calculate initial state
    for (int i=0; i<state.getLogNuY().length; i++){
      counts.setLogNuY(i, state.getLogNuY()[i]);
    }
    
    return counts;
  }
  
  
  public void setLogNuY(int i, double[] value){
    for (int c=0; c<value.length; c++){
      logNuYSums.get(c).setSummand(i, value[i]);
    }
  }
  public double getLogNuY(int c){
    return logNuYSums.get(c).getSum();
  }
  

  public ClassificationMeasurementModelCounts copy() {
    ClassificationMeasurementModelCounts copy = new ClassificationMeasurementModelCounts();
    copy.logNuYSums = copySums(this.logNuYSums);
    return copy;
  }
  
  
  private List<MutableSum> copySums(List<MutableSum> sums){
    List<MutableSum> copy = Lists.newArrayList();
    for (MutableSum sum: sums){
      copy.add(sum.copy());
    }
    return copy;
  }
}
