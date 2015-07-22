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
package measurements;

import java.util.Map;

import org.apache.commons.math3.random.RandomGenerator;

import edu.byu.nlp.classify.data.DatasetLabeler;
import edu.byu.nlp.crowdsourcing.PriorSpecification;
import edu.byu.nlp.data.types.DatasetInstance;
import edu.byu.nlp.util.IntArrayCounter;

/**
 * @author plf1
 *
 */
public class BasicMeasurementModel implements MeasurementModel{

  /** {@inheritDoc} */
  @Override
  public Double sample(String variableName, int iteration, String[] args) {
    // TODO Auto-generated method stub
    return null;
  }

  /** {@inheritDoc} */
  @Override
  public Double maximize(String variableName, int iteration, String[] args) {
    // TODO Auto-generated method stub
    return null;
  }

  /** {@inheritDoc} */
  @Override
  public DatasetLabeler getIntermediateLabeler() {
    // TODO Auto-generated method stub
    return null;
  }

  /** {@inheritDoc} */
  @Override
  public State getCurrentState() {
    // TODO Auto-generated method stub
    return null;
  }

  /** {@inheritDoc} */
  @Override
  public Map<String, Integer> getInstanceIndices() {
    // TODO Auto-generated method stub
    return null;
  }

  /** {@inheritDoc} */
  @Override
  public IntArrayCounter getMarginalYs() {
    // TODO Auto-generated method stub
    return null;
  }

  /** {@inheritDoc} */
  @Override
  public double logJoint() {
    // TODO Auto-generated method stub
    return 0;
  }

  /** {@inheritDoc} */
  @Override
  public double[] fitOutOfCorpusInstance(DatasetInstance instance) {
    return null; // not interesting for this model
  }
  
  
  
  public static class Builder extends MeasurementModelBuilder{
    /** {@inheritDoc} */
    @Override
    protected MeasurementModel initializeModel(PriorSpecification priors, int[] y, RandomGenerator rnd) {
      // TODO Auto-generated method stub
      return null;
    }
  }




}
