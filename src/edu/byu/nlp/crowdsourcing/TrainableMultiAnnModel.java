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


import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author pfelt
 */
public abstract class TrainableMultiAnnModel implements MultiAnnModel {
  private static final Logger logger = LoggerFactory.getLogger(TrainableMultiAnnModel.class);

  private static final int NUM_SAMPLES_PER_JOINT_CALCULATION = 25;
  public static final int MAXIMIZE_BATCH_SIZE = 3;
  
  public abstract void sample();

  public abstract void sampleY();

  public abstract void sampleM();

  public abstract void maximize();

  public abstract void maximizeY();

  public abstract void maximizeM();

  public abstract void setTemp(double temp);

  /** {@inheritDoc} */
  @Override
  public Double sample(String variableName, int iteration, String[] args) {
    // get args (temp)
    double temp = 1;
    if (args.length >= 1) {
      temp = Double.parseDouble(args[0]);
    }
    setTemp(temp);
    logger.debug("sampling at annealing temp " + temp);

    if (variableName.toLowerCase().equals("y")) {
      sampleY();
    } else if (variableName.toLowerCase().equals("m")) {
      sampleM();
    } else if (variableName.toLowerCase().equals("all")) {
      sample();
    } else {
      throw new IllegalArgumentException(
          "cannot sample from unknown variable name: " + variableName);
    }
    if (iteration%NUM_SAMPLES_PER_JOINT_CALCULATION==0){
      return logJoint();
    }
    return null;
  }

  @Override
  public Double maximize(String variableName, int iteration, String[] args) {
    if (variableName.toLowerCase().equals("y")) {
      maximizeY();
    } else if (variableName.toLowerCase().equals("m")) {
      maximizeM();
    } else if (variableName.toLowerCase().equals("all")) {
      maximize();
    } else {
      throw new IllegalArgumentException(
          "cannot maximize by unknown variable name: " + variableName);
    }
    return logJoint();
  }

}
