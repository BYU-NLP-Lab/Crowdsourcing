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

import com.google.common.base.Preconditions;

/**
 * @author pfelt
 */
public abstract class TrainableMultiAnnModel implements MultiAnnModel {
  private static final Logger logger = Logger
      .getLogger(TrainableMultiAnnModel.class.getName());

  public abstract void sample();

  public abstract void sampleY();

  public abstract void sampleM();

  public abstract void maximize();

  public abstract void maximizeY();

  public abstract void maximizeM();

  public abstract void setTemp(double temp);

  @Override
  public void sample(String variableName, String[] args) {
    // get args (num_samples; temp)
    Preconditions
        .checkArgument(args.length >= 1,
            "Must specify number of samples. Sampling until convergence is not supported.");
    int numSamples = Integer.parseInt(args[0]);

    double temp = 1;
    if (args.length >= 2) {
      temp = Double.parseDouble(args[1]);
    }
    setTemp(temp);

    if (variableName.toLowerCase().equals("y")) {
      for (int s = 0; s < numSamples; s++) {
        sampleY();
      }
    } else if (variableName.toLowerCase().equals("m")) {
      for (int s = 0; s < numSamples; s++) {
        sampleM();
      }
    } else if (variableName.toLowerCase().equals("all")) {
      for (int s = 0; s < numSamples; s++) {
        sample();
      }
    } else {
      throw new IllegalArgumentException(
          "cannot sample from unknown variable name: " + variableName);
    }

    logger.info(numSamples + " samples at annealing temp " + temp
        + " gives LogJoint=" + logJoint());
  }

  @Override
  public void maximize(String variableName, String[] args) {
    Preconditions.checkNotNull(args);

    int numIterations = -1;
    // maximize until convergence
    if (args.length == 0) {
      numIterations = maximizeUntilConvergence();
    }
    // maximize for num_iterations
    else {
      numIterations = Integer.parseInt(args[0]);

      if (variableName.toLowerCase().equals("y")) {
        for (int s = 0; s < numIterations; s++) {
          maximizeY();
        }
      } else if (variableName.toLowerCase().equals("m")) {
        for (int s = 0; s < numIterations; s++) {
          maximizeM();
        }
      } else if (variableName.toLowerCase().equals("all")) {
        for (int s = 0; s < numIterations; s++) {
          maximize();
        }
      } else {
        throw new IllegalArgumentException(
            "cannot maximize by unknown variable name: " + variableName);
      }
    }
    logger.info("Maximization for " + numIterations
        + " iterations gives LogJoint=" + logJoint());
  }

  public static final double MAXIMIZE_IMPROVEMENT_THRESHOLD = 1e-6;
  public static final int MAXIMIZE_MAX_ITERATIONS = 20;
  public static final int MAXIMIZE_BATCH_SIZE = 3;

  private int maximizeUntilConvergence() {
    int iterations = 0;
    double prevJoint = logJoint();
    logger.info("Initialized Log Joint=" + prevJoint);
    double improvement = Double.MAX_VALUE;
    do {
      // optimize
      maximize();
      // check for convergence every few cycles
      if (iterations % MAXIMIZE_BATCH_SIZE == 0) {
        double newJoint = logJoint();
        logger.info("iteration " + iterations + " Log Joint=" + newJoint);
        improvement = newJoint - prevJoint;
        // pfelt: this sanity check breaks when you start doing document
        // weighting
        // (since doc weighting isn't implemented in the logProb method)
        // it also can throw spurious warnings when a method is very close to
        // convergence
        if (improvement < 0) { // sanity check
          logger.warning(getClass().getName() + ": Model got worse ("
              + improvement + ") after maximizing. This should never happen.");
        }
        prevJoint = newJoint;
      }
      ++iterations;
      // } while(iterations < 10);
    } while (improvement > MAXIMIZE_IMPROVEMENT_THRESHOLD
        && iterations < MAXIMIZE_MAX_ITERATIONS);
    return iterations;
  }

}
