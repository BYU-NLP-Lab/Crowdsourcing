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
package edu.byu.nlp.crowdsourcing.meanfield;

import java.util.Map;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.fest.assertions.Assertions;
import org.junit.Test;

import com.google.common.collect.Maps;

import edu.byu.nlp.crowdsourcing.CrowdsourcingUtils;
import edu.byu.nlp.crowdsourcing.PriorSpecification;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.util.DoubleArrays;

/**
 * @author pfelt
 */
public class MeanFieldMomRespModelTest {

  // how many times should each stochastic test be run?
  private static final int reps = 100000;
  // start the trials with the following seed (will be incremented)
  private static final int seedStart = 10000;

  // test settings
  private static final int numClasses = 2;
  private static final int numAnnotators = 2;
  private static final int numInstances = 2;
  private static final int numFeatures = 2;
  
  private static MeanFieldMomRespModel buildModel(RandomGenerator rnd, int numClasses, int numAnnotators, int numInstances, int numFeatures){

    // priors
    double bTheta = rnd.nextDouble();
    double bMu = rnd.nextDouble();
    double cMu = rnd.nextDouble()*10;
    double bAlpha = rnd.nextDouble();
    double cAlpha = rnd.nextDouble()*10;
    double bPhi = rnd.nextDouble();
    PriorSpecification priors = new PriorSpecification(bTheta, bMu, cMu, bAlpha, cAlpha, bPhi, false, numAnnotators);
    
    int[][][] a = new int[numInstances][numAnnotators][numClasses];
    for (int i=0; i<numInstances; i++){
      for (int j=0; j<numAnnotators; j++){
        for (int k=0; k<numClasses; k++){
          a[i][j][k] = rnd.nextInt(10);
        }
      }
    }
    double[][] countOfXAndF = new double[numInstances][numFeatures];
    for (int i=0; i<numInstances; i++){
      for (int f=0; f<numFeatures; f++){
        countOfXAndF[i][f] = rnd.nextDouble();
      }
    }
    double[][][] gammaParams = new double[numAnnotators][numClasses][numClasses];
    for (int j=0; j<numAnnotators; j++){
      CrowdsourcingUtils.initializeConfusionMatrixWithPrior(gammaParams[j], priors.getBGamma(j), priors.getCGamma());
    }
    Map<String, Integer> instanceIndices = Maps.newHashMap();
    for (int i=0; i<numInstances; i++){
      instanceIndices.put(""+i, i);
    }
    Dataset data = null;
//    MeanFieldMomRespModel model = new MeanFieldMomRespModel(priors, a, countOfXAndF, gammaParams, instanceIndices, data, rnd);
    MeanFieldMomRespModel model = new MeanFieldMomRespModel(priors, a, gammaParams, instanceIndices, data, rnd);
    
    // init variables
    for (int i=0; i<numInstances; i++){
      for (int k=0; k<numClasses; k++){
        model.vars.logg[i][k] = rnd.nextDouble();
      }
      DoubleArrays.normalizeAndLogToSelf(model.vars.logg[i]);
    }
    for (int k=0; k<numClasses; k++){
      model.vars.pi[k] = rnd.nextDouble()*10;
    }
    for (int j=0; j<numAnnotators; j++){
      for (int k=0; k<numClasses; k++){
        for (int k2=0; k2<numClasses; k2++){
          model.vars.nu[j][k][k2] = rnd.nextDouble()*10;
        }
      }
    }
    for (int k=0; k<numClasses; k++){
      for (int f=0; f<numFeatures; f++){
        model.vars.lambda[k][f] = rnd.nextDouble()*10;
      }
    }
    return model;
  }
  
  
  /**
   * ensure that for a variety of random settings, variational updates do not decrease the logJoint
   */
  @Test
  public void testMaximizeG() {
    
    // g
    for (int test=seedStart; test<seedStart+reps; test++){
      RandomGenerator rnd = new MersenneTwister(test);
      MeanFieldMomRespModel model = buildModel(rnd, numClasses, numAnnotators, numInstances, numFeatures);
      model.fitPi(model.vars.pi);
      model.fitNu(model.vars.nu);
      model.fitLambda(model.vars.lambda);
      double before = model.logJoint();
      model.fitG(model.vars.logg);
      double after = model.logJoint();
      Assertions.assertThat(after).isGreaterThanOrEqualTo(before);
    }
    
  }
  

  /**
   * ensure that for a variety of random settings, variational updates do not decrease the logJoint
   */
  @Test
  public void testMaximizeLambda() {

    // lambda
    for (int test=seedStart; test<seedStart+reps; test++){
      RandomGenerator rnd = new MersenneTwister(test);
      MeanFieldMomRespModel model = buildModel(rnd, numClasses, numAnnotators, numInstances, numFeatures);
      model.fitPi(model.vars.pi);
      model.fitNu(model.vars.nu);
      model.fitG(model.vars.logg);
      double before = model.logJoint();
      model.fitLambda(model.vars.lambda);
      double after = model.logJoint();
      Assertions.assertThat(after).isGreaterThanOrEqualTo(before);
    }
    
  }

  /**
   * ensure that for a variety of random settings, variational updates do not decrease the logJoint
   */
  @Test
  public void testMaximizeNu() {
    
    // nu
    for (int test=seedStart; test<seedStart+reps; test++){
      RandomGenerator rnd = new MersenneTwister(test);
      MeanFieldMomRespModel model = buildModel(rnd, numClasses, numAnnotators, numInstances, numFeatures);
      model.fitPi(model.vars.pi);
      model.fitG(model.vars.logg);
      model.fitLambda(model.vars.lambda);
      double before = model.logJoint();
      model.fitNu(model.vars.nu);
      double after = model.logJoint();
      Assertions.assertThat(after).isGreaterThanOrEqualTo(before);
    }

  }

  /**
   * ensure that for a variety of random settings, variational updates do not decrease the logJoint
   */
  @Test
  public void testMaximizePi() {
    
    // pi
    for (int test=seedStart; test<seedStart+reps; test++){
      RandomGenerator rnd = new MersenneTwister(test);
      MeanFieldMomRespModel model = buildModel(rnd, numClasses, numAnnotators, numInstances, numFeatures);
      model.fitG(model.vars.logg);
      model.fitNu(model.vars.nu);
      model.fitLambda(model.vars.lambda);
      double before = model.logJoint();
      model.fitPi(model.vars.pi);
      double after = model.logJoint();
      Assertions.assertThat(after).isGreaterThanOrEqualTo(before);
    }
    
  }

}
