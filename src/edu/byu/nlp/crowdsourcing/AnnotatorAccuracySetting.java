/**
 * Copyright 2013 Brigham Young University
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

import java.util.Iterator;

import org.apache.commons.math3.random.RandomGenerator;

import com.google.common.base.Preconditions;

import edu.byu.nlp.stats.DirichletDistribution;
import edu.byu.nlp.util.DoubleArrays;
import edu.byu.nlp.util.Indexer;
import edu.byu.nlp.util.Indexers;

public enum AnnotatorAccuracySetting {
  GOOD (new double[] { 0.99999, 0.99999, 0.99999, 0.99999, 0.99999 }, 1e100), 
  NOISY (new double[] { 0.90, 0.85, 0.80, 0.75, 0.70 }, 1e100), 
  VERY_NOISY (new double[] { 0.70, 0.65, 0.60, 0.55, 0.50 }, 1e100),
  CROWD (new double[] { 0.50, 0.40, 0.30, 0.20, 0.10 }, 1e100),
  CONFLICT (new double[] { 0.50, 0.40, 0.30, 0.20, 0.10 }, 0.1);
  // when this is very large, the off-diagonal is uniform
  private final double symmetricDirichletParam;
  private final double[] accuracies;
  double[][][] confusionMatrices;
  private AnnotatorAccuracySetting(double[] accuracies, double symmetricDirichletParam){
    this.accuracies=accuracies;
    this.symmetricDirichletParam=symmetricDirichletParam;
  }
  public int getNumAnnotators(){
    return accuracies.length;
  }
  public double[] getAccuracies(){
    return accuracies;
  }
  public double[][][] getConfusionMatrices(){
    Preconditions.checkNotNull(confusionMatrices, "call generateConfusionMatrices() first");
    return confusionMatrices;
  }
  public double[][][] generateConfusionMatrices(RandomGenerator rnd, int numLabels){
    if (confusionMatrices==null){
      confusionMatrices = new double[getNumAnnotators()][numLabels][numLabels];
      // a matrix where non-diag entries
      // are sampled and scaled to make each row sum to 1
      for (int a = 0; a < getNumAnnotators(); a++) {
        double rowDiag = accuracies[a];
        for (int i = 0; i < numLabels; i++) {
          // off-diag elements are Dirichlet. Note that when 
          // symmetricDirichletParam is very large, off-diag elements are uniform
          double[] offDiag = DirichletDistribution.sampleSymmetric(symmetricDirichletParam, numLabels - 1, rnd);
          // scale offDiag so row sums to 1
          DoubleArrays.multiplyToSelf(offDiag, 1.0 - rowDiag); 
          Iterator<Double> offDiagItr = DoubleArrays.iterator(offDiag);
          for (int j = 0; j < numLabels; j++) {
            confusionMatrices[a][i][j] = i == j ? rowDiag : offDiagItr.next();
          }
        }
      }
    }

    return confusionMatrices;
  }
  /**
   * A trivial identity indexer where id=index
   */
  public Indexer<Long> getAnnotatorIdIndexer(){
    return Indexers.indexerOfLongs(getNumAnnotators());
  }
}
